import torch
import torch.nn as nn
from ..utils.rewards import get_scores, get_self_cider_scores
import torch.nn.functional as F
from captioning.utils.misc import *
import json

torch.autograd.set_detect_anomaly = True


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        if input.ndim == 4:
            input = input.view(input.shape[0], -1, input.shape[-1])
        if seq.ndim == 3:
            seq = seq.view(seq.shape[0], -1)
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        input = input.reshape(-1)
        reward = reward.reshape(-1)
        y = seq.clone()
        mask = (y > 0).to(input)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class StructureLosses(nn.Module):
    """
    This loss is inspired by Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018).
    """

    def __init__(self, opt):
        super(StructureLosses, self).__init__()
        self.opt = opt
        self.loss_type = opt.structure_loss_type

    def forward(self, input, seq, data_gts):
        """
        Input is either logits or log softmax
        """
        out = {}

        batch_size = input.size(0)  # batch_size = sample_size * seq_per_img
        seq_per_img = batch_size // len(data_gts)

        assert seq_per_img == self.opt.train_sample_n, seq_per_img

        mask = (seq > 0).to(input)
        mask = torch.cat([mask.new_full((mask.size(0), 1), 1), mask[:, :-1]], 1)

        scores = get_scores(data_gts, seq, self.opt)
        scores = torch.from_numpy(scores).type_as(input).view(-1, seq_per_img)
        out['reward'] = scores  # .mean()
        if self.opt.entropy_reward_weight > 0:
            entropy = - (F.softmax(input, dim=2) * F.log_softmax(input, dim=2)).sum(2).data
            entropy = (entropy * mask).sum(1) / mask.sum(1)
            print('entropy', entropy.mean().item())
            scores = scores + self.opt.entropy_reward_weight * entropy.view(-1, seq_per_img)
        # rescale cost to [0,1]
        costs = - scores
        if self.loss_type == 'risk' or self.loss_type == 'softmax_margin':
            costs = costs - costs.min(1, keepdim=True)[0]
            costs = costs / costs.max(1, keepdim=True)[0]
        # in principle
        # Only risk need such rescale
        # margin should be alright; Let's try.

        # Gather input: BxTxD -> BxT
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        if self.loss_type == 'seqnll':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)
        elif self.loss_type == 'risk':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1)
            input = input.view(-1, seq_per_img)

            output = (F.softmax(input.exp()) * costs).sum(1).mean()

            # test
            # avg_scores = input
            # probs = F.softmax(avg_scores.exp_())
            # loss = (probs * costs.type_as(probs)).sum() / input.size(0)
            # print(output.item(), loss.item())            

        elif self.loss_type == 'max_margin':
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input).max(1)[0] / 2
            output = output.mean()

            # sanity test
            # avg_scores = input + costs
            # scores_with_high_target = avg_scores.clone()
            # scores_with_high_target.scatter_(1, costs.min(1)[1].view(-1, 1), 1e10)

            # target_and_offender_index = scores_with_high_target.sort(1, True)[1][:, 0:2]
            # avg_scores = avg_scores.gather(1, target_and_offender_index)
            # target_index = avg_scores.new_zeros(avg_scores.size(0), dtype=torch.long)
            # loss = F.multi_margin_loss(avg_scores, target_index, size_average=True, margin=0)
            # print(loss.item() * 2, output.item()) 

        elif self.loss_type == 'multi_margin':
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input)
            output = output.mean()

            # sanity test
            # avg_scores = input + costs
            # loss = F.multi_margin_loss(avg_scores, costs.min(1)[1], margin=0)
            # print(output, loss)

        elif self.loss_type == 'softmax_margin':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)

        elif self.loss_type == 'real_softmax_margin':
            # input is logits
            # This is what originally defined in Kevin's paper
            # The result should be equivalent to softmax_margin
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)

        elif self.loss_type == 'new_self_critical':
            """
            A different self critical
            Self critical uses greedy decoding score as baseline;
            This setting uses the average score of the rest samples as baseline
            (suppose c1...cn n samples, reward1 = score1 - 1/(n-1)(score2+..+scoren) )
            """
            baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)
            scores = scores - baseline
            # self cider used as reward to promote diversity (not working that much in this way)
            if getattr(self.opt, 'self_cider_reward_weight', 0) > 0:
                _scores = get_self_cider_scores(data_gts, seq, self.opt)
                _scores = torch.from_numpy(_scores).type_as(scores).view(-1, 1)
                _scores = _scores.expand_as(scores - 1)
                scores = scores + self.opt.self_cider_reward_weight * _scores
            output = - input * mask * scores.view(-1, 1)
            output = torch.sum(output) / torch.sum(mask)

        out['loss'] = output
        return out


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.cuda = True
        self.adaptive_method = 'exp2'
        self.adaptive_training = True
        self.weight_drop = 0.1
        self.adaptive_T = 1
        self.fp16 = False
        self.eps = 0    # label smoothing, 0 means no label smoothing

        if self.adaptive_training:
            self.weight_drop = self.weight_drop
            freq = []
            nount_weight = []
            with open("/home/ai/data/yss/data/freq.txt") as f:
                data = f.readlines()

            with open("/home/ai/data/yss/data/nounts.txt") as f:
                nounts = f.readlines()

            for index, x in enumerate(data):
                freq.append(int(x.strip()))
                nount_weight.append(int(nounts[index].strip()))

            freq = torch.tensor(freq)
            nount_weight = torch.tensor(nount_weight)
            # mid = freq[int(len(freq) / 2)]
            mid = np.median(freq)
            if self.adaptive_method is 'exp':
                self.weight = [torch.exp(-1 * self.adaptive_T * item / mid) for item in freq]
                b = max(self.weight)
                # self.weight = self.weight / b * (np.e - 1) + 1
                self.weight = [item / b * (np.e - 1) + 1 for item in self.weight]
            else:
                self.weight = [torch.pow(item / mid, torch.tensor(2)) * torch.exp(-1 * self.adaptive_T * item / mid) for item in freq]
                b = max(self.weight)
                self.weight = [item / b * (np.e - 1) + 1 for item in self.weight]
            # self.weight = torch.cat([torch.tensor([1., 1., 1., 1.]), self.weight], dim=0)
            if self.cuda:
                self.weight = torch.tensor(self.weight).cuda()
            if self.fp16:
                self.weight = torch.tensor(self.weight).half()
            self.weight.add_(nount_weight.cuda())
            self.weight = torch.cat([self.weight, torch.tensor([1., 1.]).cuda()], dim=0)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        target = target.view(-1,1)
        non_pad_mask = target.ne(self.padding_idx)
        loss_weight = self.weight[target]
        drop_p = self.weight_drop * torch.ones_like(loss_weight)
        drop_mask = torch.bernoulli(drop_p).byte()
        loss_weight.masked_fill_(drop_mask, 1.)
        nll_loss = -(loss_weight * (lprobs.gather(dim=-1, index=target)))[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def compute_loss_v2(self, output, target, mask, reduce=True):
        lprobs = output.view(-1, output.size(-1))
        target = target.reshape(-1, 1)
        non_pad_mask = mask.reshape(-1, 1).bool()

        loss_weight = self.weight[target]
        drop_p = self.weight_drop * torch.ones_like(loss_weight)
        drop_mask = torch.bernoulli(drop_p).byte()
        loss_weight.masked_fill_(drop_mask, 1.)
        nll_loss = -(loss_weight * (lprobs.gather(dim=-1, index=target)))[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def forward(self, _input, frame_weight, att_weight, frame_surpervise, att_surpervise, target, mask,
                ada_frame_out, ada_att_out, pre_probs, probs):
        # loss, mll_loss = self.compute_loss_v2(_input, target, mask)
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])

        if _input.ndim == 4:
            _input = _input.reshape(-1, _input.shape[-2], _input.shape[-1])

        if len(att_weight) > 0:
            att_target = att_surpervise.reshape(att_surpervise.size(0), att_surpervise.size(1), -1)
            att_target = att_target.unsqueeze(1).expand_as(att_weight)
            att_loss = -torch.mean(torch.masked_select(F.log_softmax(att_weight, dim=3), att_target.bool().cuda()))

        if len(frame_weight) > 0:
            frame_loss = -torch.mean(torch.masked_select(F.log_softmax(frame_weight, dim=2), frame_surpervise.bool().cuda()))

        # truncate to the same size
        target = target[:, :_input.size(1)]
        mask = mask[:, :_input.size(1)].to(_input)

        output = -_input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        # Average over each token
        output = torch.sum(output) / torch.sum(mask)

        if len(att_weight) and len(frame_weight):
            return output + 0.5*att_loss + 0.5*frame_loss, output, frame_loss, att_loss
            # return loss+frame_loss+att_loss, loss, frame_loss, att_loss

        elif len(frame_weight):
            return output+frame_loss, output, frame_loss, output

        elif len(att_weight):
            return output+att_loss, output, output, att_loss

        elif len(pre_probs):
            pre_probs = pre_probs.view(-1, pre_probs.shape[-1])
            probs = probs.view(-1, 1).squeeze(1)
            stop_loss = torch.nn.CrossEntropyLoss()(pre_probs, probs)
            return output + stop_loss, output, stop_loss, stop_loss
        else:
            return output, output, output, output


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1).to(input)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()
