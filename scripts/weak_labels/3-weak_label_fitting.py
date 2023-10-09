'''
This file is a part of 'WGAM-HI\captioning\modules\losses.py' and is primarily used to demonstrate how the model fits the predicted weak labels to the ground truth weak labels through optimization of the loss function (line 93-101).
For more details, please see 'WGAM-HI\captioning\modules\losses.py'

'''


import torch
import torch.nn as nn
from ..utils.rewards import get_scores, get_self_cider_scores
import torch.nn.functional as F
from captioning.utils.misc import *
import json

torch.autograd.set_detect_anomaly = True

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
            mid = np.median(freq)
            if self.adaptive_method is 'exp':
                self.weight = [torch.exp(-1 * self.adaptive_T * item / mid) for item in freq]
                b = max(self.weight)
                self.weight = [item / b * (np.e - 1) + 1 for item in self.weight]
            else:
                self.weight = [torch.pow(item / mid, torch.tensor(2)) * torch.exp(-1 * self.adaptive_T * item / mid) for item in freq]
                b = max(self.weight)
                self.weight = [item / b * (np.e - 1) + 1 for item in self.weight]
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

    def forward(self, _input, frame_weight, att_weight, frame_surpervise, att_surpervise, target, mask,
                ada_frame_out, ada_att_out, pre_probs, probs):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])

        if _input.ndim == 4:
            _input = _input.reshape(-1, _input.shape[-2], _input.shape[-1])

        ############ fit the predicted and ground truth spatial weak labels ############
        if len(att_weight) > 0:
            att_target = att_surpervise.reshape(att_surpervise.size(0), att_surpervise.size(1), -1)
            att_target = att_target.unsqueeze(1).expand_as(att_weight)
            att_loss = -torch.mean(torch.masked_select(F.log_softmax(att_weight, dim=3), att_target.bool().cuda()))

        ############ fit the predicted and ground truth frame weak labels ############
        if len(frame_weight) > 0:
            frame_loss = -torch.mean(torch.masked_select(F.log_softmax(frame_weight, dim=2), frame_surpervise.bool().cuda()))

        target = target[:, :_input.size(1)]
        mask = mask[:, :_input.size(1)].to(_input)

        output = -_input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        if len(att_weight) and len(frame_weight): # into this branch
            return output + 0.5*att_loss + 0.5*frame_loss, output, frame_loss, att_loss

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
