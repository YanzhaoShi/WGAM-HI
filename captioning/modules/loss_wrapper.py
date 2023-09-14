import torch
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward
from captioning.utils import misc as utils


class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)

    def forward(self, fc_feats, att_feats, labels, short_description, masks, short_mask, att_masks, gts, gt_indices,
                sc_flag, struc_flag, frame_surpervise, att_surpervise, probs):
        opt = self.opt

        out = {}
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                                                         opt={'sample_method': opt.train_sample_method,
                                                              'beam_size': opt.train_beam_size,
                                                              'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin' \
                                                                                   or not 'margin' in opt.structure_loss_type,
                                                              'sample_n': opt.train_sample_n},
                                                         mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1 - opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag:
            output, frame_weight, att_weight, ada_frame_out, ada_att_out, pre_probs = self.model(fc_feats, att_feats,
                                                                                         labels[..., :-1], short_description[
                                                                                                   ...,
                                                                                             :15],
                                             att_masks)

            loss, lang_loss, frame_loss, att_loss = self.crit(output,
                                                              frame_weight,
                                                              att_weight,
                                                              frame_surpervise,
                                                              att_surpervise,
                                                              labels[..., 1:],
                                                              masks[..., 1:],
                                                              ada_frame_out,
                                                              ada_att_out, pre_probs, probs)
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _, _ = self.model(fc_feats, att_feats, short_description[..., :15], att_masks,
                                           mode='sample',
                                           opt={'sample_method': opt.sc_sample_method,
                                                'seq_per_img': opt.seq_per_img
                                                })
            self.model.train()
            gen_result, sample_logprobs,_ = self.model(fc_feats, att_feats, short_description[..., :15], att_masks,
                                                     opt={'sample_method': opt.train_sample_method,
                                                          'beam_size': opt.train_beam_size,
                                                          'sample_n': 1,
                                                          'seq_per_img': opt.seq_per_img},
                                                     mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:, ].mean()
        out['loss'] = loss
        out['lang_loss'] = lang_loss
        out['frame_loss'] = frame_loss
        out['att_loss'] = att_loss
        out['outputs'] = output
        return out
