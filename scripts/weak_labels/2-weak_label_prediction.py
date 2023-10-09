'''
This file is a part of 'WGAM-HI\captioning\models\AttModel.py' and is primarily used to showcase the model's predictions on weak labels.
For more details, please see 'WGAM-HI\captioning\models\AttModel.py'
'''

############ predict frame weak labels via Frame_Attention ############
class Frame_Attention(nn.Module):
    def __init__(self, opt):
        super(Frame_Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.fc2p = nn.Linear(self.rnn_size, self.att_hid_size)
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, fc_feats, att_masks=None):
        # The p_att_feats here is already projected
        p_fc_feats = self.fc2p(fc_feats)
        att_size = fc_feats.numel() // fc_feats.size(0) // fc_feats.size(-1)
        fc = p_fc_feats.view(-1, att_size, fc_feats.size(-1))
        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(fc)  # batch * att_size * att_hid_size
        dot = fc + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size
        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).to(weight)
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        fc_feats_ = fc_feats.view(-1, att_size, fc_feats.size(-1))  # batch * att_size * att_feat_size
        fc_res = torch.bmm(weight.unsqueeze(1), fc_feats_).squeeze(1)  # batch * att_feat_size
        return fc_res, weight

############ predict spatial weak labels via Spatial_Attention ############
class Spatial_Attention(nn.Module):
    def __init__(self, opt):
        super(Spatial_Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size
        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).to(weight)
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        return att_res, weight

############ model to predict frame and spatial weak labels ############
class UpDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(UpDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.first_lang_lstm = nn.LSTMCell(15 * opt.input_encoding_size, opt.rnn_size)
        self.sen_lstm = nn.LSTMCell((opt.max_length+1) * opt.input_encoding_size + opt.rnn_size, opt.rnn_size)
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size)
        self.frame_attention = Frame_Attention(opt)
        self.spatial_attention = Spatial_Attention(opt)

    def forward(self, sen_index, word_index, short_description, already_sen, it, xt, fc_feats, att_feats, p_att_feats,
                state,
                scan_weight, att_masks=None):
        att_weight = []
        fc_ = torch.bmm(scan_weight.unsqueeze(1), fc_feats).squeeze(1)
        if sen_index == 0 and word_index == 0:
            h_lang, c_lang = self.first_lang_lstm(short_description.view(fc_feats.size(0), -1),(state[0][1], state[1][1]))
            h_sen = h_lang
            c_sen = c_lang
        else:
            if word_index == 0 or (sen_index == 0 and word_index == 1):
                _, scan_weight = self.frame_attention(state[0][0], fc_feats, att_masks)
                sen_input = torch.cat([already_sen.view(fc_feats.size(0), -1), fc_], 1)
                h_sen, c_sen = self.sen_lstm(sen_input, (state[0][0], state[1][0]))
                ######### frame_weight #################
            else:
                h_sen, c_sen = state[0][0], state[1][0]
            ######### att_weight #################
            att_feats = att_feats.permute(1, 0, 2, 3)
            p_att_feats = p_att_feats.permute(1, 0, 2, 3)
            att_weight = []
            for i in range(att_feats.size(0)):
                _, weight = self.spatial_attention(state[0][1], att_feats[i], p_att_feats[i], att_masks)
                att_weight.append(weight)
            att_weight = torch.cat([_.unsqueeze(1) for _ in att_weight], 1)
            att = torch.bmm(att_weight.view(-1, 196).unsqueeze(1),
                            att_feats.contiguous().view(-1, 196, 512)).squeeze(1).view(-1, 24, 512)
            att = torch.bmm(scan_weight.unsqueeze(1), att).squeeze(1)
            lang_lstm_input = torch.cat([att, h_sen, xt], 1)
            h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_sen,h_lang]), torch.stack([c_sen, c_lang]))
        return output, state, scan_weight, att_weight

class UpDownModel(AttModel):
    def __init__(self, opt):
        super(UpDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = UpDownCore(opt)