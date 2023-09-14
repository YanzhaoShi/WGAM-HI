from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import json
import os
import sys
import time
from captioning.utils import misc as utils
import matplotlib.pyplot as plt

# load coco-caption if available

try:
    from coco_caption.pycocotools.coco import COCO
    from coco_caption.pycocoevalcap.eval import COCOEvalCap
    from coco_caption.pycocoevalcap.eval_spice import SpiceEval

except:
    print('Warning: coco-caption not available')

bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to', 'is',
               'are', 'am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def getCOCO(dataset):
    if 'coco' in dataset:
        annFile = 'coco-caption/annotations/captions_' \
                  'val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'data/f30k_captions4eval.json'

    return COCO("/home/ai/data/yss/data/long_annotations_clean/brain_ct_data_2048.json")


def language_eval(dataset, preds, preds_n, eval_kwargs, split):
    model_id = eval_kwargs['id']
    eval_oracle = eval_kwargs.get('eval_oracle', 0)

    # create output dictionary
    out = {}

    if len(preds_n) > 0:
        # vocab size and novel sentences
        if 'coco' in dataset:
            dataset_file = 'data/dataset_coco.json'
        elif 'flickr30k' in dataset or 'f30k' in dataset:
            dataset_file = 'data/dataset_flickr30k.json'
        training_sentences = set(
            [' '.join(__['tokens']) for _ in json.load(open(dataset_file))['images'] if
             not _['split'] in ['val', 'test'] for __ in _['sentences']])
        generated_sentences = set([_['caption'] for _ in preds_n])
        novels = generated_sentences - training_sentences
        out['novel_sentences'] = float(len(novels)) / len(preds_n)
        tmp = [_.split() for _ in generated_sentences]
        words = []
        for _ in tmp:
            words += _
        out['vocab_size'] = len(set(words))

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    cache_path = os.path.join('eval_results/', '.cache_' + model_id + '_' + split + '.json')

    coco = getCOCO(dataset)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set
    preds_filt = [p for p in preds if p['image_id'] in valids]

    try:
        mean_perplexity = sum([_['perplexity'] for _ in preds_filt]) / len(preds_filt)
        mean_entropy = sum([_['entropy'] for _ in preds_filt]) / len(preds_filt)
    except:
        mean_perplexity, mean_entropy = 0, 0
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w', encoding='utf-8'),
              ensure_ascii=False)  # serialize to temporary json file. Sigh,
    # COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity
    out['perplexity'] = mean_perplexity
    out['entropy'] = mean_entropy

    imgToEval = cocoEval.imgToEval
    # for k in list(imgToEval.values())[0]['SPICE'].keys():
    #     if k != 'All':
    #         out['SPICE_' + k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
    #         out['SPICE_' + k] = (out['SPICE_' + k][out['SPICE_' + k] == out['SPICE_' + k]]).mean()
    file_handle = open('best.txt', mode='w')
    for p in preds_filt:
        image_id, caption, gts = p['image_id'], p['caption'], p['gts']
        imgToEval[image_id]['caption'] = caption
        imgToEval[image_id]['gts'] = gts
        aa = str(image_id) + ' ' + p['caption']
        file_handle.write(aa)
        file_handle.write('\n')

    if len(preds_n) > 0:
        from . import eval_multi
        cache_path_n = os.path.join('eval_results/', '.cache_' + model_id + '_' + split + '_n.json')
        allspice = eval_multi.eval_allspice(dataset, preds_n, model_id, split)
        out.update(allspice['overall'])
        div_stats = eval_multi.eval_div_stats(dataset, preds_n, model_id, split)
        out.update(div_stats['overall'])
        if eval_oracle:
            oracle = eval_multi.eval_oracle(dataset, preds_n, model_id, split)
            out.update(oracle['overall'])
        else:
            oracle = None
        self_cider = eval_multi.eval_self_cider(dataset, preds_n, model_id, split)
        out.update(self_cider['overall'])
        with open(cache_path_n, 'w') as outfile:
            json.dump({'allspice': allspice, 'div_stats': div_stats, 'oracle': oracle, 'self_cider': self_cider},
                      outfile)

    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile, ensure_ascii=False)
    file_handle.write(json.dumps(out))

    return out


def eval_split(model, crit, loader, eval_kwargs={}):
    start = time.time()
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = 1
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(
        remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration
    device = eval_kwargs.get('device', 'cuda')
    eval_seq = []
    # Make sure in the evaluation mode
    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    space_weight = []
    sequence_weight = []
    n_predictions = []  # when sample_n > 1
    count = 0
    while True:
        data = loader.get_batch(split)
        n = n + len(data['infos'])
        count += 1
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['short_description'][..., :15], data['masks'],
               data['short_mask'],
               data['att_masks'], data['frame_surpervise'], data['att_surpervise'], data['probs']]
        tmp = [_ if _ is None else _.cuda() for _ in tmp]
        fc_feats, att_feats, labels, short_description, masks, short_mask, att_masks, frame_surpervise, \
        att_surpervise, probs = tmp
        if labels is not None and verbose_loss:
            # forward the model to get loss
            with torch.no_grad():
                output, frame_weight, att_weight, ada_frame_out, ada_att_out, pre_probs = model(fc_feats, att_feats,
                                                                                                labels[..., :-1],
                                                                                                short_description[
                                                                                                ..., :15], att_masks)
                loss, _, _, _ = crit(output, frame_weight, att_weight, frame_surpervise,
                                     att_surpervise, labels[..., 1:], masks[..., 1:], ada_frame_out,
                                     ada_att_out, pre_probs, probs)
                # input_index = output.argmax(dim=2)
                # input_decode = utils.decode_sequence(json.load(open("/devdata/Dataset_CT/data2077/data.json"))['ix_to_word'], input_index)
                # target_decode = utils.decode_sequence(json.load(open("/devdata/Dataset_CT/data2077/data.json"))['ix_to_word'], labels[..., 1:])

            loss_sum = loss_sum + loss.item()
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            tmp_eval_kwargs = eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': 1})
            seq, seq_logprobs, weight, frame_weight = model(fc_feats, att_feats, short_description[..., :15],
                                                            att_masks, opt=tmp_eval_kwargs, mode='sample')
            seq = seq.data
            eval_seq.append(seq.tolist())


        # Print be
        # am search
        if beam_size > 1 and verbose_beam:
            for i in range(fc_feats.shape[0]):
                print('\n'.join(
                    [utils.decode_sequence(model.vocab, _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(model.vocab, seq)


        for k, sent in enumerate(sents):
            gts = copy.deepcopy(data['labels'][:, :, 1:-1].contiguous())
            res = sent
            # sen_num = 7
        

            entry = {'image_id': data['infos'][k]['id'], 'caption': res, 'gts': utils.decode_sequence(model.vocab,
                                                                                                      gts)[k],
                     'perplexity': 0, 'entropy': 0}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                            data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
                    len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        if sample_n > 1:
            eval_split_n(model, n_predictions, [fc_feats, att_feats, att_masks, data], eval_kwargs)

        # ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (n, ix1, loss))

        if num_images >= 0 and n >= num_images:
            break

    json.dump(predictions, open('/home/ai/data/yss/self_critical/tools/second_work/visualize/prediction.json', 'w'),
              ensure_ascii=False)
    lang_stats = None

    print('time:', time.time() - start)
    # json.dump(eval_seq, open(eval_json, 'w'))
    lang_stats = None
    if len(n_predictions) > 0 and 'perplexity' in n_predictions[0]:
        n_predictions = sorted(n_predictions, key=lambda x: x['perplexity'])
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    torch.save((predictions, n_predictions),
               os.path.join('eval_results/', '.saved_pred_' + eval_kwargs['id'] + '_' + split + '.pth'))
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, n_predictions, eval_kwargs, split)

    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats


# Only run when sample_n > 0
def eval_split_n(model, n_predictions, input_data, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')

    fc_feats, att_feats, att_masks, data = input_data

    tmp_eval_kwargs = eval_kwargs.copy()
    if sample_n_method == 'bs':
        # case 1 sample_n == beam size
        tmp_eval_kwargs.update({'sample_n': 1, 'beam_size': sample_n, 'group_size': 1})  # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(fc_feats.shape[0]):
            _sents = utils.decode_sequence(model.vocab,
                                           torch.stack([model.done_beams[k][_]['seq'] for _ in range(sample_n)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    # case 2 sample / gumbel / topk sampling/ nucleus sampling
    elif sample_n_method == 'sample' or \
            sample_n_method == 'gumbel' or \
            sample_n_method.startswith('top'):
        tmp_eval_kwargs.update(
            {'sample_n': sample_n, 'sample_method': sample_n_method, 'beam_size': 1})  # randomness from sample
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        _perplexity = - _sampleLogprobs.gather(2, _seq.unsqueeze(2)).squeeze(2).sum(1) / (
                    (_seq > 0).to(_sampleLogprobs).sum(1) + 1)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent,
                     'perplexity': _perplexity[k].item()}
            n_predictions.append(entry)
    elif sample_n_method == 'dbs':
        # Use diverse beam search
        tmp_eval_kwargs.update({'beam_size': sample_n * beam_size, 'group_size': sample_n})  # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(loader.batch_size):
            _sents = utils.decode_sequence(model.vocab,
                                           torch.stack([model.done_beams[k][_]['seq'] for _ in
                                                        range(0, sample_n * beam_size, beam_size)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    else:
        tmp_eval_kwargs.update(
            {'sample_method': sample_n_method[1:], 'group_size': sample_n, 'beam_size': 1})  # randomness from softmax
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
            n_predictions.append(entry)
    if verbose:
        for entry in sorted(n_predictions[-fc_feats.shape[0] * sample_n:], key=lambda x: x['image_id']):
            print('image %s: %s' % (entry['image_id'], entry['caption']))
