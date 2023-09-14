from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.append('/home/ai/data/yss/self_critical/')
# sys.path.append('/home/ai/data/yangsisi/self-critical.pytorch-master/')
import torch

torch.autograd.set_detect_anomaly = True
from torch.utils.tensorboard import SummaryWriter

import traceback
from collections import defaultdict

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
import captioning.utils.eval_utils as eval_utils
import captioning.utils.misc as utils
from captioning.utils.rewards import init_scorer
from captioning.modules.loss_wrapper import LossWrapper
import time
import os

torch.autograd.set_detect_anomaly = True


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def train(opt):
    ################################
    # Build dataloader
    ################################
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##########################
    # Initialize infos
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }
    # Load old infos(if there is) and check if models are compatible
    if opt.start_from is not None and os.path.isfile(
            os.path.join(os.getcwd(), opt.start_from, 'infos_' + opt.id + '.pkl')):
        with open(os.path.join(os.getcwd(), opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt,
                                                                    checkme), "Command line argument and saved model disagree on '%s' " % checkme
    infos['opt'] = opt

    #########################
    # Build logger
    #########################
    # naive dict logger
    histories = defaultdict(dict)
    if opt.start_from is not None and os.path.isfile(
            os.path.join(os.getcwd(), opt.start_from, 'histories_' + opt.id + '.pkl')):
        with open(os.path.join(os.getcwd(), opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
            histories.update(utils.pickle_load(f))

    # tensorboard logger
    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    ##########################
    # Build model
    ##########################
    opt.vocab = loader.get_vocab()
    model = models.setup(opt).cuda()
    del opt.vocab
    # Load pretrained weights:

    if opt.start_from is not None and os.path.isfile(os.path.join(os.getcwd(), opt.start_from, 'model.pth')):
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), opt.start_from, 'model.pth')))

    # Wrap generation model with loss function(used for training)
    # This allows loss function computed separately on each machine
    lw_model = LossWrapper(model, opt)
    dp_model = model
    dp_lw_model = lw_model
    dp_model.vocab = getattr(model, 'vocab', None)  # nasty

    ##########################
    #  Build optimizer
    ##########################
    if opt.noamopt:
        assert opt.caption_model in ['transformer', 'bert', 'm2transformer'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, optim_func=opt.optim, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer,
                                            factor=opt.reduce_on_plateau_factor,
                                            patience=opt.reduce_on_plateau_patience)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if opt.start_from is not None and os.path.isfile(os.path.join(os.getcwd(), opt.start_from,
                                                                  "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(os.getcwd(), opt.start_from, 'optimizer.pth')))

    #########################
    # Get ready to start
    #########################
    iteration = infos['iter']
    epoch = infos['epoch']
    # For back compatibility
    if 'iterators' in infos:
        infos['loader_state_dict'] = {
            split: {'index_list': infos['split_ix'][split], 'iter_counter': infos['iterators'][split]} for split in
            ['train', 'val', 'test']}
    loader.load_state_dict(infos['loader_state_dict'])
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    if opt.noamopt:
        optimizer._step = iteration
    # flag indicating finish of an epoch
    # Always set to True at the beginning to initialize the lr or etc.
    epoch_done = True
    # Assure in training mode
    dp_lw_model.train()

    best_val_score_10, best_val_score_20, best_val_score_30, best_val_score_40, best_val_score_50, best_val_score_60, \
    best_val_score_70, best_val_score_80, best_val_score_90, \
    best_val_score_100 \
        = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Start training
    try:
        while True:
            # Stop if reaching max epochs
            if epoch >= opt.max_epochs:
                break
            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)

            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
                # Assign the scheduled sampling prob
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                else:
                    sc_flag = False

                # If start structure loss training
                if opt.structure_after != -1 and epoch >= opt.structure_after:
                    struc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    struc_flag = False

                epoch_done = False

            start = time.time()
            data = loader.get_batch('train')
            print('Read data:', time.time() - start)

            torch.cuda.synchronize()
            start = time.time()

            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['short_description'], data['masks'],
                   data['short_mask'],
                   data['att_masks'], data['frame_surpervise'], data['att_surpervise'], data['probs']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, short_description, masks, short_mask, att_masks, frame_surpervise, \
            att_surpervise, probs = tmp

            optimizer.zero_grad()
            model_out = dp_lw_model(fc_feats, att_feats, labels, short_description, masks, short_mask, att_masks,
                                    data['gts'],
                                    torch.arange(0, len(data['gts'])), sc_flag,
                                    struc_flag, frame_surpervise, att_surpervise, probs)

            loss = model_out['loss'].mean()
            lang_loss = model_out['lang_loss'].mean().item()
            frame_loss = model_out['frame_loss'].mean().item()
            att_loss = model_out['att_loss'].mean().item()
            # for o in model_out['outputs']:
            #     out_seq = utils.decode_sequence(loader.get_vocab(), torch.argmax(o, 2))
            loss.backward()
            if opt.grad_clip_value != 0:
                getattr(torch.nn.utils, 'clip_grad_%s_' % (opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            end = time.time()
            if struc_flag:
                print(
                    "iter {} (epoch {}), lr ={}, train_loss = {:.3f}, lm_loss = {:.3f}, "
                    "struc_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(iteration, epoch, opt.current_lr, train_loss, model_out['lm_loss'].mean().item(),
                                model_out['struc_loss'].mean().item(), end - start))
            elif not sc_flag:
                print("iter {} (epoch {}), lr = {}, train_loss = {:.3f}, lang_loss = {:.3f}, frame_loss = {:.3f}, "
                      "att_loss = {:.3f},  time/batch = {:.3f}" \
                      .format(iteration, epoch, opt.current_lr, train_loss, lang_loss, frame_loss, att_loss,
                              end - start))

            else:
                print("iter {} (epoch {}),  lr ={}, avg_reward = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, opt.current_lr, model_out['reward'].mean(), end - start))

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                tb_summary_writer.add_scalar('train_loss', train_loss, iteration)
                tb_summary_writer.add_scalar('lang_loss', lang_loss, iteration)
                tb_summary_writer.add_scalar('frame_loss', frame_loss, iteration)
                tb_summary_writer.add_scalar('att_loss', att_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                tb_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
                tb_summary_writer.add_scalar('scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    tb_summary_writer.add_scalar('avg_reward', model_out['reward'].mean(), iteration)
                elif struc_flag:
                    tb_summary_writer.add_scalar('lm_loss', model_out['lm_loss'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('struc_loss', model_out['struc_loss'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('reward', model_out['reward'].mean().item(), iteration)
                    tb_summary_writer.add_scalar('reward_var', model_out['reward'].var(1).mean(), iteration)

                histories['loss_history'][iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                histories['lr_history'][iteration] = opt.current_lr
                histories['ss_prob_history'][iteration] = model.ss_prob

            # # update infos
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['loader_state_dict'] = loader.state_dict()

            # make evaluation on validation set, and save model
            if (iteration % opt.save_checkpoint_every == 0 and not opt.save_every_epoch) or \
                    (epoch_done and opt.save_every_epoch):
                # eval model
                eval_kwargs = {'split': 'val',
                               'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(
                    dp_model, lw_model.crit, loader, eval_kwargs)

                if opt.reduce_on_plateau:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler_step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler_step(val_loss)
                # Write validation result into summary
                tb_summary_writer.add_scalar('validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k, v in lang_stats.items():
                        tb_summary_writer.add_scalar(k, v, iteration)
                histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats,
                                                              'predictions': predictions}

                # Save model if is improving on validation result
                if opt.language_eval == 1:
                    current_score = lang_stats['Bleu_4']
                else:
                    current_score = - val_loss

                best_flag = False

                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score

                utils.save_checkpoint(opt, model, infos, optimizer, histories)
                if opt.save_history_ckpt:
                    utils.save_checkpoint(opt, model, infos, optimizer,
                                          append=str(epoch) if opt.save_every_epoch else str(iteration))

                if best_flag:
                    utils.save_checkpoint(opt, model, infos, optimizer, append='best')

                if epoch < 10:
                    if current_score > best_val_score_10:
                        best_val_score_10 = current_score
                        utils.save_checkpoint(opt, model, infos, optimizer, append='best_10')

                elif 10 <= epoch < 20:
                    if current_score > best_val_score_20:
                        best_val_score_20 = current_score
                        utils.save_checkpoint(opt, model, infos, optimizer, append='best_20')

                elif 20 <= epoch < 30:
                    if current_score > best_val_score_30:
                        best_val_score_30 = current_score
                        utils.save_checkpoint(opt, model, infos, optimizer, append='best_30')

                elif 30 <= epoch < 40:
                    if current_score > best_val_score_40:
                        best_val_score_40 = current_score
                        utils.save_checkpoint(opt, model, infos, optimizer, append='best_40')

                elif 40 <= epoch < 50:
                    if current_score > best_val_score_50:
                        best_val_score_50 = current_score
                        utils.save_checkpoint(opt, model, infos, optimizer, append='best_50')

                elif 50 <= epoch < 60:
                    if current_score > best_val_score_60:
                        best_val_score_60 = current_score
                        utils.save_checkpoint(opt, model, infos, optimizer, append='best_60')

                elif 60 <= epoch < 70:
                    if current_score > best_val_score_70:
                        best_val_score_70 = current_score
                        utils.save_checkpoint(opt, model, infos, optimizer, append='best_70')

                else:
                    if current_score > best_val_score_100:
                        best_val_score_100 = current_score
                        utils.save_checkpoint(opt, model, infos, optimizer, append='best_100')


    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(opt, model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


opt = opts.parse_opt()
train(opt)
