import logging
import time
import torch
import math
import numpy as np
from losses import objectives
from model.build import DATPS
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
import random
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import torch.nn.functional as F  
from typing import List
from .masking import * 
from .ccd import *

def _adjust_margin_loss(fixed_margin:float, sim_scores:torch.Tensor, clean_muy:float, a=10, b=4, is_hard=False, mode=None):
    """margin = fixed_margin / (1+e^-(a* sim_scores+b))"""
    is_x_higher_muy = (sim_scores >= clean_muy).float()
    is_muy_positive = (clean_muy>=0).float()
    sim_scores = -abs(sim_scores/(clean_muy+1e-7)) * (1-is_muy_positive) + sim_scores/(clean_muy+1e-7)
    if is_hard:
        if mode==0: sim_scores = ( sim_scores / (clean_muy + 1e-7) ) * a + b
        elif mode==1: sim_scores = sim_scores * 5 #/0.2
        elif mode==2: sim_scores = sim_scores * 8 #
        elif mode==3: sim_scores = sim_scores * 10 #/0.1
        elif mode==4: sim_scores = sim_scores * 12
        elif mode==5: sim_scores = sim_scores * 16 
        elif mode==6: sim_scores = sim_scores * 20 #/0.05
        elif mode==7: sim_scores = sim_scores * 25
        elif mode==8: sim_scores = sim_scores * 50 #/0.02
        elif mode==9: sim_scores = sim_scores * 80
        elif mode==10: sim_scores = sim_scores * 100 #0.01
        elif mode==11: sim_scores = sim_scores * 200 #0.005

        sim_scores  = torch.sigmoid(sim_scores)
        sim_scores = sim_scores * (1-is_x_higher_muy) + is_x_higher_muy
        new_margin = fixed_margin * sim_scores
        return new_margin
    return fixed_margin

def do_train(start_epoch, args, models:List[DATPS], train_loader, evaluator, optimizers:list, schedulers:list, checkpointers:list):

    log_period = args.trainer.log_period
    eval_period = args.trainer.eval_period
    device = "cuda"
    num_epoch = args.trainer.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("DANK!1910.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
    }


    best_top1 = 0.0
    args.cur_step = 0
    stpe = len(train_loader)  #step per epoch
    total_step =  args.total_step = num_epoch * stpe
    current_task = args.losses.loss_names
    logger.info(f'Training Model with {current_task} tasks')

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        if (args.dataloader.dataset_name=='RSTPReid' and epoch > 40 ) or (args.noisy_rate > 0 and epoch > 45): break
        start_time = time.time()
        for meter in meters.values(): meter.reset()
        train_loader.dataset.selecting_samples(None)
        #Calculate the corresponsive consensus division before start training for dataset with N-pairs
        if args.ccd.enable and epoch > args.ccd.warmup_epochs :
            result = get_per_sample_loss(args, models if args.ccd.all_model else [random.choice(models)], train_loader)
            if getattr(args.ccd, 'uncertainty_aware', False):
                # New: Uncertainty-Aware Multi-Signal CCD returns 7 values
                pred_A, pred_B, simAB, conf_A, conf_B, combined_conf, disagreement = result
            else:
                # Legacy: original 3-return signature for backward compatibility
                pred_A, pred_B, simAB = result
                conf_A, conf_B, combined_conf, disagreement = None, None, None, None

            use_soft_weighting = getattr(args.ccd, 'uncertainty_aware', False)

            if use_soft_weighting:
                # ===== SOFT CONFIDENCE WEIGHTING =====
                min_weight = getattr(args.ccd, 'ua_min_weight', 0.05)
                clean_quantile = getattr(args.ccd, 'ua_clean_quantile', 0.8)

                # Use geometric mean directly from ccd.py (not the over-optimistic weighted average).
                # geometric mean penalizes disagreement: conf_A=0.1, conf_B=0.7 → sqrt(0.07)=0.265
                # weighted average was too optimistic: (0.01+0.49)/0.8=0.625
                soft_conf = combined_conf  # already geometric mean from ccd.py + multi-signal boost

                # Super clean set: samples with highest confidence (MUST compute FIRST)
                super_clean_threshold = torch.quantile(soft_conf, clean_quantile)
                super_clean_set = torch.where(soft_conf >= super_clean_threshold)[0]

                # Split samples: super-clean vs non-clean
                # Final weight = totloss_coef only (no squaring)
                # Super-clean: label_hat=1, weight=1 (học full, y hệt GMM cũ)
                # Non-clean: label_hat=1, weight=soft_conf (giảm theo confidence, không bình phương)
                is_super_clean = soft_conf >= super_clean_threshold
                totloss_coef = torch.where(
                    is_super_clean,
                    1.0,  # super-clean: full weight like original GMM
                    soft_conf.clamp(min_weight, 0.9)  # non-clean: soft weight, capped at 0.9
                )
                label_hat = torch.ones_like(soft_conf)  # all samples learn, weight controlled by totloss_coef

                print(f"\t\t\t[UACS] Soft weighting: conf_mean={soft_conf.mean():.3f}, "
                      f"super_clean={len(super_clean_set)} (>{super_clean_threshold:.3f}), "
                      f"non_clean={(~is_super_clean).sum()}, "
                      f"weight_range=[{totloss_coef.min():.3f}, {totloss_coef.max():.3f}]")

            else:
                # ===== ORIGINAL HARD LABEL LOGIC =====
                consensus_division = pred_A + pred_B # 0,1,2
                label_hat = consensus_division.clone() #Nx1
                totloss_coef = simAB.clone()
                super_clean_set = torch.arange(0, len(label_hat), 1)[label_hat == max(label_hat)]

                consensus_division[consensus_division==1] += torch.randint(0, 2, size=(((consensus_division==1)+0).sum(),))
                label_hat[consensus_division>1] = 1
                label_hat[consensus_division<=1] = 0

                if args.ccd.remakeds and not args.losses.dynamic:
                    new_sample_list = torch.where(label_hat==1)[0].tolist()
                    totloss_coef = totloss_coef[label_hat==1]
                    train_loader.dataset.selecting_samples(new_sample_list)
                    label_hat = torch.ones((train_loader.dataset.__len__()))
                else:
                    label_hat += 1
                    label_hat /= label_hat

            if args.losses.dynamic_muy == 'min':
                if len(super_clean_set) > 0:
                    super_clean_set_muy = simAB[super_clean_set].min()
                else:
                    super_clean_set_muy = simAB.min()
            elif args.losses.dynamic_muy == 'mean':
                if len(super_clean_set) > 0:
                    super_clean_set_muy = simAB[super_clean_set].mean()
                else:
                    super_clean_set_muy = simAB.mean()
            else:
                if len(super_clean_set) > 0:
                    super_clean_set_muy = torch.quantile(simAB[super_clean_set], float(args.losses.dynamic_muy) / 100)
                else:
                    super_clean_set_muy = torch.quantile(simAB, float(args.losses.dynamic_muy) / 100)

            if args.losses.dynamic and epoch > args.ccd.warmup_epochs :
                totloss_coef = _adjust_margin_loss(1, totloss_coef, clean_muy=super_clean_set_muy, is_hard=True, mode = abs(args.losses.dynamic_type))
            else:
                if not use_soft_weighting:
                    totloss_coef = torch.ones((train_loader.dataset.__len__()))

            if not use_soft_weighting:
                print("\t\t\t===================Number correct pairs:", (consensus_division / (consensus_division + 1e-8) ).sum(), "over", len(consensus_division),  f"and Smuy = {super_clean_set_muy} ===================") 

        else:
            super_clean_set_muy = 0
            totloss_coef        = torch.ones((train_loader.dataset.__len__()))
            label_hat           = torch.ones((train_loader.dataset.__len__()))
        
        for model in models: model.train()
        for n_iter, samples in enumerate(train_loader):
            rets = {model.name : dict() for model in models}
            cur_step = args.cur_step =  (epoch-1) * stpe + n_iter + 1
            samples['label_hat'] = label_hat[samples['index']]
            samples['totloss_coef'] = totloss_coef[samples['index']]

            for midx, model in enumerate(models):
                batch = {k: v.to(device) for k, v in samples.items()}
                model_output    = model(batch)
                logit_scale     = model_output['logit_scale']
                gInorm_feats    = model_output["image_norms_fused_feats"] #local feature
                gTnorm_feats    = model_output["text_norms_fused_feats"]
                total_loss_weight = batch['totloss_coef']
                gscoret2i = gTnorm_feats @ gInorm_feats.t()
                rets[model.name].update({'temperature': 1 / logit_scale})

                if 'sdm' in current_task:
                    _loss_value = objectives.compute_sdm(gscoret2i, batch['pids'], logit_scale) *  batch['label_hat'] * total_loss_weight #==> only take the sample with 1 label
                    _loss_value = _loss_value.sum() / (batch['label_hat'] .sum() + 1e-8)  * args.losses.sdm_loss_weight
                    rets[model.name].update({'sdm_loss': _loss_value })


            #########################################
                total_loss = sum([v for k, v in rets[model.name].items() if "loss" in k]) 
                optimizers[midx].zero_grad()
                total_loss.backward()
                optimizers[midx].step()

                if args.dev: raise "huhu"
            # LOG synchronize()
            ret = dict()
            for k, v in rets["a"].items(): ret[k] = 0
            for k, v in rets.items(): 
                for kk, vv in v.items(): ret[kk] += vv
            for k, v in ret.items(): ret[k] /= len(rets)
    
            batch_size = batch['images_a'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)


            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if (v.avg > 0):
                        info_str += f", {k}: {v.avg:.3f}"
                info_str += f", Base Lr: {schedulers[0].get_lr()[0]:.3e}"
                logger.info(info_str)

        for scheduler in schedulers: scheduler.step()



        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,train_loader.batch_size / time_per_batch))
        if (((epoch > 10 and args.dataloader.dataset_name!='RSTPReid') or (epoch > 5 and args.dataloader.dataset_name=='RSTPReid') ) \
            and (epoch % eval_period  == 0)) or epoch==start_epoch :
            if get_rank() == 0:
                sims_dict_all = {
                    'GE': 0,
                    'FS': 0,
                    # 'C': 0
                }
                for mmidx, model in enumerate(models):
                    top1, table, sims_dict, (qids, gids) = evaluator.eval([model.eval()], i2t_metric=False, print_log=False, return_all=True)
                    sims_dict_all['GE'] += sims_dict['GE']
                    sims_dict_all['FS'] += sims_dict['FS']
                    # sims_dict_all['C'] += 1
                    torch.cuda.empty_cache()
                    if best_top1 < top1:
                        best_top1 = top1
                        arguments["epoch"] = epoch
                        checkpointers[mmidx].save("best", **arguments)
                    logger.info("\n\t==MODEL {}=> Validation Single Results - Epoch: {} - Top1={} \n ".format(model.name, epoch, top1) + str(table) + "===============")

                top1, table = evaluator.eval([model.eval() for model in models], sims_dict=sims_dict_all, qids=qids, gids=gids,
                                            i2t_metric=False, return_table=True, print_log=False)
                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    for checkpointer in checkpointers: checkpointer.save("best-ensemble", ema=True, **arguments)
                logger.info("\n\t==MODEL MEAN=> Validation Single Results - Epoch: {} - Top1={} \n ".format(epoch, top1) + str(table) + "===============")

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(models, test_img_loader, test_txt_loader):

    logger = logging.getLogger("DANK!1910.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    sims_dict_all = {
        'GE': 0,
        'FS': 0,
        # 'C': 0
    }
    for mmidx, model in enumerate(models):
        top1, table, sims_dict, (qids, gids) = evaluator.eval([model.eval()], i2t_metric=False, print_log=False, return_all=True)
        sims_dict_all['GE'] += sims_dict['GE']
        sims_dict_all['FS'] += sims_dict['FS']
        # sims_dict_all['C'] += 1
        torch.cuda.empty_cache()
        logger.info("\n\t==MODEL {}=> Validation Single Results - Top1={} \n ".format(model.name, top1) + str(table) + "===============")

    top1, table = evaluator.eval([model.eval() for model in models], sims_dict=sims_dict_all, qids=qids, gids=gids,
                                i2t_metric=False, return_table=True, print_log=False)
    torch.cuda.empty_cache()
    logger.info("\n\t==MODEL MEAN=> Validation Single Results  - Top1={} \n ".format(top1) + str(table) + "===============")

