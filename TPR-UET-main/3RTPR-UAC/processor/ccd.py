import torch
import numpy as np
from typing import List
from model.build import DATPS
from losses import objectives
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import random


def _split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)


def _fit_gmm_improved(scores, dataset_name):
    """Fit GMM với improved parameters"""
    if dataset_name == 'RSTPReid':
        gmm = GaussianMixture(
            n_components=2,
            max_iter=100,
            tol=1e-4,
            reg_covar=1e-5,
            covariance_type='full',
            n_init=3,
            random_state=42
        )
    else:
        gmm = GaussianMixture(
            n_components=2,
            max_iter=50,
            tol=1e-3,
            reg_covar=1e-4,
            covariance_type='full',
            n_init=3,
            random_state=42
        )
    return gmm


def _compute_per_loss_4ccd(args, models: List[DATPS], batch, **kwargs):
    loss_values = [0 for _ in models]
    Gsims = 0
    for midx, model in enumerate(models):
        #<-------DONT CARE THIS PART---->
        org_tasks = model.args.losses.loss_names
        org_name = model.name
        model.name = random.choice([i for i in ['a', 'b'] if i != org_name])
        model.args.losses.loss_names = [k for k in org_tasks if not k in ['mim', 'mlm']]
        model_output = model(batch)
        model.args.losses.loss_names = org_tasks
        model.name = org_name
        #<---DONT CARE THE PART ABOVE--->
        logit_scale = model_output['logit_scale']
        gInorm_feats = model_output["image_norms_fused_feats"]
        gTnorm_feats = model_output["text_norms_fused_feats"]
        gscoret2i = gTnorm_feats @ gInorm_feats.t()
        cur_task = args.losses.loss_names
        Gsims += gscoret2i.diagonal().detach().cpu()
        Gloss = 0
        if 'sdm' in cur_task and args.losses.sdm_loss_weight > 0:
            Gloss += objectives.compute_sdm(gscoret2i, batch['pids'], logit_scale) * args.losses.sdm_loss_weight

        loss_values[midx] += Gloss
    return [loss.detach().cpu() for loss in loss_values], (Gsims / len(models)).detach().cpu()


def get_per_sample_loss(args, models, data_loader):
    """
    Uncertainty-Aware Multi-Signal CCD.

    Returns:
        pred_A, pred_B: hard labels (0/1) - for backward compatibility
        simAB: average similarity across models
        conf_A, conf_B: clean probability [0,1] from each model's GMM posterior
        combined_conf: geometric mean of conf_A and conf_B
        disagreement: |pred_A - pred_B| raw (before thresholding)
    """
    use_uncertainty_aware = getattr(args.ccd, 'uncertainty_aware', False)
    dataset_name = models[0].args.dataloader.dataset_name

    # Signal weights from config (with defaults)
    w_sim = getattr(args.ccd, 'ua_w_sim', 0.3)
    w_agree = getattr(args.ccd, 'ua_w_agree', 0.2)

    print("\t\t\t===================Calculate Multi-Signal CCD before starting epoch===================")
    for model in models:
        model.eval()
    device = "cuda"
    data_size = data_loader.dataset.__len__()

    # Multi-signal storage
    lossA = torch.zeros(data_size)
    lossB = torch.zeros(data_size)
    Sims = torch.zeros(data_size)
    CrossAgree = torch.zeros(data_size)

    # Per-model similarity for cross-model agreement
    Sims_A = torch.zeros(data_size)
    Sims_B = torch.zeros(data_size)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']
            batch_size = index.size(0)

            glosses = []
            gsims_list = []
            simA_list = []
            simB_list = []

            for midx, model in enumerate(models):
                org_tasks = model.args.losses.loss_names
                org_name = model.name
                model.name = random.choice([j for j in ['a', 'b'] if j != org_name])
                model.args.losses.loss_names = [k for k in org_tasks if k not in ['mim', 'mlm']]

                model_output = model(batch)

                model.args.losses.loss_names = org_tasks
                model.name = org_name

                logit_scale = model_output['logit_scale']
                gInorm = model_output["image_norms_fused_feats"]
                gTnorm = model_output["text_norms_fused_feats"]
                gscore = gTnorm @ gInorm.t()

                sim_diag = gscore.diagonal().detach().cpu()
                if midx == 0:
                    simA_list.append(sim_diag)
                else:
                    simB_list.append(sim_diag)

                gsims_list.append(sim_diag)

                cur_task = args.losses.loss_names
                Gloss = 0
                if 'sdm' in cur_task and args.losses.sdm_loss_weight > 0:
                    Gloss += objectives.compute_sdm(gscore, batch['pids'], logit_scale) * args.losses.sdm_loss_weight

                glosses.append(Gloss.detach().cpu())

            glossa, glossb = glosses[0], glosses[-1]
            gsims = torch.stack(gsims_list).mean(dim=0)

            if len(models) >= 2:
                simA = simA_list[0]
                simB = simB_list[0]
                cross_agree = (simA * simB + 1e-8).sqrt()
            else:
                simA = simA_list[0]
                simB = simA.clone()
                cross_agree = simA.clone()

            for b in range(batch_size):
                lossA[index[b]] = glossa[b]
                lossB[index[b]] = glossb[b]
                Sims[index[b]] = gsims[b]
                CrossAgree[index[b]] = cross_agree[b]
                Sims_A[index[b]] = simA[b]
                Sims_B[index[b]] = simB[b]

    # ===== Multi-Signal Normalization =====
    lossA_norm = ((lossA - lossA.min()) / (lossA.max() - lossA.min() + 1e-9)).reshape(-1, 1)
    lossB_norm = ((lossB - lossB.min()) / (lossB.max() - lossB.min() + 1e-9)).reshape(-1, 1)
    sims_norm = ((Sims - Sims.min()) / (Sims.max() - Sims.min() + 1e-9)).reshape(-1, 1)
    agree_norm = ((CrossAgree - CrossAgree.min()) / (CrossAgree.max() - CrossAgree.min() + 1e-9)).reshape(-1, 1)

    # ===== Signal Weighting =====
    # Loss weight: high loss -> likely noisy (positive weight)
    # Similarity weight: high sim -> likely clean (negative contribution)
    # Agreement weight: high agreement -> confident (negative contribution)
    if use_uncertainty_aware:
        # Weighted combination: loss dominates, sim/agree help disambiguate
        w_loss = 1.0

        combined_A = (w_loss * lossA_norm
                     + w_sim * (1 - sims_norm)
                     + w_agree * (1 - agree_norm))
        combined_B = (w_loss * lossB_norm
                     + w_sim * (1 - sims_norm)
                     + w_agree * (1 - agree_norm))
    else:
        # Original behavior: only loss
        combined_A = lossA_norm
        combined_B = lossB_norm

    print('\n\t\t\t===================Fitting Enhanced GMM ...===================')

    # Fit improved GMM
    gmm_A = _fit_gmm_improved(combined_A.cpu().numpy(), dataset_name)
    gmm_B = _fit_gmm_improved(combined_B.cpu().numpy(), dataset_name)

    gmm_A.fit(combined_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(combined_A.cpu().numpy())
    # component with smaller mean = cleaner class
    clean_component_A = gmm_A.means_.argmin()
    conf_A = prob_A[:, clean_component_A]

    gmm_B.fit(combined_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(combined_B.cpu().numpy())
    clean_component_B = gmm_B.means_.argmin()
    conf_B = prob_B[:, clean_component_B]

    # Hard predictions for backward compatibility
    pred_A = _split_prob(conf_A, 0.5)
    pred_B = _split_prob(conf_B, 0.5)

    # Combined confidence: geometric mean of two models' posteriors
    # Geometric mean penalizes disagreement more than arithmetic mean
    if len(models) >= 2:
        combined_conf = (conf_A * conf_B + 1e-8).sqrt()
        disagreement = ((Sims_A - Sims_B).abs() / (Sims.max() - Sims.min() + 1e-9)).clamp(0, 1)
    else:
        # Single model: use its confidence, disagreement = 0
        combined_conf = torch.from_numpy(conf_A).clone()
        disagreement = torch.zeros_like(Sims)

    if use_uncertainty_aware:
        print(f"\t\t\t[UACS] A_clean={pred_A.sum()}, B_clean={pred_B.sum()}, "
              f"conf_mean={combined_conf.mean():.3f}, agree_mean={CrossAgree.mean():.3f}, "
              f"disagree_mean={disagreement.mean():.3f}")

    return (torch.Tensor(pred_A), torch.Tensor(pred_B), torch.Tensor(Sims),
            torch.Tensor(conf_A), torch.Tensor(conf_B),
            torch.Tensor(combined_conf), torch.Tensor(disagreement))
