import torch
import numpy as np
from typing import List
from model.build import DATPS
from losses import objectives
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import random
import math


def _split_prob(prob, threshld):
    if prob.min() > threshld:
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)


def _fit_gmm_improved(scores, dataset_name):
    if dataset_name == 'RSTPReid':
        gmm = GaussianMixture(
            n_components=2, max_iter=100, tol=1e-4,
            reg_covar=1e-5, covariance_type='full', n_init=3, random_state=42
        )
    else:
        gmm = GaussianMixture(
            n_components=2, max_iter=50, tol=1e-3,
            reg_covar=1e-4, covariance_type='full', n_init=3, random_state=42
        )
    return gmm


def _compute_per_loss_4ccd(args, models, batch, **kwargs):
    loss_values = [0 for _ in models]
    Gsims = 0
    for midx, model in enumerate(models):
        org_tasks = model.args.losses.loss_names
        org_name = model.name
        model.name = random.choice([i for i in ['a', 'b'] if i != org_name])
        model.args.losses.loss_names = [k for k in org_tasks if k not in ['mim', 'mlm']]
        model_output = model(batch)
        model.args.losses.loss_names = org_tasks
        model.name = org_name

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


def _compute_epistemic_uncertainty(model, batch):
    """
    Estimate epistemic uncertainty via MC Dropout.

    Runs multiple forward passes with dropout enabled and computes
    variance of predictions. High variance = high epistemic uncertainty.
    """
    n_mc = getattr(model, 'n_mc_samples', 5)
    device = next(model.parameters()).device
    B = batch[f'images_{model.name}'].size(0)

    mc_diag_sims = []

    model.train()

    for _ in range(n_mc):
        org_tasks = model.args.losses.loss_names
        org_name = model.name
        model.name = random.choice([i for i in ['a', 'b'] if i != org_name])
        model.args.losses.loss_names = [k for k in org_tasks if k not in ['mim', 'mlm']]

        output = model(batch)

        model.args.losses.loss_names = org_tasks
        model.name = org_name

        img_feat = output["image_norms_fused_feats"]
        txt_feat = output["text_norms_fused_feats"]
        sim = (txt_feat @ img_feat.t()).diagonal().detach()
        mc_diag_sims.append(sim)

    model.eval()

    mc_diag = torch.stack(mc_diag_sims, dim=0)  # [n_mc, B]

    # Epistemic = variance / n_mc (standard error of mean)
    epistemic_unc = mc_diag.var(dim=0) / n_mc

    # Also compute full matrix variance for off-diagonal (negatives)
    mc_full_sims = []
    model.train()
    for _ in range(n_mc):
        org_tasks = model.args.losses.loss_names
        org_name = model.name
        model.name = random.choice([i for i in ['a', 'b'] if i != org_name])
        model.args.losses.loss_names = [k for k in org_tasks if k not in ['mim', 'mlm']]

        output = model(batch)

        model.args.losses.loss_names = org_tasks
        model.name = org_name

        img_feat = output["image_norms_fused_feats"]
        txt_feat = output["text_norms_fused_feats"]
        sim_matrix = txt_feat @ img_feat.t()
        mc_full_sims.append(sim_matrix)

    model.eval()
    mc_full = torch.stack(mc_full_sims, dim=0)  # [n_mc, B, B]

    off_diag_vars = []
    for b in range(B):
        off = mc_full[:, b, :].clone()
        off[:, b] = 0
        off_diag_vars.append(off.var(dim=0).mean())

    off_diag_unc = torch.stack(off_diag_vars) / n_mc

    # Combine diagonal + off-diagonal epistemic
    combined_epistemic = (epistemic_unc + off_diag_unc) / 2
    max_val = combined_epistemic.max() + 1e-8
    combined_epistemic = combined_epistemic / max_val

    return combined_epistemic.cpu()


def _compute_aleatoric_from_sim(sim_matrix):
    """
    Estimate aleatoric uncertainty from cross-modal matching distributions.

    Aleatoric = inherent data ambiguity:
    - Entropy of matching score distribution
    - Cross-modal consistency (T→I vs I→T)
    """
    B = sim_matrix.size(0)
    temperature = 0.02

    sim_probs = torch.softmax(sim_matrix / temperature, dim=1)
    epsilon = 1e-10
    entropy = -torch.sum(sim_probs * torch.log(sim_probs + epsilon), dim=1)
    max_entropy = math.log(B + epsilon)
    normalized_entropy = entropy / max_entropy

    sim_T2I = sim_matrix
    sim_I2T = sim_matrix.t()

    rank_agreement = []
    for i in range(B):
        rank_T2I = torch.argsort(sim_T2I[i], descending=True)
        rank_I2T = torch.argsort(sim_I2T[i], descending=True)
        k = min(10, B)
        topk_T2I = set(rank_T2I[:k].tolist())
        topk_I2T = set(rank_I2T[:k].tolist())
        agreement = len(topk_T2I & topk_I2T) / k
        rank_agreement.append(agreement)

    rank_agreement = torch.tensor(rank_agreement, device=sim_matrix.device)
    aleatoric = (1 - rank_agreement) * 0.5 + normalized_entropy * 0.5

    return aleatoric


def get_per_sample_loss(args, models, data_loader):
    """
    UACS-2: Uncertainty-Aware Multi-Signal CCD with Epistemic Integration.

    Enhancements over original CCD:
    1. Multi-signal GMM: loss + similarity + cross-model agreement
    2. Epistemic uncertainty: MC Dropout variance
    3. Aleatoric uncertainty: entropy of matching distributions
    4. Combined uncertainty-weighted sample confidence

    Returns 11 values (backward compatible with 7):
        0-6: original UACS
        7:   epistemic_combined  [B]
        8:   aleatoric_combined [B]
        9:   total_uncertainty  [B]
        10:  uacs2_conf         [B]
    """
    use_uncertainty_aware = getattr(args.ccd, 'uncertainty_aware', False)
    use_epistemic = getattr(args.ccd, 'use_epistemic', False)
    epistemic_alpha = getattr(args.ccd, 'epistemic_alpha', 0.3)
    dataset_name = models[0].args.dataloader.dataset_name

    w_sim = getattr(args.ccd, 'ua_w_sim', 0.3)
    w_agree = getattr(args.ccd, 'ua_w_agree', 0.2)

    print("\t\t\t===================Calculate Multi-Signal CCD before starting epoch===================")
    for model in models:
        model.eval()
    device = "cuda"
    data_size = data_loader.dataset.__len__()

    lossA = torch.zeros(data_size)
    lossB = torch.zeros(data_size)
    Sims = torch.zeros(data_size)
    CrossAgree = torch.zeros(data_size)
    Sims_A = torch.zeros(data_size)
    Sims_B = torch.zeros(data_size)

    epistemic_A = torch.zeros(data_size)
    epistemic_B = torch.zeros(data_size)
    aleatoric_A = torch.zeros(data_size)
    aleatoric_B = torch.zeros(data_size)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']
            batch_size = index.size(0)

            glosses = []
            gsims_list = []
            simA_list = []
            simB_list = []
            aleatoric_batch_A = []
            aleatoric_batch_B = []
            epistemic_batch_A = []
            epistemic_batch_B = []

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
                gsims_list.append(sim_diag)

                if midx == 0:
                    simA_list.append(sim_diag)
                else:
                    simB_list.append(sim_diag)

                cur_task = args.losses.loss_names
                Gloss = 0
                if 'sdm' in cur_task and args.losses.sdm_loss_weight > 0:
                    Gloss += objectives.compute_sdm(gscore, batch['pids'], logit_scale) * args.losses.sdm_loss_weight

                glosses.append(Gloss.detach().cpu())

                if use_uncertainty_aware or use_epistemic:
                    aleatoric = _compute_aleatoric_from_sim(gscore)
                    if midx == 0:
                        aleatoric_batch_A.append(aleatoric.cpu())
                    else:
                        aleatoric_batch_B.append(aleatoric.cpu())

                if use_epistemic:
                    epistemic = _compute_epistemic_uncertainty(model, batch)
                    if midx == 0:
                        epistemic_batch_A.append(epistemic)
                    else:
                        epistemic_batch_B.append(epistemic)

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

            aleatoric_A_batch = torch.stack(aleatoric_batch_A).mean(
                dim=0) if aleatoric_batch_A else torch.zeros(batch_size)
            aleatoric_B_batch = torch.stack(aleatoric_batch_B).mean(
                dim=0) if aleatoric_batch_B else torch.zeros(batch_size)
            epistemic_A_batch = torch.stack(epistemic_batch_A).mean(
                dim=0) if epistemic_batch_A else torch.zeros(batch_size)
            epistemic_B_batch = torch.stack(epistemic_batch_B).mean(
                dim=0) if epistemic_batch_B else torch.zeros(batch_size)

            for b in range(batch_size):
                idx = index[b].item()
                lossA[idx] = glossa[b]
                lossB[idx] = glossb[b]
                Sims[idx] = gsims[b]
                CrossAgree[idx] = cross_agree[b]
                Sims_A[idx] = simA[b]
                Sims_B[idx] = simB[b]
                aleatoric_A[idx] = aleatoric_A_batch[b]
                aleatoric_B[idx] = aleatoric_B_batch[b]
                epistemic_A[idx] = epistemic_A_batch[b]
                epistemic_B[idx] = epistemic_B_batch[b]

    lossA_norm = ((lossA - lossA.min()) / (lossA.max() - lossA.min() + 1e-9)).reshape(-1, 1)
    lossB_norm = ((lossB - lossB.min()) / (lossB.max() - lossB.min() + 1e-9)).reshape(-1, 1)
    sims_norm = ((Sims - Sims.min()) / (Sims.max() - Sims.min() + 1e-9)).reshape(-1, 1)
    agree_norm = ((CrossAgree - CrossAgree.min()) / (CrossAgree.max() - CrossAgree.min() + 1e-9)).reshape(-1, 1)

    if use_uncertainty_aware:
        w_loss = 1.0
        combined_A = (w_loss * lossA_norm
                     + w_sim * (1 - sims_norm)
                     + w_agree * (1 - agree_norm))
        combined_B = (w_loss * lossB_norm
                     + w_sim * (1 - sims_norm)
                     + w_agree * (1 - agree_norm))
    else:
        combined_A = lossA_norm
        combined_B = lossB_norm

    print('\n\t\t\t===================Fitting Enhanced GMM ...===================')

    gmm_A = _fit_gmm_improved(combined_A.cpu().numpy(), dataset_name)
    gmm_B = _fit_gmm_improved(combined_B.cpu().numpy(), dataset_name)

    gmm_A.fit(combined_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(combined_A.cpu().numpy())
    clean_component_A = gmm_A.means_.argmin()
    conf_A = prob_A[:, clean_component_A]

    gmm_B.fit(gmm_input_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(gmm_input_B.cpu().numpy())
    clean_component_B = gmm_B.means_.argmin()
    conf_B = prob_B[:, clean_component_B]

    pred_A = _split_prob(conf_A, 0.5)
    pred_B = _split_prob(conf_B, 0.5)

    if len(models) >= 2:
        combined_conf = (conf_A * conf_B + 1e-8).sqrt()
        disagreement = ((Sims_A - Sims_B).abs() / (Sims.max() - Sims.min() + 1e-9)).clamp(0, 1)
    else:
        combined_conf = torch.from_numpy(conf_A).clone()
        disagreement = torch.zeros_like(Sims)

    epistemic_combined = (epistemic_A + epistemic_B) / 2
    aleatoric_combined = (aleatoric_A + aleatoric_B) / 2

    if use_epistemic:
        total_uncertainty = (epistemic_alpha * epistemic_combined +
                            (1 - epistemic_alpha) * aleatoric_combined)
        epistemic_boost = (1.0 - epistemic_combined.clamp(0, 1))
        combined_conf_t = torch.from_numpy(combined_conf).clone() if isinstance(combined_conf, np.ndarray) else combined_conf.clone()
        uacs2_conf = combined_conf_t * epistemic_boost

        if use_uncertainty_aware:
            print(f"\t\t\t[UACS-2] Epistemic: mean={epistemic_combined.mean():.3f}, "
                  f"max={epistemic_combined.max():.3f}")
            print(f"\t\t\t[UACS-2] Aleatoric: mean={aleatoric_combined.mean():.3f}, "
                  f"max={aleatoric_combined.max():.3f}")
            print(f"\t\t\t[UACS-2] Total unc: mean={total_uncertainty.mean():.3f}, "
                  f"uacs2_conf: mean={uacs2_conf.mean():.3f}")
    else:
        total_uncertainty = torch.zeros(data_size)
        uacs2_conf = torch.from_numpy(combined_conf).clone() if isinstance(combined_conf, np.ndarray) else combined_conf.clone()

    if use_uncertainty_aware:
        # Ensure conf arrays are torch tensors for downstream processing
        if isinstance(conf_A, np.ndarray):
            conf_A = torch.from_numpy(conf_A).float()
        if isinstance(conf_B, np.ndarray):
            conf_B = torch.from_numpy(conf_B).float()
        print(f"\t\t\t[UACS] A_clean={pred_A.sum()}, B_clean={pred_B.sum()}, "
              f"conf_mean={combined_conf.mean():.3f}, agree_mean={CrossAgree.mean():.3f}, "
              f"disagree_mean={disagreement.mean():.3f}")

    return (torch.Tensor(pred_A), torch.Tensor(pred_B), torch.Tensor(Sims),
            torch.Tensor(conf_A), torch.Tensor(conf_B),
            torch.Tensor(combined_conf), torch.Tensor(disagreement),
            epistemic_combined, aleatoric_combined, total_uncertainty, uacs2_conf)
