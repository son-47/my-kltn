'''
Build from https://github.com/anosorae/IRRA
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .triplet import SoftTripletLoss, TopKTripletLoss, _batch_hard, _batch_hard_2, euclidean_dist
import torch.nn.functional as F
from .mim_target import HOGLayerC, L2MIMLoss
from .ema_loss import KLDivLoss



def compute_sdm(t2iscore, pid, logit_scale,  epsilon=1e-8, margin=0, **kwargs):
    """
    Similarity Distribution Matching
    """
    batch_size = t2iscore.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    t2iscore = t2iscore - labels*margin
    text_proj_image = logit_scale * t2iscore
    image_proj_text = logit_scale * t2iscore.t()

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)
    i2t_loss = F.softmax(image_proj_text, dim=1) * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_loss = F.softmax(text_proj_image, dim=1) * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)
    return loss


def compute_ndf(t2iscore, pid, temp=0.015,  epsilon=1e-8, margin=0, **kwargs):
    """
    Similarity Distribution Matching uses Forward KL Divergence and Backward KL Divergence to accelerate distribution fitting.
    """
    batch_size = t2iscore.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    t2iscore = t2iscore - labels*margin
    text_proj_image = torch.exp(t2iscore / temp)  / torch.sum(torch.exp(t2iscore / temp),dim=1, keepdim=True)
    image_proj_text = torch.exp(t2iscore.t() / temp)  / torch.sum(torch.exp(t2iscore.t() / temp),dim=1, keepdim=True)

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)
    i2t_loss = F.softmax(image_proj_text, dim=1) * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon)) \
              + F.softmax(labels_distribute, dim=1) * (F.log_softmax(labels_distribute, dim=1) - torch.log(image_proj_text + epsilon)) 
    t2i_loss = F.softmax(text_proj_image, dim=1) * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon)) \
              + F.softmax(labels_distribute, dim=1) * (F.log_softmax(labels_distribute, dim=1) - torch.log(text_proj_image + epsilon)) 

    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)
    return loss

# def compute_TRL(scores, pid, margin = 0.2,  tau=0.02,  topk=1, margin_weight=None):    
#     batch_size = scores.shape[0]
#     pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
#     pid_dist = pid - pid.t()
#     labels = (pid_dist == 0).float().cuda()
#     mask = 1 - labels
#     if not margin_weight is None: margin = margin_weight * margin

#     alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
#     alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

#     pos_1 = (alpha_1 * scores).sum(1)
#     pos_2 = (alpha_2 * scores.t()).sum(1)

#     neg_1 = (mask*scores).max(1)[0]
#     neg_2 = (mask*scores.t()).max(1)[0]

#     cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
#     cost_2 = (margin + neg_2 - pos_2).clamp(min=0)

#     return cost_1 + cost_2


def compute_TRL_origin(scores, pid, margin = 0.2,  tau=0.02,  topk=1, margin_weight=None):    
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels
    if not margin_weight is None: margin = margin_weight * margin

    # alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    # alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    pos_1 = scores.diagonal()
    pos_2 = scores.t().diagonal()

    neg_1 = (mask*scores).max(1)[0]
    neg_2 = (mask*scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)

    return cost_1 + cost_2


def compute_TRL(scores, pid, tau, margin, hat_label=None, margin_weight=None,  **kwargs):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    hat_label = labels * (1 if hat_label is None else (hat_label > 0.5).float() )
    mask = 1 - labels
    if not margin_weight is None: margin = margin_weight * margin

    alpha_i2t =((scores/tau).exp() * hat_label / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach() #onlyconsider positive pairs
    alpha_t2i = ((scores.t()/tau).exp() * hat_label / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    
    avgpos_i2t = (alpha_i2t*scores).sum(1)
    avgpos_t2i = (alpha_t2i*scores.t()).sum(1)

    loss = (- avgpos_i2t + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (- avgpos_t2i + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)

    return loss 


def compute_ITF(scores, pid,  **kwargs):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels
    klfunction = KLDivLoss().cuda()
    loss = klfunction(scores, labels)

    return loss 


def compute_mlm(scores, labels, reducation='mean'):
    ce = nn.CrossEntropyLoss(ignore_index=0, reduction=reducation).cuda() #skip label = 0 (unmasked label)
    return ce(scores, labels)

def compute_ntlm(scores, labels):
    ce = nn.CrossEntropyLoss(reduction='none').cuda() 
    return ce(scores, labels)


def compute_mim(pred, target, patch_mask, reduce_mean=True, ltype="hog", patch_size=16, norm_pix=False):
    """
    pred: [N, L, p*p*3]  L = patches # masks generated by hogloss
    mask: [N, (W/block size) * (H / block size)], 0 is keep, 1 is remove, 
    """
    #compute loss:
    if ltype == 'l2':
        mim_l2 = L2MIMLoss(patch_size,norm_pix).cuda()
        mim_loss = mim_l2(target, pred, patch_mask)

    elif ltype=='hog':
        # B, N, C = pred.shape
        # H = W = int(N**0.5)
        target = target.permute(0, 2, 3, 1) #BxPxPx(Cxbins)

        target = target.flatten(1, 2)
        pred   = pred.flatten(1).reshape(target.shape)
        patch_mask = patch_mask.flatten(1).bool()
        mim_loss = (pred[patch_mask] - target[patch_mask]) ** 2
    else: raise f"mim loss-{ltype} is not supported!!! "

    if reduce_mean:
        mim_loss = mim_loss.mean()
    return mim_loss



def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()


    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    return loss


def compute_sdm_uncertainty(t2iscore, pid, logit_scale, aleatoric_unc=None,
                              epistemic_unc=None, combined_unc=None,
                              epsilon=1e-8, margin=0, **kwargs):
    """
    Uncertainty-Aware Similarity Distribution Matching (UACS-2).

    Extends SDM by weighting each sample's loss based on estimated uncertainty:
    - High uncertainty → lower weight (model/data uncertain about this sample)
    - Low uncertainty → higher weight (confident prediction)

    This is inspired by Kendall & Gal (2017) where uncertainty is used to
    automatically modulate loss contributions.

    Three uncertainty sources:
    1. Aleatoric: inherent noise in the data (label noise, ambiguity)
    2. Epistemic: model uncertainty (lack of training data in this region)
    3. Combined: weighted combination of both

    The weighting scheme:
    - uncertainty_weight = 1 / (1 + uncertainty)^temperature
    - This smoothly interpolates between full weight (certain) and reduced weight (uncertain)

    Args:
        t2iscore: [B, B] cross-modal similarity matrix
        pid: [B] person IDs
        logit_scale: scalar temperature for softmax
        aleatoric_unc: [B] optional aleatoric uncertainty per sample
        epistemic_unc: [B] optional epistemic uncertainty per sample
        combined_unc: [B] optional pre-combined uncertainty
        epsilon: numerical stability
        margin: margin for positive pairs (from original SDM)

    Returns:
        uncertainty_weighted_loss: [B] per-sample loss weighted by uncertainty
    """
    batch_size = t2iscore.shape[0]
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    t2iscore = t2iscore - labels * margin
    text_proj_image = logit_scale * t2iscore
    image_proj_text = logit_scale * t2iscore.t()
    labels_distribute = labels / labels.sum(dim=1)

    i2t_loss = F.softmax(image_proj_text, dim=1) * (
        F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_loss = F.softmax(text_proj_image, dim=1) * (
        F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    base_loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)

    if combined_unc is not None:
        uncertainty = combined_unc
    elif aleatoric_unc is not None and epistemic_unc is not None:
        uncertainty = 0.5 * aleatoric_unc + 0.5 * epistemic_unc
    elif aleatoric_unc is not None:
        uncertainty = aleatoric_unc
    elif epistemic_unc is not None:
        uncertainty = epistemic_unc
    else:
        return base_loss

    temperature = kwargs.get('unc_temperature', 1.0)
    uncertainty_weight = 1.0 / (1.0 + uncertainty / temperature)

    return base_loss * uncertainty_weight

