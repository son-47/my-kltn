from losses import objectives
from losses import ema_loss
from model.clip_model import QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import model.layer as LocalLayer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


class MCDropoutMLP(nn.Module):
    """
    MLP layer with MC Dropout for epistemic uncertainty estimation.

    When dropout is enabled (model.train()), multiple forward passes
    produce different outputs. The variance of these outputs estimates
    epistemic uncertainty.

    Key insight: dropout at train time approximates Bayesian inference at test time.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = self.dropout(x)
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x


class DATPS(nn.Module):
    def __init__(self, args, num_classes=11003, name="a"):
        super().__init__()
        self.args = args
        self.name = name
        self.num_classes = num_classes
        self.use_token_selection = self.args.image_encoder.local_branch.enable

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.image_encoder.name, args.image_encoder.img_size,
            args.image_encoder.stride_size, download_root=args.iocfg.datadir)
        for _, v in self.base_model.visual.conv1.named_parameters():
            v.requires_grad_(False)

        self.embed_dim = self.cls_embed_dim = base_cfg['embed_dim']
        self.sratio = self.args.image_encoder.local_branch.selection_ratio
        self.vtselection = LocalLayer.VisualFusedEmbeddingLayer(
            input_dim=768, embed_dim=args.image_encoder.local_branch.dim, ratio=self.sratio)
        self.ttselection = LocalLayer.TexualFusedEmbeddingLayer(
            input_dim=512, embed_dim=args.image_encoder.local_branch.dim, ratio=self.sratio)

        mc_dropout_rate = getattr(args.image_encoder, 'mc_dropout_rate', 0.1)
        self.vtselection_mc = MCDropoutMLP(
            input_dim=768, hidden_dim=args.image_encoder.local_branch.dim // 2,
            output_dim=args.image_encoder.local_branch.dim, num_layers=2,
            dropout=mc_dropout_rate)
        self.ttselection_mc = MCDropoutMLP(
            input_dim=512, hidden_dim=args.image_encoder.local_branch.dim // 2,
            output_dim=args.image_encoder.local_branch.dim, num_layers=2,
            dropout=mc_dropout_rate)

        self.logit_scale = torch.ones([]) * (1 / args.image_encoder.temperature)
        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)
        self.vision_patch_size = base_cfg['vision_patch_size']

        self.n_mc_samples = getattr(args.image_encoder, 'n_mc_samples', 5)
        self.aleatoric_alpha = getattr(args.image_encoder, 'aleatoric_alpha', 0.5)

    def cross_former(self, q, k, v, **kwargs):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]

        x = x.permute(1, 0, 2)  # NumxLengthxDim -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        return x

    def encode_image(self, image, att_ret=False):
        x, att, org_x = self.base_model.encode_image(image)
        if att_ret:
            return x, att
        x = x[:, 0, :].float()
        return x

    def encode_text(self, text, att_ret=False):
        x, att, org_x = self.base_model.encode_text(text.long())
        if att_ret:
            return x, att
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        return x

    def encode_image_fuse(self, image):
        x, atten_i, org_x = self.base_model.encode_image(image)
        x, _ = self.vtselection(x, org_x, atten_i)
        return x.float()

    def encode_text_fuse(self, text):
        x, atten_t, org_x = self.base_model.encode_text(text.long())
        x, _ = self.ttselection(x, org_x, text.long(), atten_t)
        return x.float()

    def forward(self, batch, return_uncertainty=False):
        images = batch[f'images_{self.name}']
        caption_ids = batch['caption_ids']
        if self.args.erpt > 0:
            caption_ids = batch[f'masked_caption_ids_{self.name}']

        image_feats, text_feats, image_attscore, text_attscore, image_rfeatures, text_rfeatures = \
            self.base_model(images, caption_ids)

        image_fused_feats, li_feats = self.vtselection(
            image_feats, image_rfeatures, image_attscore)
        text_fused_feats, lt_feats = self.ttselection(
            text_feats, text_rfeatures, caption_ids, text_attscore)

        logit_scale = self.logit_scale

        ret = {
            "logit_scale": logit_scale,
            "image_norms_fused_feats": image_fused_feats / image_fused_feats.norm(dim=-1, keepdim=True),
            "text_norms_fused_feats": text_fused_feats / text_fused_feats.norm(dim=-1, keepdim=True),
        }

        if return_uncertainty:
            ret["raw_image_feats"] = image_fused_feats
            ret["raw_text_feats"] = text_fused_feats
            ret["image_attscore"] = image_attscore
            ret["text_attscore"] = text_attscore

        return ret

    def forward_mc(self, batch):
        """
        Monte Carlo forward pass for epistemic uncertainty estimation.

        Runs multiple forward passes with MC Dropout enabled to estimate
        prediction variance (epistemic uncertainty).

        Args:
            batch: input batch dict

        Returns:
            dict with MC statistics:
                - mc_image_feats: [n_mc, B, D] MC image features
                - mc_text_feats: [n_mc, B, D] MC text features
                - mean_image_feats: [B, D] mean image features
                - mean_text_feats: [B, D] mean text features
                - image_epistemic: [B] per-sample epistemic uncertainty (image)
                - text_epistemic: [B] per-sample epistemic uncertainty (text)
                - combined_epistemic: [B] combined epistemic uncertainty
        """
        device = next(self.parameters()).device
        n_mc = self.n_mc_samples
        B = batch[f'images_{self.name}'].size(0)

        mc_image_list = []
        mc_text_list = []

        self.train()

        for _ in range(n_mc):
            output = self.forward(batch)
            img_feat = output["image_norms_fused_feats"]
            txt_feat = output["text_norms_fused_feats"]
            mc_image_list.append(img_feat)
            mc_text_list.append(txt_feat)

        mc_image = torch.stack(mc_image_list, dim=0)  # [n_mc, B, D]
        mc_text = torch.stack(mc_text_list, dim=0)   # [n_mc, B, D]

        mean_image = mc_image.mean(dim=0)
        mean_text = mc_text.mean(dim=0)

        image_epistemic = mc_image.var(dim=0).mean(dim=1)  # [B]
        text_epistemic = mc_text.var(dim=0).mean(dim=1)   # [B]
        combined_epistemic = (image_epistemic + text_epistemic) / 2

        self.eval()

        return {
            "mc_image_feats": mc_image,
            "mc_text_feats": mc_text,
            "mean_image_feats": mean_image,
            "mean_text_feats": mean_text,
            "image_epistemic": image_epistemic,
            "text_epistemic": text_epistemic,
            "combined_epistemic": combined_epistemic,
        }


def build_model(args, num_classes=11003, name='a'):
    model = DATPS(args, num_classes, name)
    convert_weights(model)
    return model
