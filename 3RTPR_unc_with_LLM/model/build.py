import logging
from losses import objectives
from losses import ema_loss
from model.clip_model import QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import model.layer as LocalLayer
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import math


    
class DATPS(nn.Module):
    def __init__(self, args, num_classes=11003, name="a"):
        super().__init__()
        self.args = args
        self.name = name
        self.num_classes = num_classes
        self.use_token_selection = self.args.image_encoder.local_branch.enable

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.image_encoder.name, args.image_encoder.img_size, args.image_encoder.stride_size, download_root=args.iocfg.datadir)
        # Trick: freeze patch projection for improved stability
        # https://arxiv.org/pdf/2104.02057.pdf
        for _, v in self.base_model.visual.conv1.named_parameters():
            v.requires_grad_(False)


        self.embed_dim = self.cls_embed_dim = base_cfg['embed_dim']
        self.sratio =  self.args.image_encoder.local_branch.selection_ratio
        self.vtselection = LocalLayer.VisualFusedEmbeddingLayer(input_dim=768, embed_dim=args.image_encoder.local_branch.dim, ratio=self.sratio)
        self.ttselection = LocalLayer.TexualFusedEmbeddingLayer(input_dim=512, embed_dim=args.image_encoder.local_branch.dim, ratio=self.sratio)


        
        self.logit_scale = torch.ones([]) * (1 / args.image_encoder.temperature)

        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)
        self.vision_patch_size = base_cfg['vision_patch_size']
        self._logged_aug_caption = False
        self._logged_aug_masked = False

    #######################################    METHOD SECTION    ####################################################

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
        x,atten_i, org_x = self.base_model.encode_image(image)
        x, _  = self.vtselection(x, org_x, atten_i) 
        return x.float()
    def encode_text_fuse(self, text):
        x,atten_t, org_x = self.base_model.encode_text(text.long())
        x, _ = self.ttselection(x, org_x, text.long(), atten_t)
        return x.float()


    ###MAIN Forward function
    def forward(self, batch):
        #data parse
        images = batch[f'images_{self.name}']
        caption_ids = batch['caption_ids']

        # Use augmented captions from LLM-DA++ with probability aug_ratio
        if 'aug_caption_ids' in batch and batch['aug_caption_ids'] is not None:
            aug_ratio = getattr(self.args, 'aug_ratio', 0.2)
            use_aug = getattr(self.args, 'use_augmented', False)
            if use_aug and aug_ratio > 0:
                batch_size = caption_ids.shape[0]
                mask = torch.rand(batch_size, device=caption_ids.device) < aug_ratio
                if not self._logged_aug_caption:
                    logger = logging.getLogger("DANK!1910.train")
                    logger.info(
                        "Augmented caption mix (%s): replace_ratio=%.3f aug_ratio=%.3f batch_size=%d",
                        self.name,
                        mask.float().mean().item(),
                        float(aug_ratio),
                        int(batch_size),
                    )
                    self._logged_aug_caption = True
                aug_caption_ids = batch['aug_caption_ids']
                # Replace captions with augmented ones where mask is True
                caption_ids = torch.where(mask.unsqueeze(1), aug_caption_ids, caption_ids)

        #text augmented input (MLM masking)
        if self.args.erpt > 0:
            masked_caption_ids = batch[f'masked_caption_ids_{self.name}']
            # If using augmented captions, also need to mask the augmented ones
            if 'aug_caption_ids' in batch and batch['aug_caption_ids'] is not None:
                aug_ratio = getattr(self.args, 'aug_ratio', 0.2)
                use_aug = getattr(self.args, 'use_augmented', False)
                if use_aug and aug_ratio > 0:
                    # Mask the augmented captions as well
                    aug_masked = batch.get(f'masked_aug_caption_ids_{self.name}', masked_caption_ids)
                    batch_size = caption_ids.shape[0]
                    mask = torch.rand(batch_size, device=caption_ids.device) < aug_ratio
                    if not self._logged_aug_masked:
                        logger = logging.getLogger("DANK!1910.train")
                        logger.info(
                            "Augmented masked-caption mix (%s): replace_ratio=%.3f aug_ratio=%.3f batch_size=%d",
                            self.name,
                            mask.float().mean().item(),
                            float(aug_ratio),
                            int(batch_size),
                        )
                        self._logged_aug_masked = True
                    # Replace masked original with masked augmented
                    masked_caption_ids = torch.where(mask.unsqueeze(1), aug_masked, masked_caption_ids)
            caption_ids = masked_caption_ids
            
        #Encode
        #//G
        image_feats, text_feats, image_attscore, text_attscore, image_rfeatures, text_rfeatures = self.base_model(images, caption_ids) #torch.Size([B, tokens, 512]) torch.Size([1, tokens, 512])
        image_fused_feats, li_feats = self.vtselection(image_feats, image_rfeatures, image_attscore)
        text_fused_feats,  lt_feats = self.ttselection(text_feats, text_rfeatures, caption_ids, text_attscore)
           
        logit_scale = self.logit_scale

        return {
            "logit_scale": logit_scale,
            "image_norms_fused_feats" : image_fused_feats / image_fused_feats.norm(dim=-1, keepdim=True), #if self.use_token_selection else None,
            "text_norms_fused_feats" : text_fused_feats / text_fused_feats.norm(dim=-1, keepdim=True),   #if self.use_token_selection else None,
        }


def build_model(args, num_classes=11003, name='a'):
    model = DATPS(args, num_classes, name)
    # covert model to fp16
    convert_weights(model)
    return model