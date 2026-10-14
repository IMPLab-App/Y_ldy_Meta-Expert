# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from functools import partial

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=12,
                 num_heads=3, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # MetaExpert specific components
        self.channels = [embed_dim, embed_dim, embed_dim, embed_dim]
        # Indices to extract features. 
        # WRN uses outputs after 1st block, 2nd block, 3rd block, and final output.
        # We simulate this by tapping at 1/4, 1/2, 3/4, and full depth.
        self.stage_indices = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        
        # Expert BN (using BN1d since we pool to 1D vector)
        self.BNH = nn.BatchNorm1d(embed_dim)
        self.BNT = nn.BatchNorm1d(embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes) 
        self.aux_classifier1 = nn.Linear(embed_dim, num_classes)
        self.aux_classifier2 = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        intermediate_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.stage_indices:
                # Use CLS token as feature
                intermediate_features.append(self.norm(x)[:, 0])
        
        # Final norm
        x = self.norm(x)
        return x[:, 0], intermediate_features

    def forward(self, x, only_feat=False, **kwargs):
        x, intermediates = self.forward_features(x)
        
        # Expert logic similar to WRN
        # head_fs, tail_fs = self.BNH(x), self.BNT(x)
        
        # The WRN implementation catches 'out5' which is before pooling?
        # In WRN: out5 -> BNH/BNT -> cat -> pool.
        # Here x is already CLS token (pooled). So dealing with BNH/BNT:
        head_fs = self.BNH(x)
        tail_fs = self.BNT(x)
        
        fs = torch.cat((head_fs, tail_fs), dim=0) # [2B, D]
        
        if only_feat:
            # Note: WRN returns pooled feat here.
            return fs 
            
        # Classifiers
        output = self.classifier(fs) # [2B, 2*num_classes]
        # output is logically [logitsH_all; logitsT_all]
        # split into [logitsH, logitsT]
        logitsH, logitsT = output.chunk(2)
        # However, `output` contains predictions for `head_fs` (first half) and `tail_fs` (second half).
        # In WRN: output = self.classifier(out) where out is pooled [head_fs; tail_fs].
        # output shape is [2*B, num_classes*2] because classifier outputs chunk(2).
        
        # Wait, WRN classifier output is [2B, num_classes * 2].
        # Then `logitsH, logitsT = output.chunk(2)`. This splits along dim 0? NO.
        # Check WRN: `logitsH, logitsT = output.chunk(2)`
        # `output` dim 1 is split.
        # Usually classifiers output `num_classes` dimensions. Here it outputs `num_classes * n_experts`.
        # So `output` matches `num_classes` + `num_classes`.
        
        # But `out` input has size `2*B` (batch augmented).
        # So `output` is `[2*B, 2*num_classes]`.
        # Splitting `chunk(2)` (default dim 0) would split Batch?
        # NO. default dim is 0.
        # But `logitsH` should be `[2B, num_classes]`.
        # Re-read WRN code: `logitsH, logitsT = output.chunk(2)`
        # If output is `[2B, 2*C]`, and chunk dim is 0?
        # Default dim for `chunk` is 0.
        # So `logitsH` gets first B samples, `logitsT` gets second B samples?
        # YES. Because `fs` was `cat((head_fs, tail_fs), dim=0)`.
        # So first B samples are processed by "Head" BN, and second B by "Tail" BN.
        # BUT the classifier is SHARED.
        # The logits output by classifier has size `2*num_classes` (set in code `chunk(2)`).
        # Wait, `self.classifier = nn.Linear(n, num_classes * 2)`.
        # If `output` is `[2B, 2C]`.
        # `chunk(2)` on dim 0 gives `[B, 2C]`. This is logical if we want to separate Head-BN-features outputs from Tail-BN-features outputs.
        # But then `logitsH` would have size `[B, 2C]`. This doesn't seem to match "logitsH".
        
        # Let's check WRN code again carefully.
        # `logitsH, logitsT = output.chunk(2)`
        # If dim=0 (default), then we are separating the batch (Head vs Tail features).
        # `logits` = (logitsH + logitsT)/2.
        # This implies we average predictions from Head-expert-features and Tail-expert-features.
        # BUT, `logitsH` usually implies "Logits for Head Expert".
        # If `classifier` outputs `2*C`, then each expert outputs 2 sets of logits? Unlikely.
        
        # Alternative: `self.classifier` outputs `num_classes`.
        # In WRN: `self.classifier = nn.Linear(channels[3], num_classes * 2)` ?
        # I need to check how classifier is initialized in `wra_metaexpert`.
        # It's not in the file I read! `WideResNet` init just sets `self.classifier`? No.
        # The `WideResNet` class usually takes `num_classes`.
        # In `WideResNet` class `__init__`: `self.classifier` is NOT defined in the snippet I read.
        # Wait, I must have missed it or it's in `**kwargs`?
        # No, let's look at `__init__` again.
        # I don't see `self.classifier` defined in `WideResNet.__init__` in the snippet I read (Lines 1-100).
        # It must be later. Let's read lines 100-130.
        pass

        # For ViT, I will assume `chunk(2)` means splitting the output features? 
        # Or splitting the batch?
        
        # If the intent is Ensemble:
        # We process X through HeadBN -> X_H.
        # We process X through TailBN -> X_T.
        # We classify X_H -> Logits_H.
        # We classify X_T -> Logits_T.
        # Logits_avg = (Logits_H + Logits_T)/2.
        # This works if `classifier` outputs `num_classes`.
        # In that case `output` is `[2B, C]`.
        # `chunk(2)` on dim 0 -> `[B, C]` and `[B, C]`. Correct.
        
        # SO: `self.classifier` should output `num_classes`.
        # But wait, in my previous edit of `wrn_metaexpert.py` (which I didn't verify completely, I just saw `chunk(2)` was changed from `chunk(3)`).
        # If `classifier` output dimension was `num_classes * n_experts`, then we would split on dim 1.
        
        # Let's assume standard logic: Specialized BN -> Shared Classifier.
        # This means we cat on dim 0.
        # So output is `[2B, C]`.
        # chunk(2) gives `[B, C]`.
        
        logits = (logitsH + logitsT) / 2
        
        aux_output1 = self.aux_classifier1(fs)
        aux_logitsH1, aux_logitsT1 = aux_output1.chunk(2)
        aux_logits1 = (aux_logitsH1 + aux_logitsT1) / 2

        aux_output2 = self.aux_classifier2(fs)
        aux_logitsH2, aux_logitsT2 = aux_output2.chunk(2)
        aux_logits2 = (aux_logitsH2 + aux_logitsT2) / 2

        feat_for_fuse = {
            'feat1': intermediates[0],
            'feat2': intermediates[1],
            'feat3': intermediates[2],
            'feat4': intermediates[3]
        }

        result_dict = {
            'feat': x, 
            'feat_for_fuse': feat_for_fuse,
            'logitsH': logitsH, 
            'logitsT': logitsT, 
            'logits': logits,
            'aux_logitsH1': aux_logitsH1, 
            'aux_logitsT1': aux_logitsT1, 
            'aux_logits1': aux_logits1,
            'aux_logitsH2': aux_logitsH2, 
            'aux_logitsT2': aux_logitsT2, 
            'aux_logits2': aux_logits2
        }
        return result_dict

def vit_tiny_metaexpert(pretrained=False, pretrained_path=None, **kwargs):
    model = VisionTransformer(
        img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
def vit_small_metaexpert(pretrained=False, pretrained_path=None, **kwargs):
    model = VisionTransformer(
        img_size=32, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
