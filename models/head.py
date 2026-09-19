# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .modules import Conv, MMBlock, MGFFN
from utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):

    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape
    
    # Split value per level
    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    
    # Rescale sampling grid from [0, 1] to [-1, 1] for grid_sample
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    
    for level, (h, w) in enumerate(value_spatial_shapes):
        # [bs, H*W, n_head, c] -> [bs, n_head, c, H, W]
        # value arrives as [bs, Len_v, n_head, c] and must be reshaped per level
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * n_head, c, h, w)
        
        # [bs, Len_q, n_head, n_points, 2] -> [bs, n_head, Len_q, n_points, 2] -> [bs*n_head, Len_q, n_points, 2]
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        
        # Bilinear sampling via grid_sample
        # value_l_: [bs*n_head, c, H, W]
        # sampling_grid_l_: [bs*n_head, Len_q, n_points, 2]
        # Compute in FP32: grid_sample produces NaN in FP16 for extreme coordinates
        sampling_value_l_ = F.grid_sample(
            value_l_.float(), sampling_grid_l_.float(),
            mode='bilinear', padding_mode='zeros', align_corners=False
        ).to(value_l_.dtype)
        sampling_value_list.append(sampling_value_l_)
    
    # [bs*n_head, c, Len_q, n_points] -> [bs, n_head, c, Len_q, n_points]
    # Recombine the attention weights
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * n_head, 1, Len_q, n_levels * n_points)
    
    # Stack the sampled values over all levels: [bs*n_head, c, Len_q, n_levels*n_points]
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, n_head * c, Len_q)
    
    return output.transpose(1, 2).contiguous()

class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=3, num_points=4):

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Sampling offsets: one (x, y) pair per query, head, level and point
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        
        # Attention weights
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        
        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Offset initialisation
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        # The bias is initialised to produce a regular initial sampling pattern
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, value, value_spatial_shapes):

        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        
        # 1. Project the values
        value = self.value_proj(value)
        value = value.view(bs, Len_v, self.num_heads, self.head_dim)
        
        # 2. Generate sampling offsets
        sampling_offsets = self.sampling_offsets(query).view(bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        
        # 3. Generate attention weights
        attention_weights = self.attention_weights(query).view(bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        

        attention_weights = F.softmax(attention_weights.float(), -1).type_as(query).view(bs, Len_q, self.num_heads, self.num_levels, self.num_points)
        
        # 4. Sampling locations = reference point + offset
        if reference_points.shape[-1] == 2:
            # 2-point: (cx, cy), offsets normalised by the feature map size
            offset_normalizer = torch.stack([value_spatial_shapes[..., 1], value_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            # 4-point: (cx, cy, w, h), offsets scaled by the box size, so the
            # sampling extent adapts to the object
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] +
                sampling_offsets / self.num_points *
                reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Reference points last dim must be 2 or 4, got {reference_points.shape[-1]}")

        # 5. Core attention computation
        output = multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)
        
        output = self.output_proj(output)
        return output



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = nn.SiLU() if act == 'silu' else nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    out = torch.log(x1 / x2)
    return out.clamp(min=-10, max=10)


def _gen_sineembed_for_memory(spatial_shapes, hidden_dim, device, dtype, temperature=10000):
    """Build sine spatial position embeddings for multi-scale feature maps
    Args:
        spatial_shapes: tensor [n_levels, 2] (h, w)
        hidden_dim: embedding dimension (sin and cos each take half of it)
    Returns:
        pos_embed: [1, total_hw, hidden_dim]
    """
    num_pos_feats = hidden_dim // 2
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pe_list = []
    for lvl in range(spatial_shapes.shape[0]):
        h, w = spatial_shapes[lvl].tolist()
        y = torch.arange(h, dtype=torch.float32, device=device) / h
        x = torch.arange(w, dtype=torch.float32, device=device) / w
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')  # [H, W]

        pos_x = x_grid[:, :, None] * 2 * math.pi / dim_t  # [H, W, num_pos_feats]
        pos_y = y_grid[:, :, None] * 2 * math.pi / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)  # [H, W, num_pos_feats]
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        pe = torch.cat((pos_y, pos_x), dim=-1)  # [H, W, hidden_dim]
        pe_list.append(pe.flatten(0, 1))  # [H*W, hidden_dim]

    return torch.cat(pe_list, dim=0).unsqueeze(0).to(dtype=dtype)  # [1, total_hw, hidden_dim] 


class RTDETRDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.0, n_levels=3, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformableAttention(d_model, nhead, n_levels, n_points)

        self.ffn = MGFFN(d_model, dim_feedforward, kernel_size=3)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, reference_points, spatial_shapes, query_pos=None, attn_mask=None):
        # 1. Self Attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False, attn_mask=attn_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Cross Attention
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos),
            reference_points=reference_points,
            value=memory,
            value_spatial_shapes=spatial_shapes
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 4. FFN
        tgt_for_ffn = tgt.transpose(1, 2).unsqueeze(-1)
        tgt2 = self.ffn(tgt_for_ffn)
        tgt2 = tgt2.squeeze(-1).transpose(1, 2)

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _targets_to_flat_batch(targets):
    """Flatten a list of per-image target dicts into a single batched dict."""
    all_cls = []
    all_bboxes = []
    all_batch_idx = []
    gt_groups = []
    for i, t in enumerate(targets):
        n = len(t["labels"])
        gt_groups.append(n)
        if n > 0:
            all_cls.append(t["labels"])
            all_bboxes.append(t["boxes"])
            all_batch_idx.append(torch.full((n,), i, dtype=torch.long, device=t["labels"].device))
    if len(all_cls) == 0:
        return {"cls": torch.zeros(0, dtype=torch.long), "bboxes": torch.zeros(0, 4),
                "batch_idx": torch.zeros(0, dtype=torch.long), "gt_groups": gt_groups}
    return {
        "cls": torch.cat(all_cls),
        "bboxes": torch.cat(all_bboxes),
        "batch_idx": torch.cat(all_batch_idx),
        "gt_groups": gt_groups,
    }


def get_cdn_group(batch, num_classes, num_queries, class_embed,
                  num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0, training=False):
    """Contrastive Denoising Training
    positive group (correct labels) plus a negative group (flipped labels),
    """
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None
    gt_groups = batch["gt_groups"]
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    bs = len(gt_groups)
    gt_cls = batch["cls"]
    gt_bbox = batch["bboxes"]
    b_idx = batch["batch_idx"]

    # Positive group + negative group (2x)
    dn_cls = gt_cls.repeat(2 * num_group)
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)

    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if cls_noise_ratio > 0:
        mask = torch.rand(dn_cls.shape, device=dn_cls.device) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label

    if box_noise_scale > 0:
        # cxcywh -> xyxy -> noise -> clip -> xyxy -> cxcywh -> inverse_sigmoid
        # Boxes are laid out as flat 2D tensors to match the decoder's expectations
        known_bbox = box_cxcywh_to_xyxy(dn_bbox)
        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        dn_bbox = box_xyxy_to_cxcywh(known_bbox)
        dn_bbox[dn_bbox < 0] *= -1
        dn_bbox = inverse_sigmoid(dn_bbox)

    num_dn_actual = int(max_nums * 2 * num_group)
    dn_cls_embed = class_embed[dn_cls]
    padding_cls = torch.zeros(bs, num_dn_actual, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn_actual, 4, device=gt_bbox.device)

    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)
    map_indices_full = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[(dn_b_idx, map_indices_full)] = dn_cls_embed
    padding_bbox[(dn_b_idx, map_indices_full)] = dn_bbox

    tgt_size = num_dn_actual + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    attn_mask[num_dn_actual:, :num_dn_actual] = True
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums*2*i:max_nums*2*(i+1), max_nums*2*(i+1):num_dn_actual] = True
        if i == num_group - 1:
            attn_mask[max_nums*2*i:max_nums*2*(i+1), :max_nums*i*2] = True
        else:
            attn_mask[max_nums*2*i:max_nums*2*(i+1), max_nums*2*(i+1):num_dn_actual] = True
            attn_mask[max_nums*2*i:max_nums*2*(i+1), :max_nums*2*i] = True

    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn_actual, num_queries],
    }
    return (padding_cls.to(class_embed.device), padding_bbox.to(class_embed.device),
            attn_mask.to(class_embed.device), dn_meta)


class RTDETRDecoder(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=300, nhead=8,
                 num_decoder_layers=3, in_channels=[128, 128, 256],
                 num_denoising=100, cls_noise_ratio=0.5, box_noise_scale=1.0):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_levels = len(in_channels)
        self.num_decoder_layers = num_decoder_layers
        self.num_denoising = num_denoising
        self.cls_noise_ratio = cls_noise_ratio
        self.box_noise_scale = box_noise_scale

        self.input_proj = nn.ModuleList([
            Conv(ch, hidden_dim, k=1, act=False)
            for ch in in_channels
        ])

        # 1. Encoder projection (Linear + LayerNorm)
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Separate classification head, without a background class
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        # 2. Query position head: 4D box -> positional embedding (3 layers + SiLU)
        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, num_layers=3, act='silu')

        # 3. Memory positional encoding (multi-scale sine PE + level embedding)
        self.level_embed = nn.Embedding(self.num_levels, hidden_dim)
        nn.init.uniform_(self.level_embed.weight)

        # 4. Decoder Layers
        self.layers = nn.ModuleList([
            RTDETRDecoderLayer(
                hidden_dim, nhead, dim_feedforward=hidden_dim * 2, dropout=0.0,
                n_levels=self.num_levels, n_points=4
            )
            for _ in range(num_decoder_layers)
        ])

        # 4. Prediction heads, one set per decoder layer
        self.class_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)])
        self.bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_decoder_layers)])

        # 5. Denoising Training
        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim) if num_denoising > 0 else None

        # 6. Encoder box head, separate from the decoder box head
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self._reset_parameters()

    def _reset_parameters(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # The bias is constant in the number of classes: prior_prob = 0.01 gives
        # every query an initial 1% probability of being an object.
        bias_cls = bias_value

        # enc_score_head bias
        nn.init.constant_(self.enc_score_head.bias, bias_cls)
        # enc_output: Linear + LN
        nn.init.xavier_uniform_(self.enc_output[0].weight)
        nn.init.constant_(self.enc_output[0].bias, 0)

        # Zero-initialise the last layer of the box heads: they predict residual
        nn.init.constant_(self.enc_bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.enc_bbox_head.layers[-1].bias.data, 0)
        # Initialise the heads
        for cls_head in self.class_head:
            nn.init.constant_(cls_head.bias, bias_cls)
        for bbox_h in self.bbox_head:
            nn.init.constant_(bbox_h.layers[-1].weight.data, 0)
            nn.init.constant_(bbox_h.layers[-1].bias.data, 0)

        # Xavier init for the query position head
        for layer in self.query_pos_head.layers:
            nn.init.xavier_uniform_(layer.weight)

        # Xavier init for the input projection
        for layer in self.input_proj:
            if hasattr(layer, 'conv'):
                nn.init.xavier_uniform_(layer.conv.weight)


    def _generate_anchors(self, spatial_shapes, grid_size=0.05, device='cpu'):
        """Build per-level anchors and the mask of in-bounds positions.

        Each anchor is (cx, cy, w, h) where:
          cx, cy: normalised grid centre coordinates
          w, h: double with each level (P3 0.05, P4 0.10, P5 0.20)

        Returns:
        anchors: [1, total_HW, 4] in inverse-sigmoid (logit) space
        valid_mask: [1, total_HW, 1], True inside the image
        """
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            h, w = int(h), int(w)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], device=device)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.cat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)
        anchors = torch.cat(anchors, dim=1).to(device)
        valid_mask = ((anchors > 1e-4) * (anchors < 1 - 1e-4)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))  # to logit space
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def forward(self, feats, targets=None):
        """Run the encoder and decoder.

        Args:
            feats: list of ``[B, C, H, W]`` feature maps, finest level first.
            targets: list of per-image target dicts, used for denoising training.

        Returns:
            A dict with the final ``pred_logits`` and ``pred_boxes``, the
            per-layer ``aux_outputs``, and the denoising tensors when
            ``targets`` is provided.
        """
        bs = feats[0].shape[0]

        # 1. Align channels and record the spatial shape of each level
        proj_feats = []
        spatial_shapes = []
        for feat, layer in zip(feats, self.input_proj):
            proj_feats.append(layer(feat))
            spatial_shapes.append(feat.shape[-2:])
        spatial_shapes = torch.tensor(spatial_shapes, device=proj_feats[0].device, dtype=torch.long)

        # 2. Build the memory (values) plus sine PE and level embedding
        memory = torch.cat([feat.flatten(2).transpose(1, 2) for feat in proj_feats], dim=1)

        # Sine PE gives the cross-attention values a spatial reference
        memory_pos = _gen_sineembed_for_memory(
            spatial_shapes, self.hidden_dim, memory.device, memory.dtype)
        # Level embedding distinguishes the feature pyramid levels
        level_pos_list = []
        for lvl in range(self.num_levels):
            h, w = spatial_shapes[lvl].tolist()
            level_pos_list.append(
                self.level_embed.weight[lvl].unsqueeze(0).expand(h * w, -1))
        level_pos = torch.cat(level_pos_list, dim=0).unsqueeze(0)  # [1, total_hw, C]
        memory = memory + memory_pos + level_pos

        # 3. Encoder projection and classification
        enc_features = self.enc_output(memory)           # [B, HW, C]

        # Per-level anchors and the in-bounds mask
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        if bs > 1:
            anchors = anchors.repeat(bs, 1, 1)
            valid_mask = valid_mask.repeat(bs, 1, 1)

        # Mask out anchor positions that fall outside the image
        memory_for_score = valid_mask.to(enc_features.dtype) * enc_features
        enc_outputs_class = self.enc_score_head(memory_for_score)  # [B, HW, nc]

        # Classification-based query selection
        enc_cls = enc_outputs_class.max(-1)[0]  # no background channel
        topk_score, topk_inds = torch.topk(enc_cls, self.num_queries, dim=1)

        batch_idx = torch.arange(bs, device=memory.device).unsqueeze(1)
        tgt = enc_features[batch_idx, topk_inds]      # selected features become queries
        topk_anchors = anchors[batch_idx, topk_inds]  # their anchors

        # 4. Encoder box prediction, expressed as an offset from the anchor
        enc_bbox_embed = self.enc_bbox_head(tgt) + topk_anchors
        enc_outputs_coord = enc_bbox_embed.sigmoid()

        ref_points = enc_outputs_coord.detach()
        enc_outputs_class_selected = enc_outputs_class[batch_idx, topk_inds]

        # 5. Denoising Training
        dn_embed, dn_bbox, attn_mask, dn_meta = None, None, None, None
        num_dn = 0
        if self.training and self.denoising_class_embed is not None and targets is not None:
            flat_batch = _targets_to_flat_batch(targets)
            dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
                flat_batch, self.num_classes, self.num_queries,
                self.denoising_class_embed.weight, self.num_denoising,
                cls_noise_ratio=self.cls_noise_ratio, box_noise_scale=self.box_noise_scale,
                training=True)
            if dn_embed is not None:
                num_dn = dn_embed.shape[1]

        # 6. Concatenate the query groups: [dn_queries, detection_queries]
        if num_dn > 0:
            tgt = torch.cat([dn_embed, tgt], dim=1)
            # get_cdn_group returns logit-space boxes while enc_outputs_coord is in
            # sigmoid space. Move both to [0, 1] so inverse_sigmoid is valid below.
            dn_ref = dn_bbox.detach().sigmoid()
            det_ref = enc_outputs_coord.detach()
            ref_points = torch.cat([dn_ref, det_ref], dim=1)
        # With no denoising queries, tgt and ref_points are left untouched

        # 7. Broadcast the reference points; they stay 4D throughout
        ref_points_input = ref_points.unsqueeze(2).expand(-1, -1, self.num_levels, -1)

        # 8. Decoder loop; the box head is applied to every query, denoising included
        output = tgt
        outputs_class_list, outputs_coord_list = [], []
        last_refined_det = None   # carried across layers to keep the gradient path
        last_refined_dn = None    # same, for the denoising queries

        for i, layer in enumerate(self.layers):
            query_pos = self.query_pos_head(ref_points)

            output = layer(tgt=output, memory=memory,
                           reference_points=ref_points_input,
                           spatial_shapes=spatial_shapes,
                           query_pos=query_pos,
                           attn_mask=attn_mask)

            # Refine boxes; the box head applies to every query
            tmp_box_delta = self.bbox_head[i](output)
            ref_points_inv = inverse_sigmoid(ref_points)
            new_ref_points_inv = ref_points_inv + tmp_box_delta
            new_ref = new_ref_points_inv.sigmoid().clamp(1e-4, 1.0 - 1e-4)

            # Detection queries keep the gradient path via last_refined
            det_new_ref = new_ref[:, num_dn:] if num_dn > 0 else new_ref
            det_delta = tmp_box_delta[:, num_dn:] if num_dn > 0 else tmp_box_delta

            # Denoising queries likewise
            if num_dn > 0:
                dn_new_ref = new_ref[:, :num_dn]
                dn_delta = tmp_box_delta[:, :num_dn]

            # The loss uses last_refined so gradients flow across layers, while
            if self.training:
                if i == 0:
                    outputs_coord_list.append(det_new_ref)
                    if num_dn > 0:
                        last_refined_dn = dn_new_ref
                else:
                    refined = (inverse_sigmoid(last_refined_det) + det_delta).sigmoid().clamp(1e-4, 1.0 - 1e-4)
                    outputs_coord_list.append(refined)
                    if num_dn > 0:
                        # Same gradient chain for the denoising queries
                        last_refined_dn = (inverse_sigmoid(last_refined_dn) + dn_delta).sigmoid().clamp(1e-4, 1.0 - 1e-4)
                last_refined_det = det_new_ref
                # Detach the reference points so they carry no loss gradient
                ref_points = new_ref.detach()
            else:
                outputs_coord_list.append(det_new_ref)
                ref_points = new_ref

            ref_points_input = ref_points.unsqueeze(2).expand(-1, -1, self.num_levels, -1)

            # Per-layer classification heads
            if self.training or i == len(self.layers) - 1:
                det_output = output[:, num_dn:] if num_dn > 0 else output
                det_query = det_output[:, :self.num_queries]
                cls_out = self.class_head[i](det_query)
                outputs_class_list.append(cls_out)

        # 9. Assemble the output dicts
        det_output = output[:, num_dn:] if num_dn > 0 else output
        if not self.training:
            return {
                'pred_logits': outputs_class_list[-1],
                'pred_boxes': outputs_coord_list[-1][:, :self.num_queries],
            }

        outputs_class = torch.stack(outputs_class_list)
        outputs_coord = torch.stack(outputs_coord_list)

        # The encoder output acts as auxiliary layer 0
        enc_aux = {
            'pred_logits': enc_outputs_class_selected,
            'pred_boxes': enc_outputs_coord[:, :self.num_queries],
        }
        dec_aux = self._set_aux_loss(outputs_class, outputs_coord)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1][:, :self.num_queries],
            'aux_outputs': [enc_aux] + dec_aux,
            '_decoder_queries': det_output,
        }
        if dn_meta is not None:
            out['dn_meta'] = dn_meta
            dn_output_last = output[:, :num_dn]
            dn_pred_logits = self.class_head[-1](dn_output_last)
            dn_pred_bboxes = last_refined_dn
            out['dn_bboxes'] = dn_pred_bboxes
            out['dn_scores'] = dn_pred_logits
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        aux = []
        for i in range(outputs_class.shape[0] - 1):
            d = {'pred_logits': outputs_class[i],
                 'pred_boxes': outputs_coord[i][:, :self.num_queries]}
            aux.append(d)
        return aux
