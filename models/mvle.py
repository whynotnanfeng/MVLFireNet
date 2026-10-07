# -*- coding: utf-8 -*-
"""
Multi-Granularity Vision-Language Enhancement (MVLE).

Forest scenes contain many natural elements that resemble fire: reddish autumn
leaves, sunset glow, morning fog, dust. A detector trained on pixels alone has
no way to tell these apart from flame and smoke, and fires false positives on
them. MVLE injects a semantic prior from text to fix that.

This is Section 2.2.4 of the paper. Two pathways align visual features with
textual descriptions at different granularities:

* **global** -- one embedding per image, aligned with the scene-level caption
  (weather, terrain, illumination, vegetation, fire development stage);
* **local** -- one embedding per image, aligned against each box-level caption
  describing the target's colour, edge morphology and occlusion.

The local pathway deliberately shares a single image-level visual embedding
across all boxes in the image instead of cropping per box. Per-box cropping was
measured and gave no accuracy gain while slowing training; image-level alignment
is enough to teach the backbone that fire and smoke look like this regardless of
shape.

Text embeddings come from a **frozen** Long-CLIP text encoder, so this branch
adds no trainable text parameters. The whole branch is discarded at inference:
it influences only the training gradients, and contributes zero parameters,
zero FLOPs and zero latency to the deployed model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUPool(nn.Module):
    """Attention pooling with a gated (SwiGLU) feed-forward network.

    A learnable query token attends over the spatial tokens and the resulting
    single vector is refined by a gated MLP. The gate widens the hidden layer
    (``hidden`` is rounded to a multiple of 64) because a lone query vector
    benefits from the extra nonlinearity.

    Args:
        dim: token dimension.
        nhead: number of attention heads.
        mlp_ratio: hidden-layer expansion before rounding.
    """

    def __init__(self, dim, nhead=4, mlp_ratio=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.sa = nn.MultiheadAttention(dim, nhead, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        hidden = ((int(dim * mlp_ratio * 2 / 3) + 63) // 64) * 64
        self.gate_proj = nn.Linear(dim, hidden)
        self.up_proj = nn.Linear(dim, hidden)
        self.down_proj = nn.Linear(hidden, dim)
        self.norm2 = nn.LayerNorm(dim)
        for m in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, key_padding_mask=None):
        was_2d = (x.dim() == 2)
        if was_2d:
            x = x.unsqueeze(0)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        if key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
        tokens = torch.cat([cls, x], dim=1)
        attn_out = self.sa(tokens, tokens, tokens, key_padding_mask=key_padding_mask)[0]
        cls_out = self.norm1(cls + attn_out[:, 0:1, :])
        residual = cls_out
        gate = F.silu(self.gate_proj(cls_out))
        cls_out = self.down_proj(gate * self.up_proj(cls_out))
        out = self.norm2(residual + cls_out).squeeze(1)
        if was_2d:
            out = out.squeeze(0)
        return out


class MVLEBranch(nn.Module):
    """Dual-pathway visual-language alignment branch.

    The two pathways share the frozen text encoder and a hierarchical text
    projection: the global trunk maps the 768-d CLIP embedding down to the global
    dimension, and the local head maps that same trunk output down to the smaller
    local dimension. Reusing the trunk means local features inherit global scene
    context while still being able to specialise on discriminative detail.

    Args:
        clip_model_path: HuggingFace directory of the Long-CLIP checkpoint.
        sa_dim_global: token dimension of the global pathway.
        sa_dim_local: token dimension of the local pathway.
        use_pos_embed: add a learnable positional embedding before attention.
        temperature: InfoNCE temperature, global pathway.
        temperature_local: InfoNCE temperature, local pathway.
        text_dim: hidden size of the frozen text encoder.
        max_tokens: tokenizer truncation length (248 for Long-CLIP).
    """

    def __init__(self, clip_model_path, sa_dim_global=128, sa_dim_local=64,
                 use_pos_embed=True, temperature=0.1, temperature_local=0.1,
                 text_dim=768, max_tokens=248):
        super().__init__()
        import gc
        from transformers import CLIPModel, CLIPTokenizer

        clip = CLIPModel.from_pretrained(clip_model_path)
        self.text_encoder = clip.text_model
        self.text_projection = clip.text_projection
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
        # Only the text tower is needed.
        del clip.vision_model
        del clip
        gc.collect()
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.text_projection.parameters():
            p.requires_grad = False

        self.max_tokens = max_tokens
        nhead = 4

        # Text projection, Eqs. (14) and (15): a shared trunk reduces the
        # 768-d embedding, and the local head reuses that trunk output.
        self.text_trunk = nn.Linear(text_dim, sa_dim_global)
        self.text_local_head = nn.Linear(sa_dim_global, sa_dim_local)
        nn.init.xavier_uniform_(self.text_trunk.weight)
        nn.init.zeros_(self.text_trunk.bias)
        nn.init.xavier_uniform_(self.text_local_head.weight)
        nn.init.zeros_(self.text_local_head.bias)

        # -- global visual pathway --
        self.proj_global = nn.Conv2d(256, sa_dim_global, 1)
        self.pool_global = SwiGLUPool(sa_dim_global, nhead)
        nn.init.xavier_uniform_(self.proj_global.weight)
        nn.init.zeros_(self.proj_global.bias)

        # -- local visual pathway --
        self.proj_local = nn.Conv2d(256, sa_dim_local, 1)
        self.pool_local = SwiGLUPool(sa_dim_local, nhead)
        nn.init.xavier_uniform_(self.proj_local.weight)
        nn.init.zeros_(self.proj_local.bias)

        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_global = nn.Parameter(torch.randn(512, sa_dim_global) * 0.02)
            self.pos_local = nn.Parameter(torch.randn(512, sa_dim_local) * 0.02)

        self.temperature = temperature
        self.temperature_local = temperature_local

    @torch.no_grad()
    def encode_text(self, texts, device):
        """Encode a list of strings with the frozen text encoder."""
        if not texts:
            return torch.empty(0, self.text_dim_out(), device=device)
        embeds = []
        for i in range(0, len(texts), 64):
            tokens = self.tokenizer(texts[i:i + 64], return_tensors="pt", padding=True,
                                    truncation=True, max_length=self.max_tokens).to(device)
            out = self.text_encoder(**tokens)
            embeds.append(self.text_projection(out.pooler_output).float())
        return torch.cat(embeds, dim=0)

    def text_dim_out(self):
        return self.text_trunk.in_features

    def _info_nce(self, visual_embed, text_embed, temperature):
        """Symmetric InfoNCE over the batch (in-batch negatives only).

        The semantic alignment objective, Eqs. (19) to (20) of the paper.
        """
        visual_embed = F.normalize(visual_embed.float(), dim=-1)
        text_embed = F.normalize(text_embed.float(), dim=-1)
        logits = visual_embed @ text_embed.T / temperature
        labels = torch.arange(len(logits), device=logits.device)
        return (F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)) / 2

    def global_loss(self, c5, global_captions):
        """Semantic alignment loss for the global pathway.

        Follows Eqs. (7), (8) and (10): project to the global space with a 1x1
        convolution, pool with attention, then refine with a SwiGLU feed-forward
        network.

        Args:
            c5: deepest backbone feature map, ``[B, 256, H, W]``.
            global_captions: list of scene-level captions, one per image.
        """
        device = c5.device
        tokens = self.proj_global(c5).flatten(2).permute(0, 2, 1)
        if self.use_pos_embed:
            tokens = tokens + self.pos_global[:tokens.shape[1]]
        visual = self.pool_global(tokens)
        text = self.text_trunk(self.encode_text(list(global_captions), device))
        return self._info_nce(visual, text, self.temperature)

    def local_loss(self, c5, local_captions):
        """Semantic alignment loss for the local pathway.

        The counterpart of :meth:`global_loss` for the local granularity, Eqs.
        (11) to (13). It projects into a narrower space so the pooling focuses
        on discriminative texture rather than broad context.

        The image-level embedding is contrasted against every box caption in the
        batch, so each caption supplies one positive pair.

        Args:
            c5: deepest backbone feature map, ``[B, 256, H, W]``.
            local_captions: list of lists; ``local_captions[b]`` holds the
                box-level captions of image ``b``.
        """
        device = c5.device
        tokens = self.proj_local(c5).flatten(2).permute(0, 2, 1)
        if self.use_pos_embed:
            tokens = tokens + self.pos_local[:tokens.shape[1]]
        image_embed = self.pool_local(tokens)

        visual, captions = [], []
        for b, caps in enumerate(local_captions):
            for caption in caps:
                visual.append(image_embed[b])
                captions.append(caption)

        if not visual:
            return torch.tensor(0.0, device=device)

        visual = torch.stack(visual, dim=0)
        text = self.text_local_head(self.text_trunk(self.encode_text(captions, device)))
        return self._info_nce(visual, text, self.temperature_local)