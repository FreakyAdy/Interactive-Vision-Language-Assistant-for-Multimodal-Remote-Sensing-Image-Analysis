"""
Unit Tests for RS-InternVL Multi-Sensor Architecture (arXiv:2603.29630).
"""

import sys
import os
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from ml_models.rs_internvl import RSInternVL, PatchEmbedding, ModalityViTEncoder


def test_patch_embedding():
    patch_embed = PatchEmbedding(in_channels=2, embed_dim=64, patch_size=15, img_size=120)
    x = torch.randn(2, 2, 120, 120)
    out = patch_embed(x)
    assert out.shape == (2, 64, 64)  # (B, num_patches=64, embed_dim=64)


def test_vit_encoder():
    encoder = ModalityViTEncoder(in_channels=10, embed_dim=64, depth=2, num_heads=2, img_size=120)
    x = torch.randn(2, 10, 120, 120)
    out = encoder(x)
    assert out.shape == (2, 65, 64)  # (B, 1+num_patches=65, embed_dim=64)


def test_rs_internvl_forward_binary():
    model = RSInternVL(vit_dim=64, llm_dim=128, img_size=120)
    s1 = torch.randn(1, 2, 120, 120)
    s2 = torch.randn(1, 10, 120, 120)
    rgb = torch.randn(1, 3, 120, 120)
    tokens = torch.tensor([[101, 205, 304]], dtype=torch.long)

    out = model(text_tokens=tokens, s1=s1, s2=s2, rgb=rgb, task_type="binary")
    assert "logits" in out
    assert out["logits"].shape == (1, 2)


def test_rs_internvl_predict_binary_vqa():
    model = RSInternVL(vit_dim=64, llm_dim=128, img_size=120)
    s1 = torch.randn(1, 2, 120, 120)
    s2 = torch.randn(1, 10, 120, 120)
    rgb = torch.randn(1, 3, 120, 120)

    ans = model.predict_binary_vqa(s1, s2, rgb, "Does any inland water border inland wetlands in this scene?")
    assert ans in ["Yes", "No"]


def test_rs_internvl_predict_mcq_vqa():
    model = RSInternVL(vit_dim=64, llm_dim=128, img_size=120)
    s1 = torch.randn(1, 2, 120, 120)
    s2 = torch.randn(1, 10, 120, 120)
    rgb = torch.randn(1, 3, 120, 120)

    ans = model.predict_mcq_vqa(s1, s2, rgb, "Which season is shown in the satellite image? a) Spring, b) Summer, c) Winter, d) Autumn")
    assert ans in ["a", "b", "c", "d"]


def test_rs_internvl_predict_referring_bbox():
    model = RSInternVL(vit_dim=64, llm_dim=128, img_size=120)
    s1 = torch.randn(1, 2, 120, 120)
    s2 = torch.randn(1, 10, 120, 120)
    rgb = torch.randn(1, 3, 120, 120)

    bbox = model.predict_referring_bbox(s1, s2, rgb, "Where can the <ref>largest area of urban fabric</ref> be found?")
    assert len(bbox) == 4
    ymin, xmin, ymax, xmax = bbox
    assert 0.0 <= ymin <= ymax <= 1.0
    assert 0.0 <= xmin <= xmax <= 1.0


def test_rs_internvl_generate_caption():
    model = RSInternVL(vit_dim=64, llm_dim=128, img_size=120)
    s1 = torch.randn(1, 2, 120, 120)
    s2 = torch.randn(1, 10, 120, 120)
    rgb = torch.randn(1, 3, 120, 120)

    meta = {"country": "Switzerland", "season": "Summer", "climate_zone": "temperate", "classes": ["arable land"]}
    caption = model.generate_caption(s1, s2, rgb, meta)
    assert "Switzerland" in caption
    assert "Summer" in caption
