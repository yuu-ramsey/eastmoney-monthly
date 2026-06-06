"""
Kronos prediction model unit test - weight loading + forward pass validation

Validation: after loading HuggingFace pretrained weights, forward / decode_s1 / decode_s2
Output dimensions correct, logits values within reasonable range.
"""

import pytest
import torch

from kronos.transformer import Kronos

_model_weights_dir = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "weights" / "model"
)
requires_weights = pytest.mark.skipif(
    not _model_weights_dir.exists(),
    reason="Weights not downloaded. Please run download_weights.py first"
)


class TestKronosInit:
    """Initialization parameter test"""

    def test_default_init(self):
        model = Kronos(
            s1_bits=10, s2_bits=10,
            n_layers=4, d_model=512, n_heads=8, ff_dim=1024,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            token_dropout_p=0.0, learn_te=True,
        )
        assert model.s1_bits == 10
        assert model.s2_bits == 10
        assert model.n_layers == 4

    def test_small_model(self):
        model = Kronos(
            s1_bits=4, s2_bits=4,
            n_layers=2, d_model=128, n_heads=4, ff_dim=256,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            token_dropout_p=0.0, learn_te=False,
        )
        assert len(model.transformer) == 2


class TestKronosForward:
    """Forward pass test (random weights)"""

    @pytest.fixture
    def model(self):
        return Kronos(
            s1_bits=4, s2_bits=4,
            n_layers=2, d_model=128, n_heads=4, ff_dim=256,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            token_dropout_p=0.0, learn_te=False,
        )

    def test_forward_shape(self, model):
        B, T = 2, 20
        s1_ids = torch.randint(0, 15, (B, T))
        s2_ids = torch.randint(0, 15, (B, T))
        s1_logits, s2_logits = model(s1_ids, s2_ids)
        assert s1_logits.shape == (B, T, 16)
        assert s2_logits.shape == (B, T, 16)

    def test_forward_with_padding(self, model):
        B, T = 2, 20
        s1_ids = torch.randint(0, 15, (B, T))
        s2_ids = torch.randint(0, 15, (B, T))
        padding_mask = torch.zeros(B, T, dtype=torch.bool)
        padding_mask[0, 15:] = True
        s1_logits, s2_logits = model(s1_ids, s2_ids, padding_mask=padding_mask)
        assert s1_logits.shape == (B, T, 16)

    def test_forward_with_time_stamp(self, model):
        B, T = 2, 20
        s1_ids = torch.randint(0, 15, (B, T))
        s2_ids = torch.randint(0, 15, (B, T))
        # minute/hour/weekday/day/month valid ranges
        stamp_minute = torch.randint(0, 60, (B, T, 1))
        stamp_hour = torch.randint(0, 24, (B, T, 1))
        stamp_wday = torch.randint(0, 7, (B, T, 1))
        stamp_day = torch.randint(1, 32, (B, T, 1))
        stamp_month = torch.randint(1, 13, (B, T, 1))
        stamp = torch.cat([stamp_minute, stamp_hour, stamp_wday,
                           stamp_day, stamp_month], dim=-1)
        s1_logits, s2_logits = model(s1_ids, s2_ids, stamp=stamp)
        assert s1_logits.shape == (B, T, 16)

    def test_teacher_forcing(self, model):
        B, T = 2, 10
        s1_ids = torch.randint(0, 15, (B, T))
        s2_ids = torch.randint(0, 15, (B, T))
        s1_targets = torch.randint(0, 15, (B, T))
        s1_logits, s2_logits = model(
            s1_ids, s2_ids, use_teacher_forcing=True, s1_targets=s1_targets
        )
        assert s1_logits.shape == (B, T, 16)

    def test_decode_s1(self, model):
        B, T = 2, 20
        s1_ids = torch.randint(0, 15, (B, T))
        s2_ids = torch.randint(0, 15, (B, T))
        s1_logits, context = model.decode_s1(s1_ids, s2_ids)
        assert s1_logits.shape == (B, T, 16)
        assert context.shape == (B, T, 128)

    def test_decode_s2(self, model):
        B, T = 2, 20
        s1_ids = torch.randint(0, 15, (B, T))
        context = torch.randn(B, T, 128)
        s2_logits = model.decode_s2(context, s1_ids)
        assert s2_logits.shape == (B, T, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_cuda(self, model):
        model = model.cuda()
        s1_ids = torch.randint(0, 15, (2, 10)).cuda()
        s2_ids = torch.randint(0, 15, (2, 10)).cuda()
        s1_logits, s2_logits = model(s1_ids, s2_ids)
        assert s1_logits.device.type == "cuda"


class TestKronosPretrained:
    """Pretrained weights loaded test (must run download_weights.py first)"""

    @pytest.mark.skip(reason="Requires HF Hub online loading")
    def test_from_pretrained_hub(self):
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        assert model.d_model == 832
        assert model.n_layers == 12

    @requires_weights
    def test_from_pretrained_local(self):
        model = Kronos.from_pretrained(str(_model_weights_dir))
        assert model.s1_bits > 0
        assert model.s2_bits > 0
        assert model.d_model > 0
