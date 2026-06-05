"""
KronosPredictor 端到端推理测试 — 合成数据验证

验证标准：用合成 K 线数据跑完整推理流程，输出 DataFrame 列和形状正确。
"""

import numpy as np
import pandas as pd
import pytest
import torch

from kronos.predictor import (
    KronosPredictor,
    auto_regressive_inference,
    calc_time_stamps,
    sample_from_logits,
    top_k_top_p_filtering,
)
from kronos.tokenizer import KronosTokenizer
from kronos.transformer import Kronos


class TestTopKTopPFiltering:
    """采样过滤函数测试"""

    def test_top_k_filtering(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.1]])
        filtered = top_k_top_p_filtering(logits.clone(), top_k=2)
        inf_count = (filtered == -float("Inf")).sum().item()
        assert inf_count == 3

    def test_top_p_filtering(self):
        logits = torch.tensor([[10.0, 1.0, 0.1, 0.05, 0.01]])
        filtered = top_k_top_p_filtering(logits.clone(), top_p=0.9)
        assert filtered[0, 0] > -float("Inf") / 2

    def test_no_filtering(self):
        logits = torch.randn(1, 100)
        filtered = top_k_top_p_filtering(logits.clone(), top_k=0, top_p=0.0)
        assert torch.equal(filtered, logits)


class TestSampleFromLogits:
    """采样函数测试"""

    def test_deterministic_with_low_temp(self):
        torch.manual_seed(42)
        logits = torch.tensor([[10.0, 1.0, 0.1]])
        samples = torch.stack([
            sample_from_logits(logits, temperature=0.1)
            for _ in range(50)
        ])
        assert (samples == 0).float().mean() > 0.8

    def test_batch_sampling(self):
        torch.manual_seed(42)
        logits = torch.randn(4, 100)
        samples = sample_from_logits(logits, temperature=1.0, top_k=10)
        assert samples.shape == (4,)
        assert samples.max() < 100


class TestCalcTimeStamps:
    """时间戳构建测试"""

    def test_monthly_stamps(self):
        stamps = calc_time_stamps(
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-06-01"),
            pred_len=6, freq="M"
        )
        assert stamps.shape == (1, 6, 5)

    def test_weekly_stamps(self):
        stamps = calc_time_stamps(
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-06-01"),
            pred_len=4, freq="W"
        )
        assert stamps.shape == (1, 4, 5)


class TestKronosPredictor:
    """预测器端到端测试"""

    @pytest.fixture
    def tokenizer(self):
        return KronosTokenizer(
            d_in=6, d_model=64, n_heads=4, ff_dim=128,
            n_enc_layers=2, n_dec_layers=2,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            s1_bits=4, s2_bits=4,
            beta=0.25, gamma0=1.0, gamma=0.1, zeta=0.1,
            group_size=4,
        )

    @pytest.fixture
    def model(self):
        return Kronos(
            s1_bits=4, s2_bits=4,
            n_layers=2, d_model=64, n_heads=4, ff_dim=128,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            token_dropout_p=0.0, learn_te=False,
        )

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 60
        close = 10 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_p = close - np.random.randn(n) * 0.2
        volume = np.random.randint(1000000, 10000000, n).astype(float)
        amount = volume * close * np.random.uniform(0.9, 1.1, n)
        dates = pd.date_range(start="2020-01-01", periods=n, freq="MS")
        return pd.DataFrame({
            "open": open_p, "high": high, "low": low,
            "close": close, "volume": volume, "amount": amount,
        }, index=dates)

    def test_predictor_init(self, tokenizer, model):
        predictor = KronosPredictor(tokenizer, model, device="cpu")
        assert predictor.device.type == "cpu"

    def test_predict_basic(self, tokenizer, model, df):
        predictor = KronosPredictor(tokenizer, model, device="cpu")
        result = predictor.predict(
            df, x_timestamp="2024-01-01", y_timestamp="2024-02-01",
            pred_len=3, context_len=40, sample_count=2,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 6)
        assert list(result.columns) == [
            "open", "high", "low", "close", "volume", "amount"
        ]

    def test_predict_no_nan(self, tokenizer, model, df):
        predictor = KronosPredictor(tokenizer, model, device="cpu")
        result = predictor.predict(
            df, x_timestamp="2024-01-01", y_timestamp="2024-02-01",
            pred_len=3, context_len=40, sample_count=2,
        )
        assert not result.isna().any().any()
        assert (result["close"] > 0).all()

    def test_insufficient_data(self, tokenizer, model, df):
        predictor = KronosPredictor(tokenizer, model, device="cpu")
        with pytest.raises(ValueError, match="历史数据不足"):
            predictor.predict(
                df, x_timestamp="2020-02-01", y_timestamp="2020-03-01",
                pred_len=3, context_len=400,
            )

    def test_extract_ohlcv_case_insensitive(self):
        df = pd.DataFrame({
            "Open": [10.0, 11.0], "High": [10.5, 11.5],
            "Low": [9.5, 10.5], "Close": [10.2, 11.2],
            "Volume": [1e6, 2e6], "Amount": [1e7, 2e7],
        })
        result = KronosPredictor._extract_ohlcv(df)
        assert result.shape == (2, 6)

    def test_extract_ohlcv_missing(self):
        df = pd.DataFrame({"open": [10.0], "high": [10.5]})
        with pytest.raises(KeyError):
            KronosPredictor._extract_ohlcv(df)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
    def test_predict_cuda(self, tokenizer, model, df):
        predictor = KronosPredictor(tokenizer, model, device="cuda")
        result = predictor.predict(
            df, x_timestamp="2024-01-01", y_timestamp="2024-02-01",
            pred_len=2, context_len=30, sample_count=2,
        )
        assert result.shape == (2, 6)


class TestAutoRegressiveInference:
    """自回归推理测试"""

    @pytest.fixture
    def model(self):
        return Kronos(
            s1_bits=4, s2_bits=4,
            n_layers=2, d_model=64, n_heads=4, ff_dim=128,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            token_dropout_p=0.0, learn_te=False,
        )

    def test_basic(self, model):
        device = torch.device("cpu")
        s1_hist = torch.randint(0, 15, (1, 20))
        s2_hist = torch.randint(0, 15, (1, 20))
        torch.manual_seed(42)
        pred_s1, pred_s2 = auto_regressive_inference(
            model, device, s1_hist, s2_hist,
            pred_len=5, temperature=0.8, top_p=0.95,
        )
        assert pred_s1.shape == (5,)
        assert pred_s2.shape == (5,)

    def test_max_context_truncation(self, model):
        device = torch.device("cpu")
        s1_hist = torch.randint(0, 15, (1, 30))
        s2_hist = torch.randint(0, 15, (1, 30))
        torch.manual_seed(42)
        pred_s1, pred_s2 = auto_regressive_inference(
            model, device, s1_hist, s2_hist,
            pred_len=10, max_context=10,
        )
        assert pred_s1.shape == (10,)
