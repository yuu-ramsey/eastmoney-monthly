"""
Kronos inference wrapper - autoregressive generation of predicted K-lines

Inference pipeline:
1. Tokenizer.encode(historical OHLCV) -> s1/s2 token sequence
2. Per-token autoregression: decode_s1 predict s1 -> sample -> decode_s2 predict s2 -> sample
3. Tokenizer.decode(s1, s2) -> reconstruction of prediction-period OHLCV
4. Multi-sample averaging for final prediction (reduces single-sample random fluctuation)
"""

# Reproduced from Kronos (https://github.com/shiyu-coder/Kronos)
# Original author: shiyu-coder | License: MIT | Paper: arXiv:2508.02739 (AAAI 2026)
# See kronos/LICENSE for full license text.


from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .tokenizer import KronosTokenizer
from .transformer import Kronos


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """
    Joint Top-k + Top-p (nucleus) filtering.

    First keep the top-k tokens with highest logits, then apply nucleus filtering
    on the remaining set, setting low-probability tokens whose cumulative probability
    exceeds top_p to filter_value.

    Args:
        logits: [B, vocab_size] raw logits
        top_k: keep top-k tokens (0 means no limit)
        top_p: nucleus sampling threshold (0.0 means no limit)
        filter_value: value assigned to filtered tokens
    """
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability exceeding top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[...] = sorted_indices_to_remove.long()
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """
    Sample a single token from logits.

    Args:
        logits: [B, vocab_size] current-step logits
        temperature: temperature coefficient (lower = more deterministic)
        top_k: top-k filtering
        top_p: nucleus filtering

    Returns:
        [B] sampled token ID
    """
    logits = logits.clone() / temperature
    logits = top_k_top_p_filtering(logits, top_k, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def auto_regressive_inference(
    model: Kronos,
    device: torch.device,
    s1_history: torch.Tensor,
    s2_history: torch.Tensor,
    pred_len: int,
    stamp: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    max_context: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Autoregressive inference: predict future sequence token by token.

    To avoid repeated forward passes over the entire historical sequence, a sliding
    window cache is used: after each token prediction, append it to the sequence end;
    when exceeding max_context, discard the earliest tokens.

    Args:
        model: Kronos prediction model
        device: computation device
        s1_history: [1, history_len] historical s1 tokens
        s2_history: [1, history_len] historical s2 tokens
        pred_len: number of prediction steps
        stamp: [1, pred_len, 5] future timestamps (optional)
        temperature: sampling temperature
        top_k: top-k filtering
        top_p: nucleus filtering
        max_context: maximum context length

    Returns:
        (pred_s1, pred_s2): each [pred_len] predicted tokens
    """
    model.eval()
    s1_buff = s1_history.clone()
    s2_buff = s2_history.clone()

    pred_s1_list: list[int] = []
    pred_s2_list: list[int] = []

    for step in range(pred_len):
        cur_len = s1_buff.shape[1]
        if cur_len > max_context:
            s1_buff = s1_buff[:, -max_context:]
            s2_buff = s2_buff[:, -max_context:]

        # Build padding mask (all False, no padding)
        padding_mask = torch.zeros(1, s1_buff.shape[1], dtype=torch.bool, device=device)

        # Build timestamp for current step (needs to provide timestamps for the entire sequence)
        step_stamp: Optional[torch.Tensor] = None
        if stamp is not None:
            t_cur = stamp.shape[1]
            t_needed = s1_buff.shape[1]
            if t_needed <= t_cur:
                step_stamp = stamp[:, :t_needed, :].to(device)

        # Decode s1 (coarse trend)
        s1_logits, context = model.decode_s1(
            s1_buff, s2_buff, stamp=step_stamp, padding_mask=padding_mask
        )
        # Take the last step's s1 logit
        last_s1_logits = s1_logits[0, -1, :]
        s1_token = sample_from_logits(
            last_s1_logits.unsqueeze(0), temperature, top_k, top_p
        )
        s1_id_val = s1_token.item()

        # Decode s2 (fine fluctuation) using the sampled s1
        s1_tensor = torch.full(
            (1, s1_buff.shape[1]), s1_id_val,
            dtype=torch.long, device=device
        )
        s2_logits = model.decode_s2(context, s1_tensor, padding_mask=padding_mask)
        last_s2_logits = s2_logits[0, -1, :]
        s2_token = sample_from_logits(
            last_s2_logits.unsqueeze(0), temperature, top_k, top_p
        )
        s2_id_val = s2_token.item()

        pred_s1_list.append(s1_id_val)
        pred_s2_list.append(s2_id_val)

        # Append predicted token to buffer
        s1_buff = torch.cat([
            s1_buff,
            torch.tensor([[s1_id_val]], dtype=torch.long, device=device)
        ], dim=1)
        s2_buff = torch.cat([
            s2_buff,
            torch.tensor([[s2_id_val]], dtype=torch.long, device=device)
        ], dim=1)

    return (torch.tensor(pred_s1_list, dtype=torch.long),
            torch.tensor(pred_s2_list, dtype=torch.long))


def calc_time_stamps(
    x_timestamp: pd.Timestamp,
    y_timestamp: pd.Timestamp,
    pred_len: int,
    freq: str = "M",
) -> torch.Tensor:
    """
    Build timestamp tensor [minutes, hours, weekday, day, month].

    Args:
        x_timestamp: data start time
        y_timestamp: prediction start time
        pred_len: prediction length
        freq: frequency ("M"=monthly, "W"=weekly, "D"=daily)

    Returns:
        [1, T, 5] timestamp tensor
    """
    if freq == "M":
        dates = pd.date_range(start=y_timestamp, periods=pred_len, freq="MS")
    elif freq == "W":
        dates = pd.date_range(start=y_timestamp, periods=pred_len, freq="W")
    elif freq == "D":
        dates = pd.date_range(start=y_timestamp, periods=pred_len, freq="D")
    else:
        dates = pd.date_range(start=y_timestamp, periods=pred_len, freq=freq)

    stamps = np.zeros((1, pred_len, 5), dtype=np.float32)
    for i, d in enumerate(dates):
        stamps[0, i, 0] = d.minute
        stamps[0, i, 1] = d.hour
        stamps[0, i, 2] = d.weekday()
        stamps[0, i, 3] = d.day
        stamps[0, i, 4] = d.month

    return torch.from_numpy(stamps)


class KronosPredictor:
    """
    Kronos predictor: encapsulates the complete tokenizer + model inference pipeline.

    Usage:
        predictor = KronosPredictor(tokenizer, model, device="cuda")
        result = predictor.predict(df, x_timestamp="2024-01-01",
                                   y_timestamp="2024-06-01", pred_len=24)

    Args:
        tokenizer: KronosTokenizer instance
        model: Kronos instance
        device: computation device ("cuda" / "cpu")
        max_context: maximum historical length retained during autoregressive generation
    """

    def __init__(
        self,
        tokenizer: KronosTokenizer,
        model: Kronos,
        device: str = "cuda",
        max_context: int = 512,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.max_context = max_context

        self.tokenizer.to(self.device)
        self.model.to(self.device)

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: str,
        y_timestamp: str,
        pred_len: int,
        context_len: int = 400,
        top_p: float = 0.95,
        top_k: int = 0,
        temperature: float = 1.0,
        sample_count: int = 5,
    ) -> pd.DataFrame:
        """
        Single sequence prediction.

        Args:
            df: DataFrame with OHLCV columns (at least open/high/low/close/volume/amount),
                sorted by time in ascending order
            x_timestamp: historical data cutoff time (e.g. "2024-01-01")
            y_timestamp: prediction start time (e.g. "2024-02-01")
            pred_len: number of predicted K-lines
            context_len: number of historical K-lines to use (taken backward from x_timestamp)
            top_p: nucleus sampling threshold
            top_k: top-k filtering (0=no limit)
            temperature: sampling temperature
            sample_count: number of independent sampling runs (averaged to reduce variance)

        Returns:
            DataFrame with predicted OHLCV columns, indexed by prediction time points
        """
        x_ts = pd.Timestamp(x_timestamp)
        y_ts = pd.Timestamp(y_timestamp)

        # Extract historical data (up to x_timestamp)
        hist_df = df[df.index <= x_ts].tail(context_len).copy()
        if len(hist_df) < context_len:
            raise ValueError(
                f"Historical data insufficient: need {context_len}, got {len(hist_df)}"
            )

        # Extract OHLCV columns (normalized column names)
        ohlcv = self._extract_ohlcv(hist_df)

        # Z-score normalization
        mean = ohlcv.mean(axis=0, keepdims=True)
        std = ohlcv.std(axis=0, keepdims=True).clip(min=1e-6)
        ohlcv_norm = (ohlcv - mean) / std

        # Tokenizer encode
        x_tensor = torch.from_numpy(ohlcv_norm.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            s1_indices, s2_indices = self.tokenizer.encode(x_tensor, half=True)

        # Multi-sample autoregressive inference
        all_s1: list[torch.Tensor] = []
        all_s2: list[torch.Tensor] = []

        for _ in range(sample_count):
            with torch.no_grad():
                pred_s1, pred_s2 = auto_regressive_inference(
                    self.model, self.device,
                    s1_indices, s2_indices,
                    pred_len,
                    stamp=None,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_context=self.max_context,
                )
            all_s1.append(pred_s1)
            all_s2.append(pred_s2)

        # Majority vote for s1 (discrete tokens use mode)
        stacked_s1 = torch.stack(all_s1, dim=0)  # [sample_count, pred_len]
        final_s1: list[int] = []
        for t in range(pred_len):
            mode_val = torch.mode(stacked_s1[:, t], dim=0).values.item()
            final_s1.append(mode_val)

        # s2 also uses mode
        stacked_s2 = torch.stack(all_s2, dim=0)
        final_s2: list[int] = []
        for t in range(pred_len):
            mode_val = torch.mode(stacked_s2[:, t], dim=0).values.item()
            final_s2.append(mode_val)

        # Decode back to OHLCV
        s1_t = torch.tensor(final_s1, dtype=torch.long, device=self.device).unsqueeze(0)
        s2_t = torch.tensor(final_s2, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            recon = self.tokenizer.decode((s1_t, s2_t), half=True)

        # De-normalize
        recon_np = recon.squeeze(0).cpu().numpy()
        recon_unorm = recon_np * std + mean

        # Clamp price columns (O/H/L/C) to non-negative, volume/amount to non-negative
        recon_unorm[:, :4] = np.clip(recon_unorm[:, :4], 0.01, None)
        recon_unorm[:, 4:] = np.clip(recon_unorm[:, 4:], 0, None)

        # Generate prediction date index
        pred_dates = pd.date_range(start=y_ts, periods=pred_len, freq="MS")

        result = pd.DataFrame(
            recon_unorm,
            index=pred_dates,
            columns=["open", "high", "low", "close", "volume", "amount"],
        )
        return result

    def predict_batch(
        self,
        dfs: list[pd.DataFrame],
        x_timestamp: str,
        y_timestamp: str,
        pred_len: int,
        context_len: int = 400,
        top_p: float = 0.95,
        temperature: float = 1.0,
        sample_count: int = 3,
    ) -> list[pd.DataFrame]:
        """
        Batch prediction for multiple stocks.

        Args:
            dfs: list of OHLCV DataFrames, one per stock
            other parameters same as predict()

        Returns:
            list of prediction DataFrames, one per stock
        """
        results: list[pd.DataFrame] = []
        for i, df in enumerate(dfs):
            result = self.predict(
                df, x_timestamp, y_timestamp, pred_len, context_len,
                top_p=top_p, temperature=temperature, sample_count=sample_count,
            )
            results.append(result)
        return results

    @staticmethod
    def _extract_ohlcv(df: pd.DataFrame) -> np.ndarray:
        """Extract OHLCV data from DataFrame, auto-matching column names (case-insensitive)"""
        cols_lower = {c.strip().lower(): c for c in df.columns}
        ohlcv_cols: list[str] = []
        for target in ["open", "high", "low", "close", "volume", "amount"]:
            matched = cols_lower.get(target)
            if matched is None:
                partial = [v for k, v in cols_lower.items() if target in k]
                if partial:
                    matched = partial[0]
                else:
                    raise KeyError(f"DataFrame missing column: {target}, available columns: {list(df.columns)}")
            ohlcv_cols.append(matched)
        return df[ohlcv_cols].values.astype(np.float32)
