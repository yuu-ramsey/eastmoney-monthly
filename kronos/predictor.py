"""
Kronos inference wrapper — autoregressive generation of predicted K-lines

Inference pipeline:
1. Tokenizer.encode(历史 OHLCV) → s1/s2 token 序列
2. 逐 token 自回归：decode_s1 预测 s1 → 采样 → decode_s2 预测 s2 → 采样
3. Tokenizer.decode(s1, s2) → 预测期 OHLCV 重建
4. 多采样取均值得最终预测（降低单次采样的随机波动）
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
    Top-k + Top-p (nucleus) 联合过滤。

    先保留 top-k 个最高 logit 的 token，再在剩余集合上做 nucleus 过滤，
    将概率累积超过 top_p 的低概率 token 设为 filter_value。

    Args:
        logits: [B, vocab_size] 原始 logits
        top_k: 保留 top-k 个 token（0 表示不限制）
        top_p: nucleus 采样阈值（0.0 表示不限制）
        filter_value: 被过滤 token 赋予的值
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

        # 移除累积概率超过 top_p 的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 至少保留一个 token
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
    从 logits 中采样单个 token。

    Args:
        logits: [B, vocab_size] 当前步 logits
        temperature: 温度系数（越低越确定）
        top_k: top-k 过滤
        top_p: nucleus 过滤

    Returns:
        [B] 采样得到的 token ID
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
    自回归推理：逐 token 预测未来序列。

    为了避免重复前向传播整个历史序列，使用滑动窗口缓存：
    每预测一个 token 后将其追加到序列末尾，超出 max_context 时丢弃最早的 token。

    Args:
        model: Kronos 预测模型
        device: 计算设备
        s1_history: [1, history_len] 历史 s1 token
        s2_history: [1, history_len] 历史 s2 token
        pred_len: 预测步数
        stamp: [1, pred_len, 5] 未来时间戳（可选）
        temperature: 采样温度
        top_k: top-k 过滤
        top_p: nucleus 过滤
        max_context: 最大上下文长度

    Returns:
        (pred_s1, pred_s2): 各 [pred_len] 预测的 token
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

        # 构建 padding mask（全 False，无 padding）
        padding_mask = torch.zeros(1, s1_buff.shape[1], dtype=torch.bool, device=device)

        # 构建当前步的时间戳（需要提供对应整个序列的时间戳）
        step_stamp: Optional[torch.Tensor] = None
        if stamp is not None:
            t_cur = stamp.shape[1]
            t_needed = s1_buff.shape[1]
            if t_needed <= t_cur:
                step_stamp = stamp[:, :t_needed, :].to(device)

        # 解码 s1（粗粒度趋势）
        s1_logits, context = model.decode_s1(
            s1_buff, s2_buff, stamp=step_stamp, padding_mask=padding_mask
        )
        # 取最后一步的 s1 logit
        last_s1_logits = s1_logits[0, -1, :]
        s1_token = sample_from_logits(
            last_s1_logits.unsqueeze(0), temperature, top_k, top_p
        )
        s1_id_val = s1_token.item()

        # 用采样的 s1 解码 s2（细粒度波动）
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

        # 追加预测 token 到缓冲区
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
    构建时间戳张量 [minutes, hours, weekday, day, month]。

    Args:
        x_timestamp: 数据起始时间
        y_timestamp: 预测起始时间
        pred_len: 预测长度
        freq: 频率（"M"=月度, "W"=周, "D"=日）

    Returns:
        [1, T, 5] 时间戳张量
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
    Kronos 预测器：封装 tokenizer + model 的完整推理流程。

    使用方式:
        predictor = KronosPredictor(tokenizer, model, device="cuda")
        result = predictor.predict(df, x_timestamp="2024-01-01",
                                   y_timestamp="2024-06-01", pred_len=24)

    Args:
        tokenizer: KronosTokenizer 实例
        model: Kronos 实例
        device: 计算设备 ("cuda" / "cpu")
        max_context: 自回归生成时最大保留的历史长度
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
        单序列预测。

        Args:
            df: 包含 OHLCV 列的 DataFrame（至少需要 open/high/low/close/volume/amount），
                按时间升序排列
            x_timestamp: 历史数据截止时间 (如 "2024-01-01")
            y_timestamp: 预测起始时间 (如 "2024-02-01")
            pred_len: 预测 K 线数量
            context_len: 使用的历史 K 线数量（从 x_timestamp 往前取）
            top_p: nucleus 采样阈值
            top_k: top-k 过滤（0=不限制）
            temperature: 采样温度
            sample_count: 独立采样次数（取均值降低方差）

        Returns:
            DataFrame 包含预测的 OHLCV 列，索引为预测时间点
        """
        x_ts = pd.Timestamp(x_timestamp)
        y_ts = pd.Timestamp(y_timestamp)

        # 提取历史数据（截止 x_timestamp）
        hist_df = df[df.index <= x_ts].tail(context_len).copy()
        if len(hist_df) < context_len:
            raise ValueError(
                f"历史数据不足: 需要 {context_len} 条，实际 {len(hist_df)} 条"
            )

        # 提取 OHLCV 列（标准化列名）
        ohlcv = self._extract_ohlcv(hist_df)

        # Z-score 归一化
        mean = ohlcv.mean(axis=0, keepdims=True)
        std = ohlcv.std(axis=0, keepdims=True).clip(min=1e-6)
        ohlcv_norm = (ohlcv - mean) / std

        # Tokenizer encode
        x_tensor = torch.from_numpy(ohlcv_norm.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            s1_indices, s2_indices = self.tokenizer.encode(x_tensor, half=True)

        # 多采样自回归推理
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

        # 多数投票取 s1（离散 token 用 mode）
        stacked_s1 = torch.stack(all_s1, dim=0)  # [sample_count, pred_len]
        final_s1: list[int] = []
        for t in range(pred_len):
            mode_val = torch.mode(stacked_s1[:, t], dim=0).values.item()
            final_s1.append(mode_val)

        # s2 也取 mode
        stacked_s2 = torch.stack(all_s2, dim=0)
        final_s2: list[int] = []
        for t in range(pred_len):
            mode_val = torch.mode(stacked_s2[:, t], dim=0).values.item()
            final_s2.append(mode_val)

        # Decode 回 OHLCV
        s1_t = torch.tensor(final_s1, dtype=torch.long, device=self.device).unsqueeze(0)
        s2_t = torch.tensor(final_s2, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            recon = self.tokenizer.decode((s1_t, s2_t), half=True)

        # 反归一化
        recon_np = recon.squeeze(0).cpu().numpy()
        recon_unorm = recon_np * std + mean

        # 钳制价格列 (O/H/L/C) 为非负，volume/amount 为非负
        recon_unorm[:, :4] = np.clip(recon_unorm[:, :4], 0.01, None)
        recon_unorm[:, 4:] = np.clip(recon_unorm[:, 4:], 0, None)

        # 生成预测日期索引
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
        批量预测多只股票。

        Args:
            dfs: 每只股票的 OHLCV DataFrame 列表
            其余参数同 predict()

        Returns:
            每只股票的预测 DataFrame 列表
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
        """从 DataFrame 提取 OHLCV 数据，自动匹配列名（大小写不敏感）"""
        cols_lower = {c.strip().lower(): c for c in df.columns}
        ohlcv_cols: list[str] = []
        for target in ["open", "high", "low", "close", "volume", "amount"]:
            matched = cols_lower.get(target)
            if matched is None:
                partial = [v for k, v in cols_lower.items() if target in k]
                if partial:
                    matched = partial[0]
                else:
                    raise KeyError(f"DataFrame 缺少列: {target}，可用列: {list(df.columns)}")
            ohlcv_cols.append(matched)
        return df[ohlcv_cols].values.astype(np.float32)
