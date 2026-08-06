"""Token-level reward bridge for Verl GRPO."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Sequence

from src.verl_integration.reward_function import reward_fn

_EXPR_RE = re.compile(r"<expr>\s*(.*?)\s*</expr>", re.DOTALL | re.IGNORECASE)


class FactorLabVerlRewardBridge:
    """Place executable scalar rewards on the final response token.

    QuantEvolver's Verl bridge returns a reward tensor with the same shape as
    ``responses`` and writes each scalar reward at the last generated token.
    This class follows that interface while delegating scoring to our existing
    Factor DSL / backtest / DiCo reward function.
    """

    def __init__(self, tokenizer: Any | None = None):
        self.tokenizer = tokenizer

    def __call__(self, data: Any, return_dict: bool = False):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - Verl runtime has torch.
            raise RuntimeError("FactorLabVerlRewardBridge requires torch") from exc

        responses = data.batch["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        texts = self._decode_responses(data)
        metadata = {"extra_info": self._extra_info(data)}
        scalar_rewards = reward_fn(texts, **metadata)
        if not hasattr(scalar_rewards, "to"):
            scalar_rewards = torch.tensor(scalar_rewards, dtype=torch.float32, device=reward_tensor.device)
        else:
            scalar_rewards = scalar_rewards.to(device=reward_tensor.device, dtype=torch.float32)

        response_lengths = self._response_lengths(data, responses)
        row_index = torch.arange(responses.shape[0], device=responses.device)
        token_index = torch.clamp(response_lengths - 1, min=0)
        reward_tensor[row_index, token_index] = scalar_rewards

        if return_dict:
            extra = defaultdict(list)
            for text, reward in zip(texts, scalar_rewards.detach().cpu().tolist()):
                extra["expr"].append(_extract_expr(text))
                extra["train_score"].append(float(reward))
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor

    def _decode_responses(self, data: Any) -> list[str]:
        if hasattr(data, "non_tensor_batch") and "responses_str" in data.non_tensor_batch:
            return [str(item) for item in data.non_tensor_batch["responses_str"]]
        if self.tokenizer is None:
            raise ValueError("tokenizer is required when responses_str is not provided")
        return self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

    @staticmethod
    def _extra_info(data: Any) -> dict[str, Any] | list[dict[str, Any]]:
        if hasattr(data, "non_tensor_batch"):
            for key in ("extra_info", "extra_infos"):
                if key in data.non_tensor_batch:
                    return data.non_tensor_batch[key]
        return {}

    @staticmethod
    def _response_lengths(data: Any, responses: Any):
        import torch

        if "attention_mask" not in data.batch or "prompts" not in data.batch:
            return torch.full((responses.shape[0],), responses.shape[1], device=responses.device, dtype=torch.long)
        prompt_len = data.batch["prompts"].shape[-1]
        return data.batch["attention_mask"][:, prompt_len:].sum(dim=1).long()


def _extract_expr(text: str) -> str:
    match = _EXPR_RE.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    return stripped if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\(.*\)$", stripped) else ""
