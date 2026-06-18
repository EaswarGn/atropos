"""
VLLMLogProbServer: APIServer implementation for vLLM using the standard
OpenAI-compatible /v1/completions and /v1/chat/completions endpoints
with --logprobs-mode processed_logprobs enabled.

Unlike VLLMServer (which uses vLLM's custom /generate endpoint), this server
uses the standard OpenAI endpoints that return rich logprob data in the
OpenAI response format when started with:

    vllm serve MODEL --logprobs-mode processed_logprobs

Response format includes:
  - choice.logprobs.token_logprobs - sampled token logprob per token
  - choice.logprobs.tokens - token strings
  - choice.logprobs.top_logprobs - top-k token→logprob dicts per position
  - choice.token_ids - actual token IDs (non-standard extension)
  - response.prompt_logprobs - prompt token logprobs (non-standard extension)

Fully compatible with ManagedServer for automatic token/logprob tracking.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from pydantic_cli import FailedExecutionException
from transformers import AutoTokenizer

from atroposlib.envs.constants import NAMESPACE_SEP, OPENAI_NAMESPACE
from atroposlib.envs.server_handling.server_baseline import (
    APIServer,
    APIServerConfig,
    ReasoningConfig,
)

logger = logging.getLogger(__name__)


class VLLMLogProbServer(APIServer):
    """
    APIServer implementation for vLLM using the OpenAI-compatible /v1/completions
    and /v1/chat/completions endpoints with --logprobs-mode processed_logprobs.

    This is the recommended server class when starting vLLM with:
        vllm serve MODEL --logprobs-mode processed_logprobs

    It provides richer logprob information than VLLMServer by using the standard
    OpenAI endpoints rather than vLLM's custom /generate endpoint.

    Fully compatible with ManagedServer for automatic token/logprob tracking.

    Args:
        config: Standard APIServerConfig. config.base_url should point at the
                vLLM OpenAI-compatible root, e.g. ``http://localhost:8000/v1``.
        reasoning_config: Optional reasoning/thinking configuration.
        timeout: Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        config: APIServerConfig,
        reasoning_config: Optional[ReasoningConfig] = None,
        timeout: Optional[int] = None,
    ):
        self.openai = openai.AsyncClient(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=timeout or config.timeout,
        )
        # Lazy aiohttp session for native logprob fetching
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        # Load tokenizer eagerly so ManagedServer can reuse it.
        tokenizer_name = (
            config.tokenizer_name
            if config.tokenizer_name and config.tokenizer_name != "none"
            else config.model_name
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info("Loaded tokenizer from %s", tokenizer_name)
        except Exception as exc:
            logger.warning(
                "Could not load tokenizer from %r: %s — tokenizer will be None.",
                tokenizer_name,
                exc,
            )
            self.tokenizer = None

        super().__init__(config, reasoning_config=reasoning_config)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return (or lazily create) the shared aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                    self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session. Call on shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Health check (required abstract method)
    # ------------------------------------------------------------------

    async def check_server_status_task(self, chat_completion: bool = True) -> None:
        """
        Poll /health until the server is reachable, then set server_healthy=True.
        """
        base = self.config.base_url.replace("/v1", "").rstrip("/")
        health_url = f"{base}/health"
        backoff = 1.0

        session = await self._get_session()

        while True:
            try:
                async with session.get(health_url) as resp:
                    if resp.status == 200:
                        self.server_healthy = True
                        logger.info(
                            "VLLMLogProbServer is healthy at %s", self.config.base_url
                        )
                        return
                    logger.warning(
                        "Health check HTTP %d, retrying in %.1fs …",
                        resp.status,
                        backoff,
                    )
            except Exception as exc:
                logger.warning(
                    "Health check failed (%s), retrying in %.1fs …", exc, backoff
                )
            self.server_healthy = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)

    # ------------------------------------------------------------------
    # Chat completion wrapper
    # ------------------------------------------------------------------

    async def _chat_completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Wrapper for chat completion using the OpenAI-compatible client.
        """
        assert kwargs.get("model") is not None, "Model is required for chat completion!"
        assert (
            kwargs.get("messages") is not None
        ), "Messages are required for chat completion!"

        kwargs.setdefault("logprobs", False)

        if self.config.n_kwarg_is_ignored:
            n = kwargs.pop("n", 1)
            completion_list = await asyncio.gather(
                *[
                    self.openai.chat.completions.create(**copy.deepcopy(kwargs))
                    for _ in range(n)
                ]
            )
            completions = completion_list[0]
            if n > 1:
                for c in completion_list[1:]:
                    completions.choices.extend(c.choices)
        else:
            n = kwargs.get("n", 1)
            completions = await self.openai.chat.completions.create(**kwargs)
            if len(completions.choices) != n:
                if len(completions.choices) != 1:
                    raise ValueError(
                        f"Expected 1 or {n} completions, got {len(completions.choices)}!"
                    )
                else:
                    warnings.warn("n kwarg is ignored by the API, setting to True")
                    self.config.n_kwarg_is_ignored = True
                    completion_list = await asyncio.gather(
                        *[
                            self.openai.chat.completions.create(**copy.deepcopy(kwargs))
                            for _ in range(1, n)
                        ]
                    )
                    for c in completion_list:
                        completions.choices.extend(c.choices)
        return completions

    # ------------------------------------------------------------------
    # Completion wrapper
    # ------------------------------------------------------------------

    async def _completion_wrapper(self, **kwargs) -> Completion:
        """
        Wrapper for completion using the OpenAI-compatible client.
        """
        assert kwargs.get("model") is not None, "Model is required for completion!"
        assert (
            kwargs.get("prompt") is not None or kwargs.get("input_ids") is not None
        ), "Prompt or input_ids is required for completion!"

        kwargs.setdefault("logprobs", False)

        if "input_ids" in kwargs:
            kwargs["prompt"] = kwargs.pop("input_ids")
            kwargs.pop("messages", None)

        if self.config.n_kwarg_is_ignored:
            n = kwargs.pop("n", 1)
            completion_list = await asyncio.gather(
                *[
                    self.openai.completions.create(**copy.deepcopy(kwargs))
                    for _ in range(n)
                ]
            )
            completions = completion_list[0]
            if n > 1:
                for c in completion_list[1:]:
                    completions.choices.extend(c.choices)
        else:
            n = kwargs.get("n", 1)
            completions = await self.openai.completions.create(**kwargs)
            if len(completions.choices) != n:
                if len(completions.choices) != 1:
                    raise ValueError(
                        f"Expected 1 or {n} completions, got {len(completions.choices)}!"
                    )
                else:
                    warnings.warn("n kwarg is ignored by the API, setting to True")
                    self.config.n_kwarg_is_ignored = True
                    completion_list = await asyncio.gather(
                        *[
                            self.openai.completions.create(**copy.deepcopy(kwargs))
                            for _ in range(1, n)
                        ]
                    )
                    for c in completion_list:
                        completions.choices.extend(c.choices)
        return completions

    # ------------------------------------------------------------------
    # Tokens-and-logprobs wrapper  ← the key ManagedServer interface
    # ------------------------------------------------------------------

    async def _tokens_and_logprobs_completion_wrapper(
        self, **kwargs
    ) -> Tuple[List[int], List[List[int]], List[List[float]], List[str]]:
        """
        Generate completions and return raw tokens + per-token logprobs.

        Uses the standard /v1/completions endpoint with logprobs=0 (sampled token only).
        With --logprobs-mode processed_logprobs, the response includes:
          - choice.token_ids: list of token IDs
          - choice.logprobs.token_logprobs: list of logprob floats aligned with token_ids

        Supports both text prompts and pre-tokenized input_ids (the ManagedServer
        multi-turn path supplies input_ids to avoid double-tokenisation).

        Returns:
            prompt_tokens      : list[int]  — token IDs of the prompt
            output_tokens_list : list[list[int]]  — one list per completion
            output_logprobs_list: list[list[float]] — sampled-token logprob per
                                   completion token, aligned with output_tokens
            finish_reasons     : list[str]  — "stop" or "length" per completion
        """
        assert kwargs.get("model") is not None, "Model is required for completion!"
        assert (
            kwargs.get("prompt") is not None or kwargs.get("input_ids") is not None
        ), "Prompt or input_ids is required for completion!"

        # Build kwargs for the OpenAI-compatible completions endpoint
        comp_kwargs: Dict[str, Any] = {}

        if "input_ids" in kwargs:
            prompt_tokens = kwargs.pop("input_ids")
            kwargs.pop("prompt", None)
            kwargs.pop("messages", None)  # Clean up messages if present

            comp_kwargs["prompt"] = prompt_tokens
        else:
            messages = kwargs.pop("messages", None)
            if messages is not None:
                if self.tokenizer is not None and hasattr(
                    self.tokenizer, "apply_chat_template"
                ):
                    # Keep tokenize=True to get token IDs directly
                    prompt_tokens = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                    comp_kwargs["prompt"] = prompt_tokens
                else:
                    # Fallback string concatenation
                    fallback_text = "\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}"
                        for m in messages
                    )
                    prompt_tokens = self.tokenizer.encode(fallback_text)
                    comp_kwargs["prompt"] = fallback_text
            else:
                # Standard text prompt path
                raw_prompt = kwargs.pop("prompt", "")
                prompt_tokens = self.tokenizer.encode(raw_prompt)
                comp_kwargs["prompt"] = raw_prompt

        # Normalize double BOS
        if (
            len(prompt_tokens) >= 2
            and prompt_tokens[0] == self.tokenizer.bos_token_id == prompt_tokens[1]
        ):
            prompt_tokens = prompt_tokens[1:]
            # If the payload is already token IDs, update it to match the sliced tokens
            if isinstance(comp_kwargs["prompt"], list):
                comp_kwargs["prompt"] = prompt_tokens

        # Sampling parameters
        comp_kwargs["model"] = kwargs.pop("model", self.config.model_name)
        comp_kwargs["n"] = kwargs.pop("n", 1)
        comp_kwargs["max_tokens"] = kwargs.pop(
            "max_tokens", kwargs.pop("max_new_tokens", 1024)
        )
        comp_kwargs["temperature"] = kwargs.pop("temperature", 1.0)
        comp_kwargs["top_p"] = kwargs.pop("top_p", 1.0)
        comp_kwargs["stop"] = kwargs.pop("stop", None)

        # CRITICAL: request logprobs=0 to get just the sampled token's logprob.
        # With --logprobs-mode processed_logprobs, each choice will have:
        #   - choice.logprobs.token_logprobs: list of floats (sampled logprob per token)
        #   - choice.token_ids: list of int (token IDs)
        comp_kwargs["logprobs"] = 0

        # Forward any remaining generation kwargs (echo, best_of, etc.)
        for key in (
            "echo",
            "best_of",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "user",
        ):
            if key in kwargs:
                comp_kwargs[key] = kwargs.pop(key)

        # Call the standard /v1/completions endpoint
        completion: Completion = await self.openai.completions.create(**comp_kwargs)

        # ── Unpack response ──────────────────────────────────────────────
        output_tokens_list: List[List[int]] = []
        output_logprobs_list: List[List[float]] = []
        finish_reasons: List[str] = []

        for choice in completion.choices:
            # Extract token IDs from choice.token_ids (vLLM extension)
            # Fall back to re-encoding if token_ids is not present
            if hasattr(choice, "token_ids") and choice.token_ids is not None:
                token_ids = list(choice.token_ids)
            else:
                # Decode from text and re-encode
                text = choice.text or ""
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)

            # Extract logprobs from choice.logprobs.token_logprobs
            # choice.logprobs is None when logprobs=False in the request
            # choice.logprobs.token_logprobs is None when logprobs=0 not honored
            logprobs: List[float] = []
            if choice.logprobs is not None:
                raw_lps = getattr(choice.logprobs, "token_logprobs", None)
                if raw_lps is not None:
                    logprobs = [float(lp) if lp is not None else 0.0 for lp in raw_lps]

            # Align logprobs length to token length
            if len(logprobs) < len(token_ids):
                logprobs = logprobs + [0.0] * (len(token_ids) - len(logprobs))
            elif len(logprobs) > len(token_ids):
                logprobs = logprobs[: len(token_ids)]

            # Finish reason
            finish_reason = choice.finish_reason or "stop"

            output_tokens_list.append(token_ids)
            output_logprobs_list.append(logprobs)
            finish_reasons.append(finish_reason)

        return prompt_tokens, output_tokens_list, output_logprobs_list, finish_reasons

    # ------------------------------------------------------------------
    # Prompt logprobs wrapper  ← used by ManagedServer.get_logprobs()
    # ------------------------------------------------------------------

    async def _get_logprobs_wrapper(self, **kwargs) -> Dict[str, Any]:
        """
        Fetch normalized prompt logprobs using the /v1/completions endpoint
        with prompt_logprobs parameter.

        With --logprobs-mode processed_logprobs, vLLM returns prompt_logprobs
        as a list of Logprob objects in the top-level response field.

        Args:
            top_k / top_logprobs: Number of logprobs per position (default 1).
            prompt or input_ids: Input text or token IDs.

        Returns:
            Normalized dict:
              - prompt_tokens
              - prompt_topk_token_ids
              - prompt_topk_logprobs
        """
        assert (
            kwargs.get("prompt") is not None or kwargs.get("input_ids") is not None
        ), "Prompt or input_ids is required for get_logprobs!"

        top_k = int(kwargs.pop("top_k", kwargs.pop("top_logprobs", 1)))
        top_k = max(1, top_k)

        # Resolve token IDs
        from_prompt_text = False
        if "input_ids" in kwargs:
            prompt_tokens = kwargs.pop("input_ids")
            kwargs.pop("prompt", None)
        else:
            prompt_tokens = self.tokenizer.encode(kwargs.pop("prompt"))
            from_prompt_text = True

        # Normalize double BOS
        if (
            from_prompt_text
            and len(prompt_tokens) >= 2
            and prompt_tokens[0] == self.tokenizer.bos_token_id == prompt_tokens[1]
        ):
            prompt_tokens = prompt_tokens[1:]

        comp_kwargs: Dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model_name),
            "prompt": prompt_tokens,
            "echo": True,
            "max_tokens": 0,  # No completion tokens
            "temperature": 0.0,
            "logprobs": top_k,
        }

        response: Completion = await self.openai.completions.create(**comp_kwargs)

        # Parse response prompt logprobs
        # With processed_logprobs, prompt_logprobs is at the top level of the response
        # It's a list of LogprobResult objects: [LogprobResult(...), ...]
        # Each LogprobResult has: top_logprobs, text_offset, token_id, token, rank
        raw_prompt_logprobs = getattr(response, "prompt_logprobs", None)

        if raw_prompt_logprobs is None:
            raise ValueError(
                "vLLM /v1/completions response missing 'prompt_logprobs'. "
                "Ensure --logprobs-mode processed_logprobs is enabled on the server."
            )

        prompt_topk_token_ids: List[List[int]] = []
        prompt_topk_logprobs: List[List[float]] = []

        for i, entry in enumerate(raw_prompt_logprobs):
            # Normalize entry - handle both LogprobResult objects and dicts
            if hasattr(entry, "top_logprobs"):
                # LogprobResult object from the OpenAI SDK
                top_lps_dict = entry.top_logprobs  # dict of {token_str: logprob}
                # Sort by logprob descending and take top_k
                sorted_lps = sorted(
                    top_lps_dict.items(), key=lambda x: x[1], reverse=True
                )[:top_k]
                pos_token_strs = [k for k, _ in sorted_lps]
                pos_logprobs = [float(v) for _, v in sorted_lps]
                # Convert token strings to IDs using our tokenizer
                pos_token_ids: List[int] = []
                for t_str in pos_token_strs:
                    try:
                        tid = self.tokenizer.convert_tokens_to_ids(t_str)
                        if tid is None:
                            tid = self.tokenizer.encode(t_str, add_special_tokens=False)
                            tid = tid[0] if tid else 0
                        pos_token_ids.append(tid)
                    except Exception:
                        pos_token_ids.append(0)
            elif isinstance(entry, dict):
                # Raw dict response
                pos_token_ids = []
                pos_logprobs = []
                for tid_str, lp in list(entry.items())[:top_k]:
                    try:
                        pos_token_ids.append(int(tid_str))
                    except (ValueError, TypeError):
                        pos_token_ids.append(0)
                    pos_logprobs.append(float(lp))
            else:
                pos_token_ids = []
                pos_logprobs = []

            # Pad to top_k if needed
            while len(pos_token_ids) < top_k:
                pos_token_ids.append(0)
            while len(pos_logprobs) < top_k:
                pos_logprobs.append(0.0)

            prompt_topk_token_ids.append(pos_token_ids[:top_k])
            prompt_topk_logprobs.append(pos_logprobs[:top_k])

        return {
            "prompt_tokens": prompt_tokens,
            "prompt_topk_token_ids": prompt_topk_token_ids,
            "prompt_topk_logprobs": prompt_topk_logprobs,
        }


def resolve_openai_configs(
    default_server_configs,
    openai_config_dict,
    yaml_config,
    cli_passed_flags,
    logger,
):
    """
    Helper to resolve the final server_configs, handling single, multiple servers, and overrides.
    """
    from atroposlib.envs.server_handling.server_manager import ServerBaseline

    openai_full_prefix = f"{OPENAI_NAMESPACE}{NAMESPACE_SEP}"
    openai_yaml_config = yaml_config.get(OPENAI_NAMESPACE, None)
    openai_cli_config = {
        k: v for k, v in cli_passed_flags.items() if k.startswith(openai_full_prefix)
    }

    is_multi_server_yaml = (
        isinstance(openai_yaml_config, list) and len(openai_yaml_config) >= 2
    )
    is_multi_server_default = (
        (not is_multi_server_yaml)
        and isinstance(default_server_configs, list)
        and len(default_server_configs) >= 2
    )

    if (is_multi_server_yaml or is_multi_server_default) and openai_cli_config:
        raise FailedExecutionException(
            message=f"CLI overrides for OpenAI settings (--{openai_full_prefix}*) are not supported "
            f"when multiple servers are defined (either via YAML list under '{OPENAI_NAMESPACE}' "
            "or a default list with length >= 2).",
            exit_code=2,
        )

    if is_multi_server_yaml:
        logger.info(
            f"Using multi-server configuration defined in YAML under '{OPENAI_NAMESPACE}'."
        )
        try:
            server_configs = [APIServerConfig(**cfg) for cfg in openai_yaml_config]
        except Exception as e:
            raise FailedExecutionException(
                f"Error parsing multi-server OpenAI configuration from YAML under '{OPENAI_NAMESPACE}': {e}"
            ) from e
    elif isinstance(default_server_configs, ServerBaseline):
        logger.info("Using ServerBaseline configuration.")
        server_configs = default_server_configs
    elif is_multi_server_default:
        logger.info("Using default multi-server configuration (length >= 2).")
        server_configs = default_server_configs
    else:
        logger.info(
            "Using single OpenAI server configuration based on merged settings (default/YAML/CLI)."
        )
        try:
            final_openai_config = APIServerConfig(**openai_config_dict)
        except Exception as e:
            raise FailedExecutionException(
                f"Error creating final OpenAI configuration from merged settings: {e}\n"
                f"Merged Dict: {openai_config_dict}"
            ) from e

        if isinstance(default_server_configs, APIServerConfig):
            server_configs = final_openai_config
        elif isinstance(default_server_configs, list):
            server_configs = [final_openai_config]
        else:
            logger.warning(
                f"Unexpected type for default_server_configs: {type(default_server_configs)}. "
                "Proceeding with single OpenAI server configuration based on merged settings."
            )
            server_configs = [final_openai_config]

    return server_configs
