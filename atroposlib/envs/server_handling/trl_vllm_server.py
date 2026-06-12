"""
TrlVllmServer: A complete APIServer implementation for trl's vLLM inference server.

Supports all inference endpoints exposed by trl/scripts/vllm_serve.py:
  - POST /generate/             — prompt-based generation with logprobs
  - POST /chat/                 — chat-template generation with logprobs
  - POST /get_sequence_logprobs/ — sequence logprobs (teacher / KD mode)
  - GET  /health/               — liveness check

Fully compatible with ManagedServer, which requires:
  - _chat_completion_wrapper(**kwargs) -> ChatCompletion
  - _completion_wrapper(**kwargs)      -> Completion
  - _tokens_and_logprobs_completion_wrapper(**kwargs)
        -> tuple[prompt_tokens, output_tokens_list, output_logprobs_list, finish_reasons]
  - _get_logprobs_wrapper(**kwargs)
        -> dict with prompt_tokens, prompt_topk_token_ids, prompt_topk_logprobs

Design notes
------------
* All HTTP I/O goes through a single long-lived aiohttp.ClientSession created on
  first use, so that connection pooling works across concurrent async requests.
* logprobs from the /generate/ and /chat/ endpoints come back as
  list[list[list[float|None]]] (per-sequence, per-token, per-top-k).
  We always request logprobs=0 (sampled token only) unless the caller asks for
  more, so ManagedServer gets exactly the single sampled-token logprob it needs.
* _tokens_and_logprobs_completion_wrapper honors `input_ids` when provided
  (the ManagedServer multi-turn extension path encodes the prompt itself and
  passes pre-tokenized IDs to avoid double-tokenisation).
* get_sequence_logprobs uses the binary response_format for zero-copy numpy
  deserialization, falling back to JSON if numpy is unavailable.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.completion import Completion, CompletionChoice
from transformers import AutoTokenizer

from atroposlib.envs.server_handling.server_baseline import (
    APIServer,
    APIServerConfig,
    ReasoningConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_logprob(per_token_topk: List[Optional[float]]) -> float:
    """Return the first non-None logprob from a top-k list, or 0.0 as fallback."""
    for lp in per_token_topk:
        if lp is not None:
            return lp
    return 0.0


def _extract_sampled_logprobs(
    logprobs_nested: Optional[List[List[List[Optional[float]]]]],
) -> List[List[float]]:
    """
    Convert the server's raw logprobs structure into a flat per-sequence list.

    The server returns shape: (num_sequences, seq_len, num_top_k).
    We collapse the top-k axis by taking the first entry (the sampled token's
    logprob when logprobs=0, or the highest-probability token when logprobs>0).
    Returns a list of lists, one per sequence.
    """
    if logprobs_nested is None:
        return []
    result = []
    for seq_logprobs in logprobs_nested:
        flat = [
            _first_logprob(tok_topk) if tok_topk else 0.0 for tok_topk in seq_logprobs
        ]
        result.append(flat)
    return result


def _decode_binary_logprobs(
    response_json: Dict[str, Any],
) -> Tuple[
    List[List[List[float]]],
    List[List[List[int]]],
    List[List[float]],
    List[List[int]],
]:
    """
    Decode base64-encoded numpy arrays from a binary-format /get_sequence_logprobs/
    response.

    Returns:
        logprobs_arr       — (batch, max_comp, top_k) sorted top-k logprobs
        token_ids_arr      — (batch, max_comp, top_k) token IDs for logprobs_arr
        actual_logprobs    — (batch, max_comp, 1) actual token's logprob
        actual_token_ids   — (batch, max_comp, 1) actual token IDs
    """
    import numpy as np

    shape = response_json["shape"]  # [batch_size, max_comp_len, top_k]
    comp_lengths = response_json["completion_lengths"]

    batch, max_comp, top_k = shape

    def _unpack_f32(b64: str, shape_: tuple) -> np.ndarray:
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype=np.float32).reshape(shape_)

    def _unpack_i32(b64: str, shape_: tuple) -> np.ndarray:
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype=np.int32).reshape(shape_)

    lp_arr = _unpack_f32(response_json["logprobs_b64"], (batch, max_comp, top_k))
    tid_arr = _unpack_i32(response_json["token_ids_b64"], (batch, max_comp, top_k))
    alp_arr = _unpack_f32(response_json["actual_logprobs_b64"], (batch, max_comp, 1))
    atid_arr = _unpack_i32(response_json["actual_token_ids_b64"], (batch, max_comp, 1))

    # Convert to native Python lists, trimming to actual completion lengths
    logprobs_list = []
    token_ids_list = []
    actual_lp_list = []
    actual_tid_list = []

    for i in range(batch):
        clean = comp_lengths[i]
        logprobs_list.append(lp_arr[i, :clean, :].tolist())
        token_ids_list.append(tid_arr[i, :clean, :].tolist())
        actual_lp_list.append(alp_arr[i, :clean, 0].tolist())
        actual_tid_list.append(atid_arr[i, :clean, 0].tolist())

    return logprobs_list, token_ids_list, actual_lp_list, actual_tid_list


# ---------------------------------------------------------------------------
# Main server class
# ---------------------------------------------------------------------------


class TrlVllmServer(APIServer):
    """
    APIServer implementation for the trl vLLM HTTP inference server.

    Compatible with ManagedServer for automatic token/logprob tracking.

    Args:
        config: Standard APIServerConfig.  config.base_url should point at the
                trl vLLM server root, e.g. ``http://localhost:8000``.
        reasoning_config: Optional reasoning/thinking configuration (passed to
                          the base class; trl's vLLM server does not use it
                          directly but it is respected by the base class hooks).
        timeout: Per-request HTTP timeout in seconds.  Defaults to
                 config.timeout if not specified.
    """

    def __init__(
        self,
        config: APIServerConfig,
        reasoning_config: Optional[ReasoningConfig] = None,
        timeout: Optional[int] = None,
    ):
        self.config = config
        self._timeout = timeout or config.timeout
        # Lazy aiohttp session — created once on first request.
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        # Load tokenizer eagerly so ManagedServer can reuse it without a
        # second round-trip to the filesystem.
        tokenizer_name = (
            config.tokenizer_name
            if config.tokenizer_name and config.tokenizer_name != "none"
            else config.model_name
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info("Loaded tokenizer from %s", tokenizer_name)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Could not load tokenizer from %r: %s — tokenizer will be None.",
                tokenizer_name,
                exc,
            )
            self.tokenizer = None

        # Call base __init__ after our attributes are set (base may call
        # check_server_status_task via asyncio.create_task on first call,
        # but that is deferred until the event loop is running).
        super().__init__(config, reasoning_config=reasoning_config)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return (or lazily create) the shared aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=self._timeout)
                    self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session.  Call on shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _url(self, path: str) -> str:
        """Build a full URL for the given endpoint path."""
        base = (self.config.base_url or "").rstrip("/")
        return f"{base}{path}"

    # ------------------------------------------------------------------
    # Health check (required abstract method)
    # ------------------------------------------------------------------

    async def check_server_status_task(self, chat_completion: bool = True) -> None:
        """
        Poll /health/ until the server is reachable, then set server_healthy=True.

        Runs as a background task; failures are logged rather than raised so
        that callers are blocked by the semaphore / while-loop in the base class
        rather than seeing an uncaught exception.
        """
        url = self._url("/health/")
        backoff = 1.0
        while True:
            try:
                session = await self._get_session()
                async with session.get(url) as resp:
                    if resp.status == 200:
                        self.server_healthy = True
                        logger.info(
                            "TrlVllmServer is healthy at %s", self.config.base_url
                        )
                        return
                    else:
                        logger.warning(
                            "Health check returned HTTP %d, retrying in %.1fs …",
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
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST *payload* to *path* and return the parsed JSON body.

        Raises aiohttp.ClientResponseError on non-2xx status codes so the
        tenacity retry in the base class can catch and retry.
        """
        url = self._url(path)
        session = await self._get_session()
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _get(self, path: str) -> Dict[str, Any]:
        """GET *path* and return the parsed JSON body."""
        url = self._url(path)
        session = await self._get_session()
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------
    # Internal generation helpers
    # ------------------------------------------------------------------

    def _build_sampling_payload(self, **kwargs) -> Dict[str, Any]:
        """
        Map standard completion kwargs to the trl vLLM server's JSON schema.

        Extra keys that are not recognised by the trl server are silently
        dropped so that the caller does not need to scrub the kwargs dict.
        """
        payload: Dict[str, Any] = {
            "n": kwargs.get("n", 1),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", -1),
            "min_p": kwargs.get("min_p", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1024),
            # Always request at least the sampled token's logprob.
            # The caller may pass logprobs=N to get top-N+1 logprobs.
            "logprobs": kwargs.get("logprobs", 0),
        }
        # Forward optional structured-output regex
        if kwargs.get("structured_outputs_regex") is not None:
            payload["structured_outputs_regex"] = kwargs["structured_outputs_regex"]
        # Forward arbitrary extra generation kwargs (seed, frequency_penalty, …)
        if kwargs.get("generation_kwargs"):
            payload["generation_kwargs"] = kwargs["generation_kwargs"]
        return payload

    # ------------------------------------------------------------------
    # Chat completion wrapper
    # ------------------------------------------------------------------

    async def _chat_completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Call POST /chat/ and reconstruct an OpenAI-compatible ChatCompletion.

        Accepts:
            messages  : list[dict] — standard OpenAI chat messages
            n         : int
            max_tokens: int
            temperature, top_p, top_k, min_p, repetition_penalty
            logprobs  : int  (number of extra top-k logprobs; 0 = sampled only)
            structured_outputs_regex, generation_kwargs, chat_template_kwargs, tools
        """
        messages = kwargs.get("messages", [])
        if not isinstance(messages[0], list):
            # The trl server expects a batch (list of conversations).
            messages_batch = [messages]
        else:
            messages_batch = messages

        payload = self._build_sampling_payload(**kwargs)
        payload["messages"] = messages_batch

        # Optional chat-template extras
        if kwargs.get("chat_template_kwargs"):
            payload["chat_template_kwargs"] = kwargs["chat_template_kwargs"]
        if kwargs.get("tools") is not None:
            payload["tools"] = kwargs["tools"]

        data = await self._post("/chat/", payload)

        completion_ids: List[List[int]] = data.get("completion_ids", [])

        choices = []
        for i, token_ids in enumerate(completion_ids):
            finish_reason = "length"
            if self.tokenizer is not None:
                # Detect EOS by checking if the last generated token is EOS
                eos_ids = set()
                if hasattr(self.tokenizer, "eos_token_id"):
                    eid = self.tokenizer.eos_token_id
                    if isinstance(eid, int):
                        eos_ids.add(eid)
                    elif isinstance(eid, list):
                        eos_ids.update(eid)
                if token_ids and token_ids[-1] in eos_ids:
                    finish_reason = "stop"
            elif token_ids:
                # Cannot determine EOS without tokenizer; default to stop
                finish_reason = "stop"

            if self.tokenizer is not None:
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            else:
                text = ""
                logger.warning(
                    "No tokenizer available; completion content will be empty."
                )

            choices.append(
                Choice(
                    finish_reason=finish_reason,
                    index=i,
                    message=ChatCompletionMessage(
                        content=text,
                        role="assistant",
                    ),
                )
            )

        return ChatCompletion(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=self.config.model_name,
            choices=choices,
        )

    # ------------------------------------------------------------------
    # Completion wrapper
    # ------------------------------------------------------------------

    async def _completion_wrapper(self, **kwargs) -> Completion:
        """
        Call POST /generate/ and reconstruct an OpenAI-compatible Completion.

        Accepts:
            prompt    : str | list[int]  — text prompt or pre-tokenized IDs
            input_ids : list[int]        — alternative to prompt (ManagedServer
                        multi-turn path passes this instead of re-encoding)
            n, max_tokens, temperature, top_p, top_k, min_p, repetition_penalty,
            logprobs, structured_outputs_regex, generation_kwargs
        """
        prompt = kwargs.get("prompt", "")
        input_ids: Optional[List[int]] = kwargs.get("input_ids")

        payload = self._build_sampling_payload(**kwargs)

        if input_ids is not None:
            # Use pre-tokenized IDs to avoid double-tokenisation in multi-turn
            payload["prompts"] = [input_ids]
        elif isinstance(prompt, list):
            payload["prompts"] = [prompt]
        else:
            payload["prompts"] = [prompt]

        data = await self._post("/generate/", payload)

        completion_ids: List[List[int]] = data.get("completion_ids", [])

        choices = []
        for i, token_ids in enumerate(completion_ids):
            finish_reason = "length"
            if self.tokenizer is not None:
                eos_ids = set()
                if hasattr(self.tokenizer, "eos_token_id"):
                    eid = self.tokenizer.eos_token_id
                    if isinstance(eid, int):
                        eos_ids.add(eid)
                    elif isinstance(eid, list):
                        eos_ids.update(eid)
                if token_ids and token_ids[-1] in eos_ids:
                    finish_reason = "stop"
            else:
                finish_reason = "stop"

            if self.tokenizer is not None:
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            else:
                text = ""

            choices.append(
                CompletionChoice(
                    finish_reason=finish_reason,
                    index=i,
                    text=text,
                )
            )

        return Completion(
            id=str(uuid.uuid4()),
            object="text_completion",
            created=int(time.time()),
            model=self.config.model_name,
            choices=choices,
        )

    # ------------------------------------------------------------------
    # Tokens-and-logprobs wrapper  ← the key ManagedServer interface
    # ------------------------------------------------------------------

    async def _tokens_and_logprobs_completion_wrapper(
        self, **kwargs
    ) -> Tuple[List[int], List[List[int]], List[List[float]], List[str]]:
        """
        Generate completions and return raw tokens + per-token logprobs.

        This is the primary method used by ManagedServer.  It always requests
        at least the sampled token's logprob (logprobs=0 in vLLM terms) so
        that ManagedServer can build the aligned SequenceNode it needs.

        Supports both text prompts and pre-tokenized input_ids (the ManagedServer
        multi-turn path supplies input_ids to avoid double-tokenisation).

        Supports both /generate/ (prompt-based) and /chat/ (message-based)
        paths: if ``messages`` is present in kwargs the /chat/ endpoint is
        used, otherwise /generate/.

        Returns:
            prompt_tokens      : list[int]  — token IDs of the prompt
            output_tokens_list : list[list[int]]  — one list per completion
            output_logprobs_list: list[list[float]] — sampled-token logprob per
                                   completion token, aligned with output_tokens
            finish_reasons     : list[str]  — "stop" or "length" per completion
        """
        messages = kwargs.get("messages")
        use_chat = messages is not None

        # Ensure we always get at least the sampled token logprob.
        kwargs.setdefault("logprobs", 0)

        if use_chat:
            # --- /chat/ path ---
            if not isinstance(messages[0], list):
                messages_batch = [messages]
            else:
                messages_batch = messages

            payload = self._build_sampling_payload(**kwargs)
            payload["messages"] = messages_batch
            if kwargs.get("chat_template_kwargs"):
                payload["chat_template_kwargs"] = kwargs["chat_template_kwargs"]
            if kwargs.get("tools") is not None:
                payload["tools"] = kwargs["tools"]

            data = await self._post("/chat/", payload)
        else:
            # --- /generate/ path ---
            input_ids: Optional[List[int]] = kwargs.get("input_ids")
            prompt = kwargs.get("prompt", "")

            payload = self._build_sampling_payload(**kwargs)

            if input_ids is not None:
                payload["prompts"] = [input_ids]
            elif isinstance(prompt, list):
                payload["prompts"] = [prompt]
            else:
                payload["prompts"] = [prompt]

            data = await self._post("/generate/", payload)

        # ── Unpack response ──────────────────────────────────────────────
        prompt_ids_batched: List[List[int]] = data.get("prompt_ids", [[]])
        completion_ids: List[List[int]] = data.get("completion_ids", [])
        logprobs_nested: Optional[List[List[List[Optional[float]]]]] = data.get(
            "logprobs"
        )

        # prompt_ids comes back as one list per prompt; we submitted one prompt
        prompt_tokens: List[int] = prompt_ids_batched[0] if prompt_ids_batched else []

        # ── Build flat sampled logprob lists, one per completion ─────────
        # logprobs_nested shape: (num_completions, seq_len, num_top_k)
        # We take the first (highest-ranked) logprob per token as the
        # "sampled token logprob" — when logprobs=0 the server always puts the
        # sampled token in position 0.
        sampled_logprobs_list: List[List[float]] = _extract_sampled_logprobs(
            logprobs_nested
        )

        # Pad/truncate to match token sequence length
        output_tokens_list: List[List[int]] = []
        output_logprobs_list: List[List[float]] = []
        finish_reasons: List[str] = []

        # Determine EOS token IDs once
        eos_ids: set = set()
        if self.tokenizer is not None and hasattr(self.tokenizer, "eos_token_id"):
            eid = self.tokenizer.eos_token_id
            if isinstance(eid, int):
                eos_ids.add(eid)
            elif isinstance(eid, list):
                eos_ids.update(eid)

        for i, token_ids in enumerate(completion_ids):
            # Finish reason
            if token_ids and token_ids[-1] in eos_ids:
                finish_reason = "stop"
            elif not token_ids:
                finish_reason = "length"
            else:
                # Fallback: if last token is not EOS and we reached max_tokens → length
                finish_reason = (
                    "length"
                    if len(token_ids) >= kwargs.get("max_tokens", 1024)
                    else "stop"
                )

            # Logprobs for this completion
            if i < len(sampled_logprobs_list):
                raw_lps = sampled_logprobs_list[i]
            else:
                raw_lps = []

            # Align logprobs length to token length
            if len(raw_lps) < len(token_ids):
                raw_lps = raw_lps + [0.0] * (len(token_ids) - len(raw_lps))
            elif len(raw_lps) > len(token_ids):
                raw_lps = raw_lps[: len(token_ids)]

            output_tokens_list.append(token_ids)
            output_logprobs_list.append(raw_lps)
            finish_reasons.append(finish_reason)

        return prompt_tokens, output_tokens_list, output_logprobs_list, finish_reasons

    # ------------------------------------------------------------------
    # Prompt logprobs wrapper  ← used by ManagedServer.get_logprobs()
    # ------------------------------------------------------------------

    async def _get_logprobs_wrapper(self, **kwargs) -> Dict[str, Any]:
        """
        Compute per-position top-k logprobs for an existing token sequence via
        POST /get_sequence_logprobs/.

        This enables ManagedServer's ``get_logprobs()`` API which downstream
        trainers use for KL-divergence / teacher-forcing objectives.

        Accepts (keyword args):
            prompt       : str        — text to evaluate (tokenized internally)
            input_ids    : list[int]  — pre-tokenized full sequence
            messages     : list[dict] — alternative to prompt; converted via
                           apply_chat_template before calling this method
            top_k        : int        — number of top logprobs per position
                           (default 1; high values needed for forward-KL)
            top_logprobs : int        — alias for top_k
            temperature  : float      — (default 1.0)
            response_format : str     — "json" or "binary" (default "binary"
                              when numpy is available, else "json")

        Returns (normalized ManagedServer schema):
            {
                "prompt_tokens"         : list[int],
                "prompt_topk_token_ids" : list[list[int]],   # [pos][k]
                "prompt_topk_logprobs"  : list[list[float]], # [pos][k]
            }
        """
        input_ids: Optional[List[int]] = kwargs.get("input_ids")
        prompt: Optional[str] = kwargs.get("prompt")

        # Resolve token IDs
        if input_ids is not None:
            full_ids = input_ids
        elif prompt is not None:
            if self.tokenizer is None:
                raise RuntimeError(
                    "TrlVllmServer requires a tokenizer to encode prompts for "
                    "_get_logprobs_wrapper.  Provide input_ids instead."
                )
            full_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            raise ValueError(
                "_get_logprobs_wrapper requires either 'input_ids' or 'prompt'."
            )

        top_k: int = kwargs.get("top_logprobs", kwargs.get("top_k", 1))
        temperature: float = float(kwargs.get("temperature", 1.0))
        prompt_length: int = kwargs.get("prompt_length", 0)

        # Prefer binary format for efficiency if numpy is present
        try:
            import numpy as _np  # noqa: F401

            response_format = kwargs.get("response_format", "binary")
        except ImportError:
            response_format = "json"

        payload = {
            "sequences": [full_ids],
            "prompt_lengths": [prompt_length],
            "top_logprobs": top_k,
            "temperature": temperature,
            "response_format": response_format,
        }

        data = await self._post("/get_sequence_logprobs/", payload)

        if response_format == "binary":
            logprobs_list, token_ids_list, _, _ = _decode_binary_logprobs(data)
        else:
            logprobs_list = data.get("logprobs", [[]])
            token_ids_list = data.get("logprob_token_ids", [[]])

        # We submitted a single sequence, take index 0
        seq_lps = logprobs_list[0] if logprobs_list else []
        seq_tids = token_ids_list[0] if token_ids_list else []

        return {
            "prompt_tokens": full_ids,
            "prompt_topk_token_ids": seq_tids,
            "prompt_topk_logprobs": seq_lps,
        }

    # ------------------------------------------------------------------
    # /generate/ — public batch-generation endpoint
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompts: List[str | List[int]],
        *,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 1024,
        logprobs: int = 0,
        structured_outputs_regex: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        images: Optional[List[Optional[List[str]]]] = None,
    ) -> Dict[str, Any]:
        """
        POST /generate/ — batch raw-prompt generation.

        Args:
            prompts: List of text strings or pre-tokenized token-ID lists.
            images:  Optional per-prompt list of base64-encoded images.
            n:       Number of completions per prompt.
            logprobs: Number of extra top-k logprobs to return (0 = sampled only).
            structured_outputs_regex: Optional regex for constrained generation.
            generation_kwargs: Extra SamplingParams (seed, frequency_penalty, …).

        Returns:
            Raw server JSON::

                {
                    "prompt_ids"         : list[list[int]],
                    "completion_ids"     : list[list[int]],
                    "logprobs"           : list[list[list[float|None]]]|None,
                    "logprob_token_ids"  : list[list[list[int]]]|None,
                }
        """
        payload: Dict[str, Any] = {
            "prompts": prompts,
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
        }
        if images is not None:
            payload["images"] = images
        if structured_outputs_regex is not None:
            payload["structured_outputs_regex"] = structured_outputs_regex
        if generation_kwargs:
            payload["generation_kwargs"] = generation_kwargs

        return await self._post("/generate/", payload)

    # ------------------------------------------------------------------
    # /chat/ — public batch-chat endpoint
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: List[List[Dict[str, Any]]],
        *,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 1024,
        logprobs: int = 0,
        structured_outputs_regex: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        POST /chat/ — batch chat-message generation.

        Args:
            messages: Batch of conversations; each item is a list of
                      ``{"role": ..., "content": ...}`` dicts.
            n, temperature, top_p, top_k, min_p, max_tokens, logprobs,
            repetition_penalty: Standard sampling parameters.
            structured_outputs_regex: Optional regex for constrained generation.
            generation_kwargs: Extra SamplingParams forwarded verbatim.
            chat_template_kwargs: Extra kwargs for apply_chat_template.
            tools: OpenAI-format tool list for tool-call-aware generation.

        Returns:
            Raw server JSON (same schema as /generate/).
        """
        payload: Dict[str, Any] = {
            "messages": messages,
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
        }
        if structured_outputs_regex is not None:
            payload["structured_outputs_regex"] = structured_outputs_regex
        if generation_kwargs:
            payload["generation_kwargs"] = generation_kwargs
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if tools is not None:
            payload["tools"] = tools

        return await self._post("/chat/", payload)

    # ------------------------------------------------------------------
    # /get_sequence_logprobs/ — teacher-logprob / KD endpoint
    # ------------------------------------------------------------------

    async def get_sequence_logprobs(
        self,
        sequences: List[List[int]],
        prompt_lengths: List[int],
        *,
        top_logprobs: int = 100,
        temperature: float = 1.0,
        response_format: str = "json",
    ) -> Dict[str, Any]:
        """
        POST /get_sequence_logprobs/ — compute teacher logprobs for existing sequences.

        Concurrent calls are batched automatically by the trl server's internal
        batcher, so it is safe to call this concurrently from many coroutines.

        Args:
            sequences:      Full token sequences (prompt + completion) per sample.
            prompt_lengths: Number of prompt tokens per sequence; completion
                            logprobs start after the prompt portion.
            top_logprobs:   Number of top-k logprobs per completion position.
            temperature:    Temperature applied to logits before ranking.
            response_format: ``"json"`` (nested lists) or ``"binary"``
                             (base64 numpy arrays — faster for large batches).

        Returns:
            For ``response_format="json"``::

                {
                    "logprobs"           : list[list[list[float|None]]],
                    "logprob_token_ids"  : list[list[list[int]]],
                }

            For ``response_format="binary"``::

                {
                    "logprobs_b64"        : str,   # float32 base64
                    "token_ids_b64"       : str,   # int32  base64
                    "actual_logprobs_b64" : str,
                    "actual_token_ids_b64": str,
                    "shape"               : [batch, max_comp_len, top_k],
                    "completion_lengths"  : list[int],
                }
        """
        payload = {
            "sequences": sequences,
            "prompt_lengths": prompt_lengths,
            "top_logprobs": top_logprobs,
            "temperature": temperature,
            "response_format": response_format,
        }
        return await self._post("/get_sequence_logprobs/", payload)

    # ------------------------------------------------------------------
    # Info endpoints
    # ------------------------------------------------------------------

    async def health(self) -> Dict[str, Any]:
        """GET /health/ — liveness probe.  Returns ``{"status": "ok"}``."""
        return await self._get("/health/")
