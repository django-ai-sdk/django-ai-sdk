"""A classifier that runs an ONNX model from the Hugging Face Hub in this process.

No torch is needed and nothing leaves the machine. onnxruntime and tokenizers run
the model, and huggingface-hub downloads it once. Install `django-ai-sdk[local]`.

It supports two kinds of model:

- **Zero-shot (NLI)**, the default. It answers yes/no, label and score lenses by
  checking whether the text entails a statement. A yes/no checks the lens's `yes`
  (or its question, if `yes` is left at the default). A label picks the criterion
  the text entails most, phrased as "This text is about <criterion>." unless
  the criterion is a full statement starting "This text". A score is how much the text entails `high`. Write `yes` as a
  statement about the text, like "This text asks to ignore previous instructions.",
  not as a question.
- **A fixed classifier**, set with `positive=` (the label that means yes). A model
  trained for one question, like a prompt-injection detector. It answers yes/no
  lenses with that label's probability, whatever the lens asks.

    LocalClassifier()                                        # English, 87 MB
    LocalClassifier(model=MULTILINGUAL)                      # 100 languages, 340 MB
    LocalClassifier(model="protectai/deberta-v3-base-prompt-injection-v2",
                    file="onnx/model.onnx", positive="INJECTION")

It reads only the judged text (the question or the answer), with secrets masked,
up to TEXT_CHARS and MAX_TOKENS. A longer text fails as too long. It abstains on
blank texts, on lenses about sources or tool calls, on lenses that need grounding,
and on "taken from". Until the model is loaded it answers unavailable and loads it
in the background, so the next judge is asked, FAIL_MODE applies, and reflect asks
again later. A failed load is logged and tried again after LOAD_RETRY_SECONDS.
Call `load()` at startup to have it ready.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar

from django_ai_sdk.logger import get_logger
from django_ai_sdk.oversight.judges.base import TOO_LONG, UNAVAILABLE, modules_installed
from django_ai_sdk.oversight.judges.classifier import Classifier, ranked_rationale, yes_answer
from django_ai_sdk.oversight.lenses import Flag, JudgeLens
from django_ai_sdk.oversight.types import Answer, Subject

if TYPE_CHECKING:
    from django_ai_sdk.oversight.lenses import Lens

logger = get_logger(__name__)

# Secret masks like "[SLACK_TOKEN]" become "(slack token)". A small model reads
# bracketed capitals as injected markup.
_MASK = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")

# Small English zero-shot DeBERTa, quantized to int8. MIT license.
ENGLISH = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
# Multilingual (XNLI, 100 languages). MIT license.
MULTILINGUAL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
QUANTIZED = "onnx/model_quantized.onnx"
# DeBERTa's token limit. TEXT_CHARS keeps Latin-script text well inside it. Denser
# scripts like Chinese may still overflow. Then nothing is judged (TOO_LONG).
MAX_TOKENS = 512
# Confidence of a score. Entailment of `high` is a rough scale, so it is never sure.
SCORE_CONFIDENCE = 0.6
# Seconds before a model that failed to load is tried again.
LOAD_RETRY_SECONDS = 60.0


class LocalClassifier(Classifier):
    """Answer yes/no, label and score lenses with an ONNX model in this process."""

    external: ClassVar[str] = ""
    timeout = 10.0
    name = "local"
    # About 400 tokens, leaving room for the statement within MAX_TOKENS.
    TEXT_CHARS: ClassVar[int] = 1_600

    def __init__(
        self,
        model: str = ENGLISH,
        *,
        file: str = QUANTIZED,
        positive: str = "",
        name: str = "",
        timeout: float | None = None,
    ) -> None:
        super().__init__(name=name, timeout=timeout)
        self.model = model
        self.file = file
        self.positive = positive

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "model": self.model,
            "file": self.file,
            "positive": self.positive,
        }

    def missing(self) -> str:
        if not modules_installed("onnxruntime", "tokenizers", "huggingface_hub"):
            return "needs django-ai-sdk[local] (onnxruntime, tokenizers, huggingface-hub)"
        runner = _loaded(self.model, self.file)
        wanted = self.positive.lower() or "entailment"
        if runner is not None and wanted not in runner.labels:
            hint = "" if self.positive else ": a fixed classifier needs positive="
            return f"{self.model} has no label {wanted!r} ({', '.join(runner.labels)}){hint}"
        return ""

    async def answer(self, lens: Lens, view: Any) -> Answer | None:
        assert isinstance(lens, JudgeLens)  # Only judge lenses have judges.
        # A lens it can't read, or nothing to read: nothing to say.
        if self.question(lens) is None or not view.text(lens.subject).strip():
            return None
        # Unavailable while the model loads in the background, so the first turn after
        # a deploy goes to the next judge instead of hitting the deadline.
        if _loaded(self.model, self.file) is None:
            return Answer(error=f"{UNAVAILABLE}: {_load_in_background(self.model, self.file)}")
        return await super().answer(lens, view)

    async def per_source(self, lens: JudgeLens, view: Any) -> Answer | None:
        # A zero-shot model reads one text and one statement, not an answer plus a source.
        return None

    def question(self, lens: JudgeLens) -> dict[str, Any] | None:
        # Only one text at a time. Lenses that need sources or a tool call go elsewhere.
        if lens.subject not in {Subject.INPUT, Subject.OUTPUT} or lens.needs_grounding:
            return None
        reads = "output" if lens.subject is Subject.OUTPUT else "input"
        kind = lens.answer.kind
        if self.positive:
            # A fixed classifier only answers yes/no lenses.
            return {"kind": "fixed", "reads": reads} if kind == "flag" else None
        if kind == "flag":
            # The default yes says nothing about the lens, so the question says more.
            statement = lens.yes if lens.yes not in {"", Flag.options["yes"]} else lens.question
            return {"kind": "flag", "reads": reads, "statements": [statement]}
        if kind == "choice":
            statements = [self.about(lens.criteria.get(label) or label) for label in lens.labels]
            return {
                "kind": "choice",
                "reads": reads,
                "labels": list(lens.labels),
                "statements": statements,
            }
        if kind == "score":
            return {"kind": "score", "reads": reads, "statements": [lens.high]}
        return None

    def text(self, text: str, limit: int) -> str:
        return _MASK.sub(
            lambda m: f"({m.group(1).lower().replace('_', ' ')})", super().text(text, limit)
        )

    def about(self, criterion: str) -> str:
        """Phrase a label's criterion as a statement, like "This text is about leave."."""
        text = criterion.strip().rstrip(".")
        if text.lower().startswith("this text"):
            return f"{text}."
        return f"This text is about {text[:1].lower() + text[1:]}."

    async def classify(
        self, state: dict[str, Any], questions: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Answer]:
        texts = {
            "input": state.get("user_message", ""),
            "output": state.get("assistant_reply", ""),
        }
        return await asyncio.to_thread(self._classify, texts, questions)

    def _classify(self, texts: dict[str, str], questions: dict[str, Any]) -> dict[str, Answer]:
        runner = _loaded(self.model, self.file)
        if runner is None:
            return {}
        answers: dict[str, Answer] = {}
        for name, question in questions.items():
            text = texts[question["reads"]]
            try:
                if question["kind"] == "fixed":
                    [probabilities] = runner.run([(text, None)])
                    yes = probabilities[runner.label_index(self.positive)]
                    answers[name] = yes_answer(yes, self.model, f"{self.positive} {yes:.2f}")
                    continue
                entailed = runner.entailment([(text, s) for s in question["statements"]])
            except _TooLong:
                answers[name] = Answer(error=TOO_LONG)
                continue
            if question["kind"] == "flag":
                answers[name] = yes_answer(entailed[0], self.model, f"entails {entailed[0]:.2f}")
            elif question["kind"] == "score":
                answers[name] = Answer(
                    value=entailed[0], confidence=SCORE_CONFIDENCE, model=self.model
                )
            else:
                total = sum(entailed) or 1.0
                shares = [e / total for e in entailed]
                best = max(range(len(shares)), key=shares.__getitem__)
                answers[name] = Answer(
                    value=question["labels"][best],
                    confidence=shares[best],
                    model=self.model,
                    rationale=ranked_rationale(zip(question["labels"], shares, strict=True)),
                )
        return answers


class _TooLong(Exception):
    """A text runs past MAX_TOKENS, so the model would only read its start."""


class _Runner:
    """One model's session, tokenizer and labels. `run` is thread-safe."""

    def __init__(self, model: str, file: str) -> None:
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        folder = file.rsplit("/", 1)[0] + "/" if "/" in file else ""
        self.session = onnxruntime.InferenceSession(
            hf_hub_download(model, file), providers=["CPUExecutionProvider"]
        )
        try:
            tokenizer_file = hf_hub_download(model, "tokenizer.json")
        except Exception:
            tokenizer_file = hf_hub_download(model, f"{folder}tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding()
        with open(hf_hub_download(model, "config.json"), encoding="utf-8") as f:
            id2label = json.load(f).get("id2label") or {}
        self.labels = [str(id2label[k]).lower() for k in sorted(id2label, key=int)]
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.lock = threading.Lock()

    def label_index(self, label: str) -> int:
        return self.labels.index(label.lower())

    def run(self, pairs: list[tuple[str, str | None]]) -> list[list[float]]:
        """Label probabilities for each (text, statement) pair."""
        import numpy as np

        if all(statement is None for _, statement in pairs):
            encoded = self.tokenizer.encode_batch([text for text, _ in pairs])
        else:
            encoded = self.tokenizer.encode_batch([(text, s or "") for text, s in pairs])
        if any(e.overflowing for e in encoded):
            # The tokenizer cut the end off, and a verdict on the rest would hide it.
            raise _TooLong
        feed = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
        }
        with self.lock:
            [logits] = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})
        logits = np.asarray(logits, dtype=np.float64)
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return (exp / exp.sum(axis=-1, keepdims=True)).tolist()

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Entailment versus contradiction for each pair, ignoring "neutral".

        Matches how Hugging Face's zero-shot pipeline scores it.
        """
        index = self.label_index("entailment")
        against = next((i for i, label in enumerate(self.labels) if "contradiction" in label), None)
        if against is None:
            against = next(i for i, label in enumerate(self.labels) if i != index)
        scores = []
        for row in self.run(list(pairs)):
            yes, no = row[index], row[against]
            scores.append(yes / (yes + no) if yes + no else 0.5)
        return scores


# Loaded models, keyed by (model, file), and for each failed load its error and time.
_runners: dict[tuple[str, str], _Runner] = {}
_load_errors: dict[tuple[str, str], tuple[str, float]] = {}
_loading: set[tuple[str, str]] = set()
# Held while a model loads. The short one guards the two above, so a turn never waits.
_load_lock = threading.Lock()
_state_lock = threading.Lock()


def _loaded(model: str, file: str) -> _Runner | None:
    return _runners.get((model, file))


def load(model: str = ENGLISH, file: str = QUANTIZED) -> None:
    """Load a model now, e.g. at startup, so no turn waits for it."""
    key = (model, file)
    with _load_lock:
        if key in _runners:
            return
        try:
            runner = _Runner(model, file)
        except Exception as exc:
            with _state_lock:
                _load_errors[key] = (type(exc).__name__, time.monotonic())
            raise
        _runners[key] = runner
        with _state_lock:
            _load_errors.pop(key, None)


def _load_in_background(model: str, file: str) -> str:
    """Load the model in a thread, unless it is loading or failed lately. Says which."""
    key = (model, file)
    with _state_lock:
        failed = _load_errors.get(key)
        if failed and time.monotonic() - failed[1] < LOAD_RETRY_SECONDS:
            return f"does not load: {failed[0]}"
        if key in _loading:
            return "loading"
        _loading.add(key)

    def run() -> None:
        try:
            load(model, file)
        except Exception as exc:
            logger.opt(exception=exc).warning("Local model {} does not load", model)
        finally:
            with _state_lock:
                _loading.discard(key)

    threading.Thread(target=run, name=f"oversight-load-{model}", daemon=True).start()
    return "loading"


__all__ = ["ENGLISH", "MULTILINGUAL", "LocalClassifier", "load"]
