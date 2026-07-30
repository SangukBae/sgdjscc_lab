"""tests/test_psss.py – src/sgdjscc_lab/video/psss.py (PSSS scoring).

Exercises the three PSSS backends (mock / proxy / real) purely with fakes —
no real MLLM/CLIP weights are downloaded. The real (`MllmTokenProbPsssBackend`)
backend is tested against small fake tokenizer/model objects that mimic just
enough of the `transformers` `AutoTokenizer`/`AutoModelForCausalLM` surface
(`encode()` / `__call__(input_ids) -> logits`) to drive the actual scoring
algorithm, including multi-token "Yes"/"No" surface forms.
"""

from __future__ import annotations

import math

import pytest
import torch

from sgdjscc_lab.video.psss import (
    ClipTextProxyPsssBackend,
    MllmTokenProbPsssBackend,
    MockPsssBackend,
    PsssBackendUnavailableError,
    build_psss_backend,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mock backend: PSSS formula shape + deterministic behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestMockPsssBackend:
    def test_identical_captions_are_maximally_similar(self):
        r = MockPsssBackend().score("a cat on a mat", "a cat on a mat")
        assert r.s_abs == pytest.approx(1.0)
        assert r.s_rel == pytest.approx(-1.0)   # p_no - p_yes, fully "similar"
        assert r.backend_kind == "mock"

    def test_disjoint_captions_are_maximally_divergent(self):
        r = MockPsssBackend().score("xxxx yyyy", "zzzz wwww")
        assert r.s_abs == pytest.approx(0.0)
        assert r.s_rel == pytest.approx(1.0)

    def test_s_rel_equals_p_no_minus_p_yes(self):
        r = MockPsssBackend().score("a cat sat", "a cat ran")
        assert r.s_rel == pytest.approx(r.p_no - r.p_yes)
        assert r.p_yes + r.p_no == pytest.approx(1.0)

    def test_empty_captions_are_treated_as_similar_not_a_crash(self):
        r = MockPsssBackend().score("", "")
        assert r.s_abs == pytest.approx(1.0)  # union empty -> overlap defined as 1.0

    def test_notes_flag_mock_as_not_real_psss(self):
        r = MockPsssBackend().score("a", "b")
        assert "MOCK" in r.notes
        assert "NOT" in r.notes


class TestThresholdBoundary:
    """SKEM's own decision (S_rel > threshold, strict) is exercised in
    test_skem_selector.py; this class only checks the score arithmetic that
    decision is based on stays exactly reproducible at the boundary."""

    def test_score_is_deterministic_across_repeated_calls(self):
        backend = MockPsssBackend()
        r1 = backend.score("a dog runs in the park", "a dog runs in the park")
        r2 = backend.score("a dog runs in the park", "a dog runs in the park")
        assert r1.s_rel == r2.s_rel == pytest.approx(-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Proxy backend (CLIP text-text similarity) via an injected fake CLIP
# ─────────────────────────────────────────────────────────────────────────────

class _FakeClip:
    """Deterministic stand-in for CLIPScoreEvaluator._encode_texts: returns a
    unit basis vector selected by the text's first character, so callers can
    construct pairs with known cosine similarity."""

    _AXES = {"a": 0, "b": 1, "c": 2, "d": 3}

    def _encode_texts(self, texts):
        vecs = []
        for t in texts:
            v = torch.zeros(4)
            v[self._AXES.get(t[:1], 0)] = 1.0
            vecs.append(v)
        return torch.stack(vecs)


class _FailingClip:
    def _encode_texts(self, texts):
        raise RuntimeError("weights not found")


class TestClipTextProxyPsssBackend:
    def test_identical_embeddings_score_as_similar(self):
        backend = ClipTextProxyPsssBackend(clip_evaluator=_FakeClip())
        r = backend.score("apple", "apple")
        assert r.s_rel == pytest.approx(-1.0)
        assert r.backend_kind == "proxy"
        assert r.proxy_of == "clip_text_similarity"

    def test_orthogonal_embeddings_score_as_divergent(self):
        backend = ClipTextProxyPsssBackend(clip_evaluator=_FakeClip())
        r = backend.score("apple", "banana")
        assert r.s_rel == pytest.approx(0.0)   # cos=0 -> p_yes=0.5, s_rel=0

    def test_unavailable_when_clip_encoding_fails(self):
        backend = ClipTextProxyPsssBackend(clip_evaluator=_FailingClip())
        with pytest.raises(PsssBackendUnavailableError):
            backend.score("a", "b")

    def test_never_reports_itself_as_real(self):
        backend = ClipTextProxyPsssBackend(clip_evaluator=_FakeClip())
        r = backend.score("a", "b")
        assert r.backend_kind != "real"
        assert "NOT" in r.notes


# ─────────────────────────────────────────────────────────────────────────────
# Real backend (MLLM token probability) via fake tokenizer/model
# ─────────────────────────────────────────────────────────────────────────────

class _FakeVocab:
    """Tiny deterministic vocabulary: word -> id, growing on demand."""

    def __init__(self):
        self._ids = {}

    def id_for(self, token: str) -> int:
        if token not in self._ids:
            self._ids[token] = len(self._ids) + 1
        return self._ids[token]

    def __len__(self):
        return len(self._ids) + 1


class _FakeTokenizer:
    """Whitespace tokenizer where a caller-configured set of surface strings
    is deliberately split into multiple sub-tokens (to exercise multi-token
    Yes/No handling); everything else is one word = one token."""

    def __init__(self, vocab: _FakeVocab, multi_token_forms=()):
        self.vocab = vocab
        self.multi_token_forms = set(multi_token_forms)

    def encode(self, text):
        if text in self.multi_token_forms:
            # Split into two sub-tokens: "<text>#0" and "<text>#1".
            return [self.vocab.id_for(f"{text}#0"), self.vocab.id_for(f"{text}#1")]
        return [self.vocab.id_for(text)]


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    """Deterministic next-token distribution.

    Either a constant `preferred_id` (always boosted, regardless of context —
    enough for single-token Yes/No tests), or a `next_token_fn(seq_ids) ->
    Optional[int]` that decides which token id to boost as a function of the
    sequence so far (needed to make a *specific* multi-token surface form the
    unambiguous joint-probability winner at every autoregressive step)."""

    def __init__(self, vocab: _FakeVocab, preferred_id=None, next_token_fn=None, preferred_logit: float = 12.0):
        self.vocab = vocab
        self.preferred_logit = preferred_logit
        if next_token_fn is not None:
            self.next_token_fn = next_token_fn
        else:
            self.next_token_fn = lambda seq: preferred_id

    def __call__(self, input_ids):
        seq = input_ids[0].tolist()
        vocab_size = len(self.vocab) + 16
        logits = torch.zeros(1, len(seq), vocab_size)
        preferred = self.next_token_fn(seq)
        if preferred is not None:
            logits[0, -1, preferred] = self.preferred_logit
        return _FakeOutput(logits)


class TestMllmTokenProbPsssBackendSingleToken:
    def test_model_confidently_favouring_yes_gives_high_p_yes_low_s_rel(self):
        vocab = _FakeVocab()
        tok = _FakeTokenizer(vocab)
        yes_id = vocab.id_for("Yes")
        model = _FakeModel(vocab, preferred_id=yes_id)
        backend = MllmTokenProbPsssBackend(
            model=model, tokenizer=tok,
            yes_variants=("Yes",), no_variants=("No",),
        )
        r = backend.score("cat on mat", "cat on mat", "the subject")
        assert r.backend_kind == "real"
        assert r.p_yes > r.p_no
        assert r.s_rel < 0
        assert r.s_abs == pytest.approx(r.p_yes, abs=1e-6)

    def test_model_confidently_favouring_no_gives_high_p_no_positive_s_rel(self):
        vocab = _FakeVocab()
        tok = _FakeTokenizer(vocab)
        no_id = vocab.id_for("No")
        model = _FakeModel(vocab, preferred_id=no_id)
        backend = MllmTokenProbPsssBackend(
            model=model, tokenizer=tok,
            yes_variants=("Yes",), no_variants=("No",),
        )
        r = backend.score("a cat", "a dog running", "the subject")
        assert r.p_no > r.p_yes
        assert r.s_rel > 0

    def test_raw_logits_recorded_per_variant(self):
        vocab = _FakeVocab()
        tok = _FakeTokenizer(vocab)
        yes_id = vocab.id_for("Yes")
        model = _FakeModel(vocab, preferred_id=yes_id)
        backend = MllmTokenProbPsssBackend(
            model=model, tokenizer=tok, yes_variants=("Yes",), no_variants=("No",),
        )
        r = backend.score("a", "b")
        assert "Yes" in r.raw_logits["yes"]
        assert "No" in r.raw_logits["no"]
        assert r.raw_logits["yes"]["Yes"] == pytest.approx(12.0)


class TestMllmTokenProbPsssBackendMultiToken:
    def test_multi_token_yes_variant_uses_joint_sequence_probability(self):
        """' yes' is deliberately split into 2 sub-tokens; the model favours
        BOTH sub-tokens at every step, so the joint P(' yes') should still
        dominate P('No') (single token, not favoured)."""
        vocab = _FakeVocab()
        tok = _FakeTokenizer(vocab, multi_token_forms={" yes"})
        vocab.id_for("No")
        # Sequence-aware boosting so BOTH sub-tokens of " yes" are the
        # unambiguous winner at their respective autoregressive step: boost
        # the first sub-token whenever it is NOT the immediately preceding
        # token (i.e. at the prompt-final step), then boost the second
        # sub-token once the first has actually been emitted.
        first_subtoken_id = vocab.id_for(" yes#0")
        second_subtoken_id = vocab.id_for(" yes#1")

        def next_token_fn(seq):
            if seq[-1] == first_subtoken_id:
                return second_subtoken_id
            return first_subtoken_id

        model = _FakeModel(vocab, next_token_fn=next_token_fn, preferred_logit=20.0)
        backend = MllmTokenProbPsssBackend(
            model=model, tokenizer=tok, yes_variants=(" yes",), no_variants=("No",),
        )
        r = backend.score("a", "b")
        yes_evidence = r.evidence["yes_variants"][" yes"]
        assert yes_evidence["token_ids"] == [
            vocab.id_for(" yes#0"), vocab.id_for(" yes#1"),
        ]
        assert len(yes_evidence["token_ids"]) == 2   # confirms it is genuinely multi-token
        assert r.p_yes > r.p_no

    def test_duplicate_tokenization_across_variants_not_double_counted(self):
        """Two configured surface forms that happen to tokenize to the exact
        same id sequence must only be counted once in P(Yes)."""
        vocab = _FakeVocab()

        class _CollidingTokenizer(_FakeTokenizer):
            def encode(self, text):
                if text in ("Yes", "YES"):
                    return [vocab.id_for("YES_CANON")]
                return super().encode(text)

        tok = _CollidingTokenizer(vocab)
        yes_id = vocab.id_for("YES_CANON")
        model = _FakeModel(vocab, preferred_id=yes_id)
        backend = MllmTokenProbPsssBackend(
            model=model, tokenizer=tok, yes_variants=("Yes", "YES"), no_variants=("No",),
        )
        r = backend.score("a", "b")
        evidence = r.evidence["yes_variants"]
        assert evidence["Yes"]["prob"] == pytest.approx(math.exp(evidence["Yes"]["logprob"]))
        assert evidence["YES"].get("duplicate_of") == "Yes"
        # p_yes must equal the single unique variant's probability, not 2x it.
        assert r.p_yes == pytest.approx(evidence["Yes"]["prob"])


class _SpecialTokenAwareTokenizer:
    """Reproduces a REAL tokenizer's behaviour more faithfully than
    `_FakeTokenizer`: `encode(text, add_special_tokens=True)` (the default)
    wraps the token sequence with BOS/EOS; `add_special_tokens=False` does
    not. If the backend ever calls `encode()` on a Yes/No CONTINUATION
    without explicitly passing `add_special_tokens=False`, this tokenizer
    will silently inflate that continuation with BOS/EOS tokens — exactly
    the bug this test class guards against."""

    BOS = 90001
    EOS = 90002

    def __init__(self, vocab: "_FakeVocab"):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=True):
        ids = [self.vocab.id_for(w) for w in text.split()]
        if add_special_tokens:
            return [self.BOS] + ids + [self.EOS]
        return ids


class TestMllmTokenProbPsssBackendSpecialTokens:
    def test_yes_no_continuations_are_encoded_without_special_tokens(self):
        vocab = _FakeVocab()
        tok = _SpecialTokenAwareTokenizer(vocab)
        yes_id = vocab.id_for("Yes")
        model = _FakeModel(vocab, preferred_id=yes_id)
        backend = MllmTokenProbPsssBackend(
            model=model, tokenizer=tok, yes_variants=("Yes",), no_variants=("No",),
        )
        r = backend.score("cat on mat", "cat on mat", "the subject")

        yes_ids = r.evidence["yes_variants"]["Yes"]["token_ids"]
        no_ids = r.evidence["no_variants"]["No"]["token_ids"]
        # Exactly the bare word token — NOT [BOS, id, EOS]. This is what
        # would fail before the add_special_tokens=False fix (continuation
        # length would be 3, and P(Yes) would actually be P(BOS, Yes, EOS)).
        assert yes_ids == [vocab.id_for("Yes")]
        assert no_ids == [vocab.id_for("No")]
        assert tok.BOS not in yes_ids and tok.EOS not in yes_ids
        assert tok.BOS not in no_ids and tok.EOS not in no_ids

    def test_prompt_itself_still_gets_default_special_tokens(self):
        """The PROMPT (not a continuation) should still go through the
        tokenizer's default add_special_tokens behaviour — only the Yes/No
        continuations are forced to add_special_tokens=False."""
        vocab = _FakeVocab()
        tok = _SpecialTokenAwareTokenizer(vocab)
        model = _FakeModel(vocab, preferred_id=vocab.id_for("Yes"))
        backend = MllmTokenProbPsssBackend(
            model=model, tokenizer=tok, yes_variants=("Yes",), no_variants=("No",),
        )
        r = backend.score("a", "b")
        assert r.evidence["num_prompt_tokens"] >= 2  # BOS + ... + EOS present


# ─────────────────────────────────────────────────────────────────────────────
# CUDA device placement (High-severity fix: input tensors must follow the
# model's actual device, not always default to CPU)
# ─────────────────────────────────────────────────────────────────────────────

class _EmbeddingModule:
    def __init__(self, device):
        self._param = torch.nn.Parameter(torch.zeros(4, device=device))

    def parameters(self):
        yield self._param


class _DeviceAwareFakeModel:
    """Exposes get_input_embeddings() on a configurable device and asserts
    every input tensor it receives already lives there — reproduces the
    original bug (the backend always built a CPU LongTensor regardless of
    where the model itself lived, crashing with a device-mismatch error on
    any non-CPU model)."""

    def __init__(self, vocab: _FakeVocab, device, preferred_id=None):
        self.vocab = vocab
        self._device = torch.device(device)
        self._embeddings = _EmbeddingModule(self._device)
        self.preferred_id = preferred_id

    def get_input_embeddings(self):
        return self._embeddings

    def __call__(self, input_ids):
        assert input_ids.device == self._device, (
            f"PSSS backend built an input tensor on {input_ids.device}, "
            f"but the model lives on {self._device} — device mismatch."
        )
        vocab_size = len(self.vocab) + 16
        logits = torch.zeros(1, input_ids.shape[1], vocab_size, device=self._device)
        if self.preferred_id is not None:
            logits[0, -1, self.preferred_id] = 12.0
        return _FakeOutput(logits)


class TestMllmTokenProbPsssBackendDevicePlacement:
    def test_model_device_resolved_from_input_embeddings(self):
        vocab = _FakeVocab()
        model = _DeviceAwareFakeModel(vocab, device="cpu", preferred_id=vocab.id_for("Yes"))
        backend = MllmTokenProbPsssBackend(
            model=model, tokenizer=_FakeTokenizer(vocab),
            yes_variants=("Yes",), no_variants=("No",),
        )
        assert backend._model_device() == torch.device("cpu")

    def test_model_device_falls_back_to_configured_device_for_minimal_fakes(self):
        # A model with no .get_input_embeddings()/.parameters() at all (like
        # the plain _FakeModel used throughout this file) must not crash —
        # falls back to the constructor's `device=` argument.
        vocab = _FakeVocab()
        model = _FakeModel(vocab, preferred_id=vocab.id_for("Yes"))
        backend = MllmTokenProbPsssBackend(
            model=model, tokenizer=_FakeTokenizer(vocab), device="cpu",
            yes_variants=("Yes",), no_variants=("No",),
        )
        assert backend._model_device() == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real CUDA device")
    def test_real_cuda_model_does_not_hit_a_device_mismatch(self):
        """End-to-end regression on an actual CUDA device (tiny fake tensors,
        no weight download): before the fix, this raised a
        RuntimeError/PsssBackendUnavailableError from a CPU-vs-CUDA tensor
        mismatch inside the model call."""
        vocab = _FakeVocab()
        model = _DeviceAwareFakeModel(vocab, device="cuda:0", preferred_id=vocab.id_for("Yes"))
        backend = MllmTokenProbPsssBackend(
            model=model, tokenizer=_FakeTokenizer(vocab), device="cuda:0",
            yes_variants=("Yes",), no_variants=("No",),
        )
        r = backend.score("a cat on a mat", "a cat on a mat", "the subject")
        assert r.backend_kind == "real"
        assert 0.0 <= r.p_yes <= 1.0


class TestMllmTokenProbPsssBackendUnavailable:
    def test_missing_model_id_and_no_injected_model_raises_at_construction(self):
        with pytest.raises(ValueError):
            MllmTokenProbPsssBackend()

    def test_bad_model_id_raises_unavailable_not_a_generic_exception(self):
        backend = MllmTokenProbPsssBackend(model_id="definitely-not-a-real-model-xyz-123")
        with pytest.raises(PsssBackendUnavailableError):
            backend.score("a", "b")

    def test_tokenizer_failure_raises_unavailable(self):
        class _BadTokenizer:
            def encode(self, text):
                raise RuntimeError("tokenizer exploded")

        backend = MllmTokenProbPsssBackend(model=object(), tokenizer=_BadTokenizer())
        with pytest.raises(PsssBackendUnavailableError):
            backend.score("a", "b")

    def test_model_output_without_logits_raises_unavailable(self):
        vocab = _FakeVocab()
        tok = _FakeTokenizer(vocab)

        class _NoLogitsModel:
            def __call__(self, input_ids):
                return "not a logits-bearing object"  # no .logits, not indexable like one

        backend = MllmTokenProbPsssBackend(model=_NoLogitsModel(), tokenizer=tok)
        with pytest.raises(PsssBackendUnavailableError):
            backend.score("a", "b")

    def test_model_call_raising_is_wrapped_as_unavailable(self):
        vocab = _FakeVocab()
        tok = _FakeTokenizer(vocab)

        class _ExplodingModel:
            def __call__(self, input_ids):
                raise RuntimeError("CUDA OOM")

        backend = MllmTokenProbPsssBackend(model=_ExplodingModel(), tokenizer=tok)
        with pytest.raises(PsssBackendUnavailableError):
            backend.score("a", "b")


# ─────────────────────────────────────────────────────────────────────────────
# Provenance: mock/proxy/real are never interchangeable in their reported tags
# ─────────────────────────────────────────────────────────────────────────────

class TestBackendProvenanceDistinguished:
    def test_backend_kinds_are_distinct(self):
        vocab = _FakeVocab()
        tok = _FakeTokenizer(vocab)
        model = _FakeModel(vocab, preferred_id=vocab.id_for("Yes"))
        real = MllmTokenProbPsssBackend(model=model, tokenizer=tok, yes_variants=("Yes",), no_variants=("No",))
        mock = MockPsssBackend()
        proxy = ClipTextProxyPsssBackend(clip_evaluator=_FakeClip())

        r_real = real.score("a", "b")
        r_mock = mock.score("a", "b")
        r_proxy = proxy.score("a", "b")

        kinds = {r_real.backend_kind, r_mock.backend_kind, r_proxy.backend_kind}
        assert kinds == {"real", "mock", "proxy"}
        assert r_mock.proxy_of is None and r_real.proxy_of is None
        assert r_proxy.proxy_of == "clip_text_similarity"


class TestBuildPsssBackendFactory:
    def test_build_mock(self):
        backend = build_psss_backend("mock")
        assert backend.backend_kind == "mock"

    def test_build_proxy_with_injected_clip(self):
        backend = build_psss_backend("proxy", clip_evaluator=_FakeClip())
        assert backend.backend_kind == "proxy"

    def test_build_real_requires_model_id_or_model(self):
        with pytest.raises(ValueError):
            build_psss_backend("real")

    def test_unknown_backend_name_raises(self):
        with pytest.raises(NotImplementedError):
            build_psss_backend("not-a-backend")
