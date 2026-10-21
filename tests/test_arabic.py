"""Fast tests: no models are downloaded. Run with `pytest tests/`."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import app  # noqa: E402
import arabic  # noqa: E402


# ---- dialect detection -------------------------------------------------------

@pytest.mark.parametrize("dialect,text", [
    ("gulf", "شلونك الحين؟ والله وايد زحمة، ابغى أروح البيت"),
    ("levantine", "هلق بدي روح، كان منيح كتير هيك"),
    ("egyptian", "أنا عايز أعرف إزاي ده حصل كده دلوقتي"),
    ("maghrebi", "دابا غادي نمشي، كاين بزاف ديال الناس، واش فهمتي؟"),
])
def test_detects_each_dialect(dialect, text):
    assert arabic.detect_dialect(text)[0] == dialect


def test_unclear_text_falls_back_to_msa():
    assert arabic.detect_dialect("افتتحت الجلسة وتفضل المحامي بالمرافعة")[0] == "msa"
    assert arabic.detect_dialect("")[0] == "msa"


def test_single_marker_is_not_enough():
    assert arabic.detect_dialect("قال دابا ثم سكت")[0] == "msa"


def test_attached_conjunction_counts():
    assert arabic.dialect_scores("ودابا وبزاف")["maghrebi"] == 2


def test_every_dialect_has_a_primer():
    assert set(arabic.PRIMERS) == set(arabic.DIALECTS)


# ---- normalisation for scoring ----------------------------------------------

def test_arabic_normalisation():
    assert arabic.normalize_arabic("إِنَّ المَحْكَمَةَ") == arabic.normalize_arabic("ان المحكمه")
    assert arabic.normalize_arabic("علىٰ") == arabic.normalize_arabic("علي")
    assert arabic.normalize_arabic("الجلسة ٣، OK؟") == "الجلسه 3 ok"


def test_metrics_routes_arabic_to_arabic_normaliser():
    metrics = pytest.importorskip("metrics")
    c = metrics.wer_counts([{"text": "إنَّ المحكمةَ، منعقدة."}], [{"text": "ان المحكمه منعقده"}])
    assert c["sub"] + c["del"] + c["ins"] == 0
    c = metrics.wer_counts([{"text": "Hello, world."}], [{"text": "hello world"}])
    assert c["sub"] + c["del"] + c["ins"] == 0


# ---- pipeline wiring with stand-in models -------------------------------------

class FakeASR:
    def __init__(self, probe_text="", text="نص"):
        self.calls = []
        self.probe_text, self.text = probe_text, text

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        word = SimpleNamespace(start=0.0, end=1.0, word=" " + self.text)
        text = self.probe_text if len(self.calls) == 1 and kw.get("language") == "ar" \
            and "hotwords" not in kw and "word_timestamps" not in kw else self.text
        seg = SimpleNamespace(start=0.0, end=1.0, text=text, words=[word])
        return iter([seg]), SimpleNamespace(duration=1.0, language=kw.get("language") or "ar")


class FakeDiar:
    def __init__(self):
        self.kwargs = None

    def __call__(self, wav, **kw):
        self.kwargs = kw
        turn = SimpleNamespace(start=0.0, end=1.0)
        return SimpleNamespace(itertracks=lambda yield_label: [(turn, None, "SPEAKER_00")])


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    return str(path)


def test_english_request_has_no_dialect_or_primer(wav):
    asr = FakeASR(text="hello")
    out = app.run_pipeline(wav, asr, None, app.Request(language="en", vocabulary=""))
    assert out["dialect"] is None and len(asr.calls) == 1
    assert "hotwords" not in asr.calls[0] and asr.calls[0]["language"] == "en"


def test_fixed_dialect_sends_primer_on_every_window(wav):
    asr = FakeASR()
    out = app.run_pipeline(wav, asr, None, app.Request(language="ar", dialect="egyptian", vocabulary=""))
    assert out["dialect"] == "egyptian" and out["dialect_detected"] is False
    assert len(asr.calls) == 1  # no probe when the dialect is given
    assert asr.calls[0]["hotwords"] == arabic.PRIMERS["egyptian"]


def test_auto_dialect_probes_then_uses_detected_primer(wav):
    asr = FakeASR(probe_text="شلونك الحين وايد زين")
    out = app.run_pipeline(wav, asr, None, app.Request(language="ar", dialect="auto", vocabulary=""))
    assert out["dialect"] == "gulf" and out["dialect_detected"] is True
    assert len(asr.calls) == 2 and asr.calls[1]["hotwords"].startswith(arabic.PRIMERS["gulf"])


def test_mixed_language_detects_per_segment_and_adds_vocabulary(wav):
    asr = FakeASR()
    req = app.Request(language="mixed", dialect="levantine", vocabulary="Emirates NBD, القاضي سالم")
    app.run_pipeline(wav, asr, None, req)
    final = asr.calls[-1]
    assert final["multilingual"] is True and final["language"] is None
    assert "Emirates NBD" in final["hotwords"] and arabic.PRIMERS["levantine"] in final["hotwords"]


def test_speaker_count_reaches_diarization(wav):
    diar = FakeDiar()
    out = app.run_pipeline(wav, FakeASR(text="hello"), diar, app.Request(language="en", num_speakers=3))
    assert diar.kwargs == {"num_speakers": 3}
    assert out["segments"][0]["speaker"] == "SPEAKER_00"


def test_min_max_speakers_when_count_unknown():
    assert app.speaker_hints(app.Request(min_speakers=2, max_speakers=5)) == {"min_speakers": 2, "max_speakers": 5}
    assert app.speaker_hints(app.Request()) == {}


# ---- request validation -------------------------------------------------------

def test_build_request_validates():
    from fastapi import HTTPException

    req = app.build_request("AR", "Gulf", "Dubai Courts", 3, None, None)
    assert (req.language, req.dialect, req.num_speakers) == ("ar", "gulf", 3)
    assert "Dubai Courts" in req.vocabulary
    for bad in (("fr", None), (None, "iraqi")):
        with pytest.raises(HTTPException):
            app.build_request(*bad, None, None, None, None)
    with pytest.raises(HTTPException):
        app.build_request(None, None, None, 0, None, None)


def test_arabic_requests_use_arabic_model():
    assert app.model_for("ar") == app.ARABIC_MODEL
    assert app.model_for("mixed") == app.ARABIC_MODEL
    assert app.model_for("en") == app.WHISPER_MODEL
