import mimetypes
from pathlib import Path
from elevenlabs import ElevenLabs
from typing import Any, Dict
from mcp2.config import settings
from openai import OpenAI


def transcribe_audio_openai(
    audio_path: str, model: str = "whisper-1"
) -> Dict[str, Any]:
    """
    Transcribe an audio file using OpenAI Whisper API via openai>=1.x client.
    Returns { text }.
    """

    api_key = settings.openai_api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    client = OpenAI(api_key=api_key)
    with path.open("rb") as f:
        res = client.audio.transcriptions.create(model=model, file=f)
    return {
        "text": getattr(res, "text", None) or getattr(res, "data", {}).get("text", "")
    }


def transcribe_audio_eleven(
    audio_path: str, model: str = "scribe_v1"
) -> Dict[str, Any]:
    """
    Transcribe an audio file using ElevenLabs Speech-to-Text (scribe_v1).
    Returns { text, details? }.
    """
    api_key = settings.elevenlabs_api_key
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not configured")

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "audio/mpeg" if path.suffix.lower() in {".mp3"} else "audio/wav"

    client = ElevenLabs(api_key=api_key)
    data = path.read_bytes()
    res = client.speech_to_text.convert(model_id=model, file=(path.name, data, mime))

    text: str = ""
    details: Dict[str, Any] = {}
    if hasattr(res, "text"):
        text = getattr(res, "text")
        details = {"language_code": getattr(res, "language_code", None)}
    elif hasattr(res, "transcripts"):
        try:
            transcripts = getattr(res, "transcripts")
            text = " ".join([getattr(t, "text", "") for t in transcripts])
        except Exception:
            text = ""
    elif hasattr(res, "message"):
        details = {
            "message": getattr(res, "message", ""),
            "transcription_id": getattr(res, "transcription_id", None),
        }

    return {"text": text, "details": details}
