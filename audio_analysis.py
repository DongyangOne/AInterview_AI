import json
import os
import re
import subprocess
import tempfile
import time

import google.generativeai as genai
from dotenv import load_dotenv

# ========= 환경 세팅 =========
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 모델/설정 환경변수 (없으면 기본값)
IMPL = os.getenv("WHISPER_IMPL", "faster")                 # "faster" or "whisper"
FASTER_MODEL = os.getenv("FASTER_WHISPER_MODEL", "tiny")   # tiny|base|small|medium...
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")         # 폴백 whisper
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")                # Pi는 cpu
COMPUTE_TYPE = os.getenv("FASTER_COMPUTE_TYPE", "int8")    # int8|int8_float16|auto
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))           # Pi5는 4 권장

# ========= Gemini (JSON 강제) =========
genai.configure(api_key=GOOGLE_API_KEY)
generation_config = {
    "temperature": 0.3,
    "max_output_tokens": 1024,
    "response_mime_type": "application/json",
}
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-001",
    generation_config=generation_config
)

TARGET_KEYS = ["feedback", "confidence", "tone", "resilience", "good", "bad"]

# ========= 공통 유틸 =========
def _coerce_int(v, default=0, lo=0, hi=100):
    try:
        x = int(round(float(v)))
        return max(lo, min(hi, x))
    except:
        return default

def _extract_json(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    i, j = s.find("{"), s.rfind("}")
    return s[i:j+1] if i != -1 and j != -1 and j > i else s

def _ensure_feedback_schema(d: dict) -> dict:
    """임의 스키마 → 우리가 원하는 6키로 정규화"""
    # overall류 → feedback
    feedback = (
        d.get("feedback") or
        d.get("overall_feedback") or
        d.get("overall") or
        d.get("summary") or
        ""
    )
    feedback = str(feedback).strip() or "핵심 사례를 먼저 제시하면 더 효과적입니다."

    # good/bad 후보
    good = (d.get("good") or "").strip()
    bad  = (d.get("bad")  or "").strip()
    pos = d.get("positive_aspects")
    if not good and isinstance(pos, dict):
        good = " ".join([str(v) for v in pos.values() if v]).strip()
    if not good and isinstance(pos, list):
        good = " ".join([str(x) for x in pos if x]).strip()
    neg = d.get("areas_for_improvement") or d.get("specific_feedback")
    if not bad and isinstance(neg, dict):
        bad = " ".join([str(v) for v in neg.values() if v]).strip()
    if not bad and isinstance(neg, list):
        bad = " ".join([str(x) for x in neg if x]).strip()
    if not good:
        good = "구체적인 사례를 잘 설명했습니다."
    if not bad:
        bad = "답변이 다소 장황했습니다."

    # 수치
    confidence = _coerce_int(d.get("confidence", d.get("overall_score", 65)), 65)
    tone = _coerce_int(d.get("tone", 70), 70)
    resilience = _coerce_int(d.get("resilience", 70), 70)

    out = {
        "feedback": feedback,
        "confidence": confidence,
        "tone": tone,
        "resilience": resilience,
        "good": good,
        "bad": bad,
    }
    return {k: out[k] for k in TARGET_KEYS}

def _extract_audio_to_wav16(video_path: str, strip_silence: bool = True) -> str:
    """
    ffmpeg로 16kHz mono wav 추출 (+옵션: 무음 구간 제거)
    strip_silence=True면 무성 구간을 제거하여 길이 축소 → 속도↑
    """
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"
        ]
        if strip_silence:
            cmd += ["-af", "silenceremove=start_periods=1:start_duration=0.5:start_threshold=-40dB:"
                            "stop_periods=1:stop_duration=0.5:stop_threshold=-40dB"]
        cmd += [tmp.name]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return tmp.name
    except Exception:
        return video_path

# ========= ASR: faster-whisper 우선, whisper 폴백 =========
_faster_model = None
_whisper_model = None
_faster_available = False

def _init_asr_models():
    global _faster_model, _whisper_model, _faster_available

    if IMPL.lower() == "faster":
        try:
            from faster_whisper import WhisperModel  # lazy import
            print(f"faster-whisper 모델 로딩 중... (model={FASTER_MODEL}, device=cpu, compute_type={COMPUTE_TYPE}, threads={CPU_THREADS})")
            _faster_model = WhisperModel(
                FASTER_MODEL, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=CPU_THREADS
            )
            _faster_available = True
            print("✅ faster-whisper 로딩 완료")
            return
        except Exception as e:
            print(f"⚠️ faster-whisper 사용 불가, whisper로 폴백합니다: {e}")

    # whisper 폴백
    try:
        import whisper
        print(f"Whisper 모델 로딩 중... (model={WHISPER_MODEL}, device=cpu)")
        _whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")
        print("✅ Whisper 로딩 완료")
    except Exception as e:
        print(f"❌ Whisper 로딩 실패: {e}")
        raise

_init_asr_models()

def _transcribe_faster_whisper(audio_path: str) -> str:
    """faster-whisper 고속 변환 (VAD on)"""
    assert _faster_model is not None
    start = time.time()
    print("🎧 faster-whisper 변환 시작...")

    segments, info = _faster_model.transcribe(
        audio_path,
        language="ko",
        beam_size=1,               # 속도 우선
        vad_filter=True,           # 무음 자동 스킵
        vad_parameters={"min_speech_duration_ms": 250}
    )
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text)
    out = " ".join(text_parts).strip()

    elapsed = round(time.time() - start, 2)
    print(f"✅ faster-whisper 변환 완료 ({elapsed}초)")
    return out

def _transcribe_whisper(audio_path: str) -> str:
    """원본 whisper 폴백 (CPU면 느릴 수 있음)"""
    import whisper
    start = time.time()
    print("🎧 Whisper 변환 시작...")
    result = _whisper_model.transcribe(
        audio_path,
        language="ko",
        fp16=False,
        beam_size=1,
        best_of=1,
        word_timestamps=False,
        condition_on_previous_text=False,
        no_speech_threshold=0.6
    )
    elapsed = round(time.time() - start, 2)
    print(f"✅ Whisper 변환 완료 ({elapsed}초)")
    return result.get("text", "").strip()

def transcribe_audio_with_asr(audio_or_video_path: str) -> str:
    """비디오 → 오디오 추출(무음 제거) → faster-whisper or whisper"""
    path = _extract_audio_to_wav16(audio_or_video_path, strip_silence=True)
    try:
        if _faster_available:
            return _transcribe_faster_whisper(path)
        else:
            return _transcribe_whisper(path)
    finally:
        if path != audio_or_video_path and os.path.exists(path):
            os.unlink(path)

# ========= Gemini 호출 =========
def generate_feedback_with_gemini(question, answer):
    print("🧠 Gemini 피드백 생성 중...")
    start = time.time()

    prompt = f"""
아래 면접 질문과 답변을 평가하세요.
반드시 JSON만 출력하세요. 
키는 다음 6개만 포함합니다: feedback, confidence, tone, resilience, good, bad.

면접 질문: "{question}"
면접자 답변: "{answer}"

출력 형식 예시:
{{
  "feedback": "핵심 사례를 먼저 제시하면 더 효과적입니다.",
  "confidence": 75,
  "tone": 68,
  "resilience": 80,
  "good": "구체적인 사례를 잘 설명했습니다.",
  "bad": "답변이 다소 장황했습니다."
}}

제약:
- 오직 위 6개 키만 포함.
- 코드 블록( ``` ) 출력 금지.
- JSON 외 텍스트 금지.
- feedback은 반드시 한 문장으로 요약된 전체 평가 문장.
"""

    raw = ""
    try:
        res = gemini_model.generate_content(prompt)
        raw = (res.text or "").strip()
        body = _extract_json(raw)
        parsed = json.loads(body)
    except Exception as e:
        print(f"⚠️ 1차 파싱 실패, 보정 시도: {e}")
        try:
            body = _extract_json(raw)
            parsed = json.loads(body)
        except Exception as e2:
            print(f"❌ 2차 파싱 실패: {e2}")
            elapsed = round(time.time() - start, 2)
            print(f"✅ Gemini 기본값 반환 ({elapsed}초)")
            return {
                "feedback": "핵심 사례를 먼저 제시하면 더 효과적입니다.",
                "confidence": 65,
                "tone": 70,
                "resilience": 70,
                "good": "구체적인 사례를 잘 설명했습니다.",
                "bad": "답변이 다소 장황했습니다."
            }

    final_dict = _ensure_feedback_schema(parsed)
    elapsed = round(time.time() - start, 2)
    print(f"✅ Gemini 피드백 완료 ({elapsed}초) -> {final_dict}")
    return final_dict

def run_interview_feedback_service(audio_or_video_path, interview_question):
    text = transcribe_audio_with_asr(audio_or_video_path)
    if not text:
        return {
            "feedback": "음성 변환 실패",
            "confidence": 0,
            "tone": 0,
            "resilience": 0,
            "good": "",
            "bad": ""
        }
    print(f"🗣️ 변환된 텍스트 일부: {text[:100]}...")
    return generate_feedback_with_gemini(interview_question, text)
