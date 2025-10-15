import whisper
import google.generativeai as genai
import json
import time
from dotenv import load_dotenv
import os
import re
import subprocess
import tempfile

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

print("Whisper 모델 로딩 중...")
whisper_model = whisper.load_model(WHISPER_MODEL)
print(f"✅ Whisper 모델 로딩 완료 (model={WHISPER_MODEL})")

# Gemini 설정 (JSON만 받도록)
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

def _coerce_int(v, default=0, lo=0, hi=100):
    try:
        x = int(round(float(v)))
        return max(lo, min(hi, x))
    except:
        return default

def _extract_json(text: str) -> str:
    """코드펜스/앞뒤 군더더기 제거 후 JSON 본문만 추출"""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    i, j = s.find("{"), s.rfind("}")
    return s[i:j+1] if i != -1 and j != -1 and j > i else s

def _ensure_feedback_schema(d: dict) -> dict:
    """Gemini가 반환한 구조를 우리가 원하는 6키 스키마로 정규화."""
    # 1️⃣ overall류 피드백
    feedback = (
        d.get("feedback") or
        d.get("overall_feedback") or
        d.get("overall") or
        d.get("summary") or
        ""
    )
    feedback = str(feedback).strip() or "핵심 사례를 먼저 제시하면 더 효과적입니다."

    # 2️⃣ 긍정적 피드백 후보
    good = d.get("good") or ""
    if not good:
        pos = d.get("positive_aspects")
        if isinstance(pos, dict):
            good = " ".join([str(v) for v in pos.values() if v])
        elif isinstance(pos, list):
            good = " ".join([str(x) for x in pos if x])
    good = str(good).strip() or "구체적인 사례를 잘 설명했습니다."

    # 3️⃣ 보완점 후보
    bad = d.get("bad") or ""
    if not bad:
        neg = d.get("areas_for_improvement") or d.get("specific_feedback")
        if isinstance(neg, dict):
            bad = " ".join([str(v) for v in neg.values() if v])
        elif isinstance(neg, list):
            bad = " ".join([str(x) for x in neg if x])
    bad = str(bad).strip() or "답변이 다소 장황했습니다."

    # 4️⃣ 수치 필드
    confidence = _coerce_int(d.get("confidence", d.get("overall_score", 65)), 65)
    tone = _coerce_int(d.get("tone", 70), 70)
    resilience = _coerce_int(d.get("resilience", 70), 70)

    result = {
        "feedback": feedback,
        "confidence": confidence,
        "tone": tone,
        "resilience": resilience,
        "good": good,
        "bad": bad
    }
    return {k: result[k] for k in TARGET_KEYS}

def _extract_audio_to_wav16(video_path: str) -> str:
    """ffmpeg로 16kHz mono wav 추출"""
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return tmp.name
    except Exception:
        return video_path

def transcribe_audio_with_whisper(audio_or_video_path: str):
    print("🎧 Whisper 변환 시작...")
    start = time.time()
    try:
        path = _extract_audio_to_wav16(audio_or_video_path)
        result = whisper_model.transcribe(path, language="ko", fp16=False)
        elapsed = round(time.time() - start, 2)
        print(f"✅ Whisper 변환 완료 ({elapsed}초)")
        if path != audio_or_video_path and os.path.exists(path):
            os.unlink(path)
        return result["text"]
    except Exception as e:
        print(f"❌ Whisper 오류: {e}")
        return None

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
    text = transcribe_audio_with_whisper(audio_or_video_path)
    if not text:
        return {"feedback": "음성 변환 실패", "confidence": 0, "tone": 0, "resilience": 0, "good": "", "bad": ""}
    print(f"🗣️ 변환된 텍스트 일부: {text[:100]}...")
    return generate_feedback_with_gemini(interview_question, text)
