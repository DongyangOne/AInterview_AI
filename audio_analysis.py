import whisper
import google.generativeai as genai
import json
import time
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Whisper & Gemini 초기화
print("Whisper 모델 로딩 중...")
whisper_model = whisper.load_model("small")
print("✅ Whisper 모델 로딩 완료")

genai.configure(api_key=GOOGLE_API_KEY)
generation_config = {"temperature": 0.7, "max_output_tokens": 1024}
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-001",
    generation_config=generation_config
)


def transcribe_audio_with_whisper(audio_path: str):
    print("🎧 Whisper 변환 시작...")
    start = time.time()
    try:
        result = whisper_model.transcribe(audio_path, language="ko", fp16=False)
        elapsed = round(time.time() - start, 2)
        print(f"✅ Whisper 변환 완료 ({elapsed}초)")
        return result["text"]
    except Exception as e:
        print(f"❌ Whisper 오류: {e}")
        return None


def generate_feedback_with_gemini(question, answer):
    print("🧠 Gemini 피드백 생성 중...")
    start = time.time()

    prompt = f"""
    면접 질문: "{question}"
    면접자 답변: "{answer}"
    ---
    JSON 형식으로 피드백을 작성하세요.
    {{
      "feedback": "핵심 사례를 먼저 제시하면 더 효과적입니다.",
      "confidence": 75,
      "tone": 68,
      "resilience": 80,
      "good": "구체적인 사례를 잘 설명했습니다.",
      "bad": "답변이 다소 장황했습니다."
    }}
    """

    try:
        res = gemini_model.generate_content(prompt)
        txt = res.text.strip()
        if txt.startswith("```"):
            txt = txt.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(txt)
        elapsed = round(time.time() - start, 2)
        print(f"✅ Gemini 피드백 완료 ({elapsed}초)")
        print(parsed)
        return parsed
    except Exception as e:
        print(f"❌ Gemini 오류: {e}")
        return {"feedback": "피드백 생성 실패", "confidence": 0, "tone": 0, "resilience": 0}


def run_interview_feedback_service(audio_path, interview_question):
    text = transcribe_audio_with_whisper(audio_path)
    if not text:
        return {"feedback": "음성 변환 실패", "confidence": 0, "tone": 0, "resilience": 0}
    print(f"🗣️ 변환된 텍스트 일부: {text[:100]}...")
    return generate_feedback_with_gemini(interview_question, text)
