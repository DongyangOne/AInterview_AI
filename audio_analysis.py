import whisper
import google.generativeai as genai
import json
from dotenv import load_dotenv
import os

# 🔹 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 🔹 Whisper 모델 로드
whisper_model = whisper.load_model("small")

# 🔹 Gemini API 설정
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 1024,
}

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)


def transcribe_audio_with_whisper(audio_file_path: str):
    """영상/음성 파일을 Whisper로 텍스트 변환"""
    print(f"음성 파일 변환 시작: {audio_file_path}")
    try:
        result = whisper_model.transcribe(audio_file_path, language="ko", fp16=False)
        transcribed_text = result["text"]
        print("음성 변환 완료.")
        return transcribed_text
    except Exception as e:
        print(f"Whisper 음성 변환 중 오류 발생: {e}")
        return None


def generate_feedback_with_gemini(질문, 답변):
    prompt = f"""
    당신은 면접관이자 면접 코치입니다. 면접 질문과 면접자의 답변 텍스트를 보고 건설적인 피드백을 1문장으로 요약해서 한국어로 제공해주세요.
    피드백은 면접자가 자신의 답변을 객관적으로 평가하고 다음 면접에서 더 나은 모습을 보일 수 있도록 매우 구체적이고 실용적인 지침을 제공하는 데 중점을 둡니다.
    
    아래 면접 질문과 답변을 보고 반드시 JSON 형식으로만 답변하세요.
    다른 설명이나 텍스트 없이 JSON만 출력하세요.
    ---
    면접 질문: "{질문}"
    
    면접자 답변 텍스트: "{답변}"
    ---
    
    피드백 항목:
    내용 및 구조, 발화 및 표현 (어휘/문장), 불필요한 반복 단어/어구, 부적절하거나 모호한 어휘, 더 전문적이거나 명확한 단어로 대체할 수 있는 부분 제시, 답변 길이 및 간결성:

    출력 예시:
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
        response = gemini_model.generate_content(prompt)
        raw_text = response.text.strip()

        # 🚨 코드 블럭 제거 (```json ... ```)
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("```")
            raw_text = raw_text.replace("json", "").strip()

        # JSON 파싱 시도
        try:
            parsed = json.loads(raw_text)

            # ✅ good, bad가 없으면 기본값 채우기
            if "good" not in parsed:
                parsed["good"] = "답변에서 긍정적인 부분이 잘 드러났습니다."
            if "bad" not in parsed:
                parsed["bad"] = "보완이 필요한 부분이 존재합니다."

            return parsed

        except json.JSONDecodeError:
            print("⚠️ Gemini 응답이 JSON 형식이 아님, 원문 반환:", raw_text)
            return {
                "feedback": raw_text,
                "confidence": 0,
                "tone": 0,
                "resilience": 0,
                "good": "긍정적인 피드백을 제공하지 못했습니다.",
                "bad": "아쉬운 점 피드백을 제공하지 못했습니다."
            }

    except Exception as e:
        print(f"Gemini API 호출 중 오류 발생: {e}")
        return {
            "feedback": "피드백 생성에 실패했습니다.",
            "confidence": 0,
            "tone": 0,
            "resilience": 0,
            "good": "긍정적인 피드백 생성 실패",
            "bad": "아쉬운 점 피드백 생성 실패"
        }




def run_interview_feedback_service(audio_file_path: str, interview_question: str):
    """Whisper + Gemini 종합 실행 함수"""
    print(f"면접 질문: {interview_question}")
    
    transcribed_answer = transcribe_audio_with_whisper(audio_file_path)
    if not transcribed_answer:
        return {"feedback": "음성 변환 실패", "confidence": 0, "tone": 0, "resilience": 0}

    print(f"면접자 답변: {transcribed_answer}")

    feedback = generate_feedback_with_gemini(interview_question, transcribed_answer)

    print(f"\n--- 면접 피드백 ---")
    print(feedback)
    
    return feedback
