from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import tempfile, os

# 🔹 두 분석 모듈 불러오기
from video_analysis import analyze_video
from audio_analysis import run_interview_feedback_service  

app = FastAPI()

@app.post("/analyze")
async def analyze(
    feedbackId: str = Form(...),
    video: UploadFile = File(...)   # ✅ 단일 파일만 받음
):
    interview_question = "인생을 살며 푹 빠진것이 있나요? 있다면 설명 부탁드립니다."

    # 임시 파일 저장
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(await video.read())
    tmp.close()

    try:
        # 🎯 1) 영상 기반 분석
        v_result = analyze_video(tmp.name)  

        # 🎯 2) Whisper + Gemini 분석
        w_result = run_interview_feedback_service(tmp.name, interview_question)

        # 🎯 최종 결과 (단일 파일이라 평균 계산 필요 없음)
        final_result = {
            "feedbackId": feedbackId,
            "good": w_result.get("good", "좋은 점 피드백 없음"),
            "bad": w_result.get("bad", "아쉬운 점 피드백 없음"),
            "content": w_result.get("feedback", ""),
            "pose": v_result.get("pose", 0),
            "facial": v_result.get("facial", 0),
            "understanding": v_result.get("understanding", 0),
            "confidence": w_result.get("confidence", 0),
            "tone": w_result.get("tone", 0),
            "risk_response": w_result.get("resilience", 0),
        }

        return final_result

    finally:
        os.unlink(tmp.name)  # 임시 파일 삭제


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
