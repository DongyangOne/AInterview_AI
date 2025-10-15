from fastapi import FastAPI, UploadFile, File, Form
import tempfile, os, time
import uvicorn

from video_analysis import analyze_video
from audio_analysis import run_interview_feedback_service

app = FastAPI()

@app.post("/analyze")
async def analyze(
    feedbackId: str = Form(...),
    videos: UploadFile = File(...)
):
    start = time.time()
    interview_question = "프로젝트 진행 중 프론트엔드 업무를 맡으며 어려움을 극복한 경험을 말해주세요."

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(await videos.read())
    tmp.close()

    try:
        print("\n====================")
        print(f"📂 파일 수신 완료: {videos.filename}")
        print("====================")

        # 1️⃣ 영상 분석
        v_result = analyze_video(tmp.name)

        # 2️⃣ Whisper + Gemini 분석
        w_result = run_interview_feedback_service(tmp.name, interview_question)

        result = {
            "feedbackId": feedbackId,
            "good": w_result.get("good", "좋은 점 없음"),
            "bad": w_result.get("bad", "아쉬운 점 없음"),
            "content": w_result.get("feedback", ""),
            "pose": v_result.get("pose", 0),
            "facial": v_result.get("facial", 0),
            "understanding": v_result.get("understanding", 0),
            "confidence": w_result.get("confidence", 0),
            "tone": w_result.get("tone", 0),
            "risk_response": w_result.get("resilience", 0),
        }

        print(f"✅ 전체 분석 완료 (총 {round(time.time() - start, 2)}초 소요)")
        print("====================\n")
        return result

    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
