from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import uvicorn
import tempfile, os

# 🔹 두 분석 모듈 불러오기
from video_analysis import analyze_video
from audio_analysis import run_interview_feedback_service  

app = FastAPI()

@app.post("/analyze")
async def analyze(
    feedbackId: str = Form(...),
    videos: UploadFile = File(...)
):
    video_results = []
    whisper_results = []
    interview_question = "인생을 살며 푹 빠진것이 있나요? 있다면 설명 부탁드립니다."

    for video in videos:
        # 임시 파일 저장
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(await video.read())
        tmp.close()

        # 🎯 1) 영상 기반 분석
        v_result = analyze_video(tmp.name)  # {"pose": 85, "facial": 90, "understanding": 75}
        video_results.append(v_result)

        # 🎯 2) Whisper + Gemini 분석
        w_result = run_interview_feedback_service(tmp.name, interview_question)
        whisper_results.append(w_result)

        os.unlink(tmp.name)  # 임시 파일 삭제

    # 🎯 여러 Whisper 결과에서 feedback/good/bad 정리
    feedback_texts = [r.get("feedback", "") for r in whisper_results if "feedback" in r]
    good_texts = [r.get("good", "") for r in whisper_results if "good" in r]
    bad_texts = [r.get("bad", "") for r in whisper_results if "bad" in r]

    # 평균값 계산 (여러 개 비디오 업로드 시)
    final_result = {
        "feedbackId": feedbackId,
        "good": " / ".join(good_texts) if good_texts else "좋은 점 피드백 없음",
        "bad": " / ".join(bad_texts) if bad_texts else "아쉬운 점 피드백 없음",
        "content":  " / ".join(feedback_texts) if feedback_texts else "",
        "pose": sum(r["pose"] for r in video_results) ,
        "facial": sum(r["facial"] for r in video_results) ,
        "understanding": sum(r["understanding"] for r in video_results) ,
        "confidence": sum(r["confidence"] for r in whisper_results) ,
        "tone": sum(r["tone"] for r in whisper_results),
        "risk_response": sum(r["resilience"] for r in whisper_results) ,
    }

    return final_result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
