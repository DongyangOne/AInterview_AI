# analyze.py
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import uvicorn
import tempfile, os
from video_analysis import analyze_video  # 위 파일에서 불러옴

app = FastAPI()

@app.post("/analyze")
async def analyze(
    feedbackId: str = Form(...),
    videos: List[UploadFile] = File(...)
):
    results = []

    for video in videos:
        # 임시 파일 저장
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(await video.read())
        tmp.close()

        # 분석
        scores = analyze_video(tmp.name)
        results.append(scores)

        os.unlink(tmp.name)  # 임시 파일 삭제

    # 평균 산출 (3개 영상 통합)
    pose = sum(r["pose"] for r in results) // len(results)
    facial = sum(r["facial"] for r in results) // len(results)
    understanding = sum(r["understanding"] for r in results) // len(results)

    return {
        "feedbackId": feedbackId,
        "pose": pose,  #자세
        "facial": facial, #표정
        "understanding": understanding #침착함
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
