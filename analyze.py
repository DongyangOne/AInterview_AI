from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import uvicorn
import tempfile, os
from video_analysis import analyze_video

app = FastAPI()

@app.post("/analyze")
async def analyze(
    feedbackId: str = Form(...),
    videos: List[UploadFile] = File(...)
):
    results = []

    for video in videos:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(await video.read())
        tmp.close()

        scores = analyze_video(tmp.name)
        results.append(scores)

        os.unlink(tmp.name)

    pose = sum(r["pose"] for r in results) // len(results)
    facial = sum(r["facial"] for r in results) // len(results)
    understanding = sum(r["understanding"] for r in results) // len(results)

    return {
        "feedbackId": feedbackId,
        "pose": pose,
        "facial": facial,
        "understanding": understanding
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
