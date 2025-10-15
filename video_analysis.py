import cv2
import mediapipe as mp
import numpy as np
import os
import logging

# 🔇 Mediapipe / TensorFlow 로그 억제
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# ---------------- 전역 Mediapipe 객체 ----------------
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- 상태 추적용 전역 변수 ----------------
prev_eye_center = None
prev_hand_center = None


# ---------------- Feature Extractors ----------------
def extract_pose(pose_landmarks):
    """상체 자세 평가 (어깨선 수평 기준)"""
    try:
        l_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        dx = r_shoulder.x - l_shoulder.x
        dy = r_shoulder.y - l_shoulder.y
        angle = np.degrees(np.arctan2(dy, dx))

        upright_score = 100 - min(abs(angle), 30) * 2  # 기울면 감점
        return max(0, int(upright_score))
    except:
        return 50


def extract_facial(landmarks):
    """표정 점수: 입꼬리 비율 기반 미소 감지"""
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    top_mouth = landmarks[13]
    bottom_mouth = landmarks[14]

    mouth_width = abs(right_mouth.x - left_mouth.x)
    mouth_height = abs(bottom_mouth.y - top_mouth.y)
    smile_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

    if 0.2 < smile_ratio < 0.35:
        return 90
    elif 0.15 < smile_ratio < 0.4:
        return 70
    else:
        return 40


def extract_understanding(landmarks, hand_landmarks):
    """침착함 점수: 눈동자 및 손 움직임 기반"""
    global prev_eye_center, prev_hand_center
    score = 100

    # 눈동자 움직임
    if landmarks:
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        eye_center = np.mean(
            [[left_eye.x, left_eye.y], [right_eye.x, right_eye.y]], axis=0
        )

        if prev_eye_center is not None:
            dx = eye_center[0] - prev_eye_center[0]
            dy = eye_center[1] - prev_eye_center[1]
            movement = np.sqrt(dx**2 + dy**2)
            score -= min(movement * 10000, 40)

        prev_eye_center = eye_center

    # 손동작
    if hand_landmarks:
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        curr_center = (np.mean(xs), np.mean(ys))

        if prev_hand_center is not None:
            dx = curr_center[0] - prev_hand_center[0]
            dy = curr_center[1] - prev_hand_center[1]
            distance = np.sqrt(dx**2 + dy**2)
            score -= min(distance * 3000, 40)

        prev_hand_center = curr_center

    return max(0, int(score))


# ---------------- Main Video Analyzer ----------------
def analyze_video(video_path):
    """영상 분석: pose / facial / understanding 점수 계산"""
    if not os.path.exists(video_path):
        return {"pose": 0, "facial": 0, "understanding": 0}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"pose": 0, "facial": 0, "understanding": 0}

    frame_count = 0
    analyzed_count = 0
    pose_total = facial_total = understanding_total = 0

    FRAME_SKIP = 5  # 5프레임마다 1번 분석
    TARGET_WIDTH = 320  # 해상도 축소용

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 샘플링
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue
        frame_count += 1

        # 해상도 축소
        h, w = frame.shape[:2]
        if w > TARGET_WIDTH:
            scale = TARGET_WIDTH / w
            frame = cv2.resize(frame, (TARGET_WIDTH, int(h * scale)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe 조건부 호출
        face_results = face_mesh.process(rgb)
        pose_results = pose.process(rgb) if analyzed_count % 2 == 0 else None
        hand_results = hands.process(rgb) if analyzed_count % 3 == 0 else None

        pose_score = facial_score = understanding_score = 0

        if pose_results and pose_results.pose_landmarks:
            pose_score = extract_pose(pose_results.pose_landmarks)

        if face_results.multi_face_landmarks:
            facial_score = extract_facial(face_results.multi_face_landmarks[0].landmark)

            if hand_results and hand_results.multi_hand_landmarks:
                understanding_score = extract_understanding(
                    face_results.multi_face_landmarks[0].landmark,
                    hand_results.multi_hand_landmarks[0],
                )
            else:
                understanding_score = extract_understanding(
                    face_results.multi_face_landmarks[0].landmark, None
                )
        else:
            understanding_score = 70  # 얼굴 인식 실패 시 기본값

        pose_total += pose_score
        facial_total += facial_score
        understanding_total += understanding_score
        analyzed_count += 1

    cap.release()

    if analyzed_count == 0:
        return {"pose": 0, "facial": 0, "understanding": 0}

    return {
        "pose": pose_total // analyzed_count,
        "facial": facial_total // analyzed_count,
        "understanding": understanding_total // analyzed_count,
    }


# ---------------- 테스트 실행 ----------------
if __name__ == "__main__":
    path = "test_video.mp4"
    result = analyze_video(path)
    print(result)
