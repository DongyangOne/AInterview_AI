import cv2
import mediapipe as mp
import numpy as np
import os
import time
import logging

# 🔇 Mediapipe / TensorFlow 로그 억제
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# ---------------- Mediapipe 초기화 ----------------
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

# ---------------- 상태 추적 ----------------
prev_eye_center = None
prev_hand_center = None


# ---------------- Feature Extractors ----------------
def extract_pose(pose_landmarks):
    try:
        l = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        dx, dy = r.x - l.x, r.y - l.y
        angle = np.degrees(np.arctan2(dy, dx))
        upright_score = 100 - min(abs(angle), 30) * 2
        return max(0, int(upright_score))
    except:
        return 50


def extract_facial(landmarks):
    left, right, top, bottom = landmarks[61], landmarks[291], landmarks[13], landmarks[14]
    width = abs(right.x - left.x)
    height = abs(bottom.y - top.y)
    ratio = height / width if width > 0 else 0
    if 0.2 < ratio < 0.35: return 90
    elif 0.15 < ratio < 0.4: return 70
    else: return 40


def extract_understanding(landmarks, hand_landmarks):
    global prev_eye_center, prev_hand_center
    score = 100

    if landmarks:
        left_eye, right_eye = landmarks[33], landmarks[263]
        eye_center = np.mean([[left_eye.x, left_eye.y], [right_eye.x, right_eye.y]], axis=0)
        if prev_eye_center is not None:
            dx, dy = eye_center[0] - prev_eye_center[0], eye_center[1] - prev_eye_center[1]
            move = np.sqrt(dx**2 + dy**2)
            score -= min(move * 10000, 40)
        prev_eye_center = eye_center

    if hand_landmarks:
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        curr_center = (np.mean(xs), np.mean(ys))
        if prev_hand_center is not None:
            dx, dy = curr_center[0] - prev_hand_center[0], curr_center[1] - prev_hand_center[1]
            dist = np.sqrt(dx**2 + dy**2)
            score -= min(dist * 3000, 40)
        prev_hand_center = curr_center

    return max(0, int(score))


# ---------------- Main ----------------
def analyze_video(video_path):
    start = time.time()
    print("🎥 영상 분석 시작...")

    if not os.path.exists(video_path):
        return {"pose": 0, "facial": 0, "understanding": 0}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"pose": 0, "facial": 0, "understanding": 0}

    FRAME_SKIP = 5
    TARGET_WIDTH = 320
    f_count, analyzed = 0, 0
    pose_sum = facial_sum = understanding_sum = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f_count += 1
        if f_count % FRAME_SKIP != 0:
            continue

        # 해상도 축소
        h, w = frame.shape[:2]
        if w > TARGET_WIDTH:
            s = TARGET_WIDTH / w
            frame = cv2.resize(frame, (TARGET_WIDTH, int(h * s)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(rgb)
        pose_results = pose.process(rgb) if analyzed % 2 == 0 else None
        hand_results = hands.process(rgb) if analyzed % 3 == 0 else None

        pose_score = facial_score = understanding_score = 0

        if pose_results and pose_results.pose_landmarks:
            pose_score = extract_pose(pose_results.pose_landmarks)

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            facial_score = extract_facial(landmarks)
            if hand_results and hand_results.multi_hand_landmarks:
                understanding_score = extract_understanding(landmarks, hand_results.multi_hand_landmarks[0])
            else:
                understanding_score = extract_understanding(landmarks, None)
        else:
            understanding_score = 70

        pose_sum += pose_score
        facial_sum += facial_score
        understanding_sum += understanding_score
        analyzed += 1

    cap.release()

    elapsed = round(time.time() - start, 2)
    print(f"✅ 영상 분석 완료 ({elapsed}초, {analyzed}프레임 분석됨)")

    if analyzed == 0:
        return {"pose": 0, "facial": 0, "understanding": 0}

    return {
        "pose": pose_sum // analyzed,
        "facial": facial_sum // analyzed,
        "understanding": understanding_sum // analyzed,
    }
