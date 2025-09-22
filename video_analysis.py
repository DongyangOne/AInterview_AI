# video_analysis.py
import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh()
pose = mp_pose.Pose()
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

prev_eye_center = None
prev_hand_center = None

def extract_posture(pose_landmarks):
    """상체만 보일 때: 어깨선의 수평 정도로 자세 평가"""
    try:
        l_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        dx = r_shoulder.x - l_shoulder.x
        dy = r_shoulder.y - l_shoulder.y
        angle = np.degrees(np.arctan2(dy, dx))

        upright_score = 100 - min(abs(angle), 30) * 2  # 기울기 감점 완화
        return max(0, int(upright_score))
    except:
        return 50

def extract_expression(landmarks):
    """표정 점수: 입꼬리 비율로 미소 여부 판별"""
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

def extract_stability(landmarks, hand_landmarks):
    """침착함 점수: 눈동자 움직임 + 손동작 과다 여부"""
    global prev_eye_center, prev_hand_center
    score = 100

    # 눈동자 움직임
    if landmarks:
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        eye_center = np.mean([[left_eye.x, left_eye.y],
                              [right_eye.x, right_eye.y]], axis=0)

        if prev_eye_center is not None:
            dx = eye_center[0] - prev_eye_center[0]
            dy = eye_center[1] - prev_eye_center[1]
            movement = np.sqrt(dx**2 + dy**2)
            score -= min(movement * 1000, 20)

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
            score -= min(distance * 1000, 20)

        prev_hand_center = curr_center

    return max(0, int(score))

def analyze_video(video_path):
    """영상 하나를 분석해서 posture/expression/stability 점수 반환"""
    if not os.path.exists(video_path):
        return {"posture": 0, "expression": 0, "stability": 0}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"posture": 0, "expression": 0, "stability": 0}

    frame_count = 0
    posture_total = expression_total = stability_total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        pose_results = pose.process(rgb)
        hand_results = hands.process(rgb)

        posture_score = expression_score = stability_score = 0

        if pose_results.pose_landmarks:
            posture_score = extract_posture(pose_results.pose_landmarks)

        if face_results.multi_face_landmarks:
            expression_score = extract_expression(face_results.multi_face_landmarks[0].landmark)

            if hand_results.multi_hand_landmarks:
                stability_score = extract_stability(face_results.multi_face_landmarks[0].landmark,
                                                    hand_results.multi_hand_landmarks[0])
            else:
                stability_score = extract_stability(face_results.multi_face_landmarks[0].landmark, None)
        else:
            stability_score = 70

        posture_total += posture_score
        expression_total += expression_score
        stability_total += stability_score
        frame_count += 1

    cap.release()

    if frame_count == 0:
        return {"posture": 0, "expression": 0, "stability": 0}

    return {
        "posture": posture_total // frame_count,
        "expression": expression_total // frame_count,
        "stability": stability_total // frame_count
    }
