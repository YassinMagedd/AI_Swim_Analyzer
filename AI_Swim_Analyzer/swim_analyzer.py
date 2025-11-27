import cv2
import mediapipe as mp
import numpy as np


# ----------------------------------------------------
# My Mathematical Angle calculating equations
# ----------------------------------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


current_elbow_angle = 0
current_knee_angle = 0
rep_counter = 0
stage = "START"
knee_status = "N/A"
last_valid_landmarks = None

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    model_complexity=1,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    smooth_landmarks=True
)

video_path = 'swimmer_video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error, Video not found.")
    print(f"Please make sure that video is found in: {video_path}")
    exit()

success, image = cap.read()
if success:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

is_running = True

cv2.namedWindow('AI Swim Analyzer', cv2.WINDOW_NORMAL)
while cap.isOpened():

    # ----------------------------------------------------
    # Video Control
    # ----------------------------------------------------
    if is_running:
        success, image_read = cap.read()

        if not success:
            print("Video finished.")
            break

        image = image_read.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:

            last_valid_landmarks = results.pose_landmarks
            landmarks = results.pose_landmarks.landmark

            # ----------------------------------------------------
            # Angles Calculations
            # ----------------------------------------------------

            # Elbow
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            # Knee
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
            knee_angle = calculate_angle(hip_r, knee_r, ankle_r)

            current_elbow_angle = elbow_angle
            current_knee_angle = knee_angle

            # ----------------------------------------------------
            # Repetition Counter
            # ----------------------------------------------------
            if elbow_angle < 100:
                stage = "PULLING"

            if elbow_angle > 150 and stage == "PULLING":
                stage = "RECOVERY"
                rep_counter += 1

            # ----------------------------------------------------
            # Knee Efficiency Analysis
            # ----------------------------------------------------
            if current_knee_angle > 170:
                knee_status = "Too Straight (Check Technique)"
            elif current_knee_angle > 150:
                knee_status = "Optimal Kick Angle"
            elif current_knee_angle < 150 and current_knee_angle > 130:
                knee_status = "Acceptable"
            elif current_knee_angle < 130:
                knee_status = "Bending Too Much (Drag)"
            else:
                knee_status = "N/A"

    if last_valid_landmarks:

        mp_drawing.draw_landmarks(
            image,
            last_valid_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    # ----------------------------------------------------
    # UI Counter
    # ----------------------------------------------------

    # DPS
    cv2.putText(image, 'REPS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, str(rep_counter), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    # Elbow Angle
    cv2.putText(image, f'ELBOW: {int(current_elbow_angle)} deg', (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Knee Angle
    cv2.putText(image, f'KNEE: {int(current_knee_angle)} deg', (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Movement Stage
    cv2.putText(image, f'STAGE: {stage}', (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # Knee Efficiency Analysis
    cv2.putText(image, f'KICK STATUS: {knee_status}', (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                cv2.LINE_AA)

    cv2.imshow('AI Swim Analyzer', image)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord(' '):
        is_running = not is_running

cap.release()
cv2.destroyAllWindows()