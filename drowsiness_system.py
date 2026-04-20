import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from scipy.spatial import distance
import math

# ===============================
# CONFIGURATION
# ===============================
CONSEC_FRAMES = 10       # Trigger faster (10 frames) for fainting
CSV_FILE = "faint_drowsiness_log.csv"
EAR_THRESHOLD = 0.22     # Eye floor

# MediaPipe Indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# 3D Model Points
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left Eye Left Corner
    (225.0, 170.0, -135.0),      # Right Eye Right Corner
    (-150.0, -150.0, -125.0),    # Left Mouth Corner
    (150.0, -150.0, -125.0)      # Right Mouth Corner
])

# ===============================
# HELPER FUNCTIONS
# ===============================
def normalize_angle(angle):
    if angle > 180: angle -= 360
    return angle

def eye_aspect_ratio(landmarks, indices, width, height):
    coords = []
    for i in indices:
        pt = landmarks[i]
        coords.append((int(pt.x * width), int(pt.y * height)))
    p1, p2, p3, p4, p5, p6 = coords
    A = distance.euclidean(p2, p6)
    B = distance.euclidean(p3, p5)
    C = distance.euclidean(p1, p4)
    return (A + B) / (2.0 * C), coords

def get_head_pose(landmarks, img_w, img_h):
    face_2d = []
    for idx in [1, 152, 33, 263, 61, 291]:
        pt = landmarks[idx]
        face_2d.append((int(pt.x * img_w), int(pt.y * img_h)))
    
    face_2d = np.array(face_2d, dtype="double")
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(model_points, face_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rot_vec)
    sy = math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rmat[2,1] , rmat[2,2])
        y = math.atan2(-rmat[2,0], sy)
        z = math.atan2(rmat[1,0], rmat[0,0])
    else:
        x = math.atan2(-rmat[1,2], rmat[1,1])
        y = math.atan2(-rmat[2,0], sy)
        z = 0

    return normalize_angle(x * 180 / math.pi), normalize_angle(y * 180 / math.pi), normalize_angle(z * 180 / math.pi)

# ===============================
# MAIN EXECUTION
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
cap = cv2.VideoCapture(0)
events = []

# State
counter_eyes = 0
counter_head = 0
drowsy_state = False

# Calibration State Machine
# 0 = Not Started, 1 = Calibrating Straight, 2 = Calibrating Down, 3 = Done
calib_state = 0 
calib_timer = 0
calib_straight_vals = []
calib_down_vals = []

# Final Calculated Thresholds
zero_pitch = 0
nod_trigger_pitch = 0  # The angle that triggers the alarm

print("[INFO] Version 6.0: Faint Detection Started.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    timestamp = time.strftime("%H:%M:%S")
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            leftEAR, leftHull = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
            rightEAR, rightHull = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
            current_ear = (leftEAR + rightEAR) / 2.0
            pitch, yaw, roll = get_head_pose(landmarks, w, h)

            # ==========================================
            # STEP 1: CALIBRATE "LOOK STRAIGHT"
            # ==========================================
            if calib_state == 0:
                calib_timer += 1
                cv2.putText(frame, "STEP 1: LOOK STRAIGHT AHEAD", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                calib_straight_vals.append(pitch)
                if calib_timer > 60: # Approx 2 seconds
                    zero_pitch = np.mean(calib_straight_vals)
                    calib_state = 1
                    calib_timer = 0
                    print(f"Zero Pitch set to: {zero_pitch:.1f}")

            # ==========================================
            # STEP 2: CALIBRATE "LOOK DOWN" (FAINT POSE)
            # ==========================================
            elif calib_state == 1:
                calib_timer += 1
                cv2.putText(frame, "STEP 2: LOOK DOWN (CHIN TO CHEST)", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                calib_down_vals.append(pitch)
                
                # Visual guide for user
                current_drop = pitch - zero_pitch
                cv2.putText(frame, f"Drop Angle: {current_drop:.1f}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                if calib_timer > 90: # Approx 3 seconds
                    avg_down = np.mean(calib_down_vals)
                    
                    # Calculate the TRIGGER POINT
                    # We set the trigger at 60% of the way down.
                    # Example: Straight is 0, Down is 40. Trigger is 24.
                    total_range = avg_down - zero_pitch
                    nod_trigger_pitch = zero_pitch + (total_range * 0.60)
                    
                    calib_state = 3 # Done
                    print(f"Down Pitch: {avg_down:.1f}, Trigger set at: {nod_trigger_pitch:.1f}")

            # ==========================================
            # STEP 3: MONITORING (FAINT DETECTION)
            # ==========================================
            elif calib_state == 3:
                reason = ""
                is_drowsy = False
                
                # Check 1: EYES
                if current_ear < EAR_THRESHOLD:
                    counter_eyes += 1
                    if counter_eyes >= CONSEC_FRAMES:
                        is_drowsy = True
                        reason = "EYES CLOSED"
                else:
                    counter_eyes = 0

                # Check 2: FAINT / HEAD NOD
                # We simply check if current pitch passed the calculated trigger line
                # We check ABS value to handle if 'down' is positive or negative
                dist_from_zero = pitch - zero_pitch
                dist_to_trigger = nod_trigger_pitch - zero_pitch
                
                # If we moved in the direction of 'down' past the trigger point
                # AND we aren't just tilting head sideways (Roll check)
                if (abs(dist_from_zero) > abs(dist_to_trigger)) and (np.sign(dist_from_zero) == np.sign(dist_to_trigger)) and (abs(roll) < 35):
                    counter_head += 1
                    if counter_head >= 5: # Fast trigger for fainting
                        is_drowsy = True
                        reason = "FAINT DETECTED"
                else:
                    counter_head = 0

                # Visuals
                color = (0, 0, 255) if is_drowsy else (0, 255, 0)
                cv2.polylines(frame, [np.array(leftHull)], True, color, 1)
                cv2.polylines(frame, [np.array(rightHull)], True, color, 1)
                
                status_text = f"PITCH: {pitch:.1f} (Tripwire: {nod_trigger_pitch:.1f})"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if is_drowsy:
                    cv2.putText(frame, f"ALERT: {reason}!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if not drowsy_state:
                        events.append([timestamp, f"{current_ear:.2f}", reason])
                        drowsy_state = True
                else:
                    drowsy_state = False

            break
    else:
        cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Faint Detection System V6.0", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if events:
    df = pd.DataFrame(events, columns=["Timestamp", "EAR", "Event"])
    df.to_csv(CSV_FILE, index=False)