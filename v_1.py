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
EAR_THRESHOLD = 0.22     
CSV_FILE = "final_faint_system.csv"

# TIMING CONSTANTS (Seconds)
TIME_DROWSY = 1.5        # Trigger "Drowsy" after 1.5 sec
TIME_FAINT_EYES = 7.0    # Trigger "Faint" after 7.0 sec

# GEOFENCE LIMITS (Degrees)
LIMIT_PITCH_DOWN = 15.0  
LIMIT_PITCH_UP = 20.0    
LIMIT_ROLL = 25.0        

# MediaPipe Indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

model_points = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
])

# ===============================
# HELPER FUNCTIONS
# ===============================
def eye_aspect_ratio(landmarks, indices, width, height):
    coords = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in indices]
    p1, p2, p3, p4, p5, p6 = coords
    A = distance.euclidean(p2, p6)
    B = distance.euclidean(p3, p5)
    C = distance.euclidean(p1, p4)
    return (A + B) / (2.0 * C), coords

def get_head_pose(landmarks, img_w, img_h):
    face_2d = []
    for idx in [1, 152, 33, 263, 61, 291]:
        face_2d.append((int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)))
    
    face_2d = np.array(face_2d, dtype="double")
    camera_matrix = np.array([[img_w, 0, img_w/2], [0, img_w, img_h/2], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(model_points, face_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rot_vec)
    sy = math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
    
    if not (sy < 1e-6):
        x = math.atan2(rmat[2,1] , rmat[2,2])
        y = math.atan2(-rmat[2,0], sy)
        z = math.atan2(rmat[1,0], rmat[0,0])
    else:
        x = math.atan2(-rmat[1,2], rmat[1,1])
        y = math.atan2(-rmat[2,0], sy)
        z = 0

    return x * 180 / math.pi, y * 180 / math.pi, z * 180 / math.pi

# ===============================
# MAIN EXECUTION
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
cap = cv2.VideoCapture(0)
events = []

# State
drowsy_state = False
eye_closure_start = None  # Timestamp when eyes first closed
faint_posture_counter = 0

# Calibration
calib_state = 0 
calib_timer = 0
calib_pitch_vals = []
calib_roll_vals = []
zero_pitch = 0
zero_roll = 0

print("[INFO] Version 10.0: Fail-Safe System Started.")

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
            # STEP 1: CALIBRATE
            # ==========================================
            if calib_state == 0:
                calib_timer += 1
                cv2.putText(frame, "CALIBRATING: SIT NORMALLY", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                calib_pitch_vals.append(pitch)
                calib_roll_vals.append(roll)
                cv2.rectangle(frame, (50, 220), (int(50 + (calib_timer * 4)), 240), (0, 255, 255), -1)

                if calib_timer > 60: 
                    zero_pitch = np.mean(calib_pitch_vals)
                    zero_roll = np.mean(calib_roll_vals)
                    calib_state = 1
                    print(f"Zero Points Set - Pitch: {zero_pitch:.1f}, Roll: {zero_roll:.1f}")

            # ==========================================
            # STEP 2: MONITORING
            # ==========================================
            elif calib_state == 1:
                reason = ""
                is_alert = False
                color = (0, 255, 0)
                
                # --- A. CHECK EYES (TIMER LOGIC) ---
                if current_ear < EAR_THRESHOLD:
                    if eye_closure_start is None:
                        eye_closure_start = time.time() # Start timer
                    
                    closed_duration = time.time() - eye_closure_start
                    
                    # Logic: "How long have they been closed?"
                    if closed_duration > TIME_FAINT_EYES:
                        reason = f"FAINT (UNCONSCIOUS) - {closed_duration:.1f}s"
                        is_alert = True
                        color = (0, 0, 255) # RED
                    elif closed_duration > TIME_DROWSY:
                        reason = "DROWSY ALERT"
                        is_alert = True
                        color = (0, 165, 255) # ORANGE
                else:
                    eye_closure_start = None # Reset timer if eyes open
                
                # --- B. CHECK HEAD POSTURE (GEOFENCE) ---
                # Only check posture if we haven't already triggered the "Unconscious" alarm
                if "UNCONSCIOUS" not in reason:
                    # Calculate Deltas
                    d_pitch = pitch - zero_pitch
                    if d_pitch > 180: d_pitch -= 360
                    if d_pitch < -180: d_pitch += 360
                    
                    d_roll = roll - zero_roll
                    if d_roll > 180: d_roll -= 360
                    if d_roll < -180: d_roll += 360

                    # Check Limits
                    posture_faint = False
                    if d_pitch > LIMIT_PITCH_DOWN: posture_faint = True
                    elif d_pitch < -LIMIT_PITCH_UP: posture_faint = True
                    elif abs(d_roll) > LIMIT_ROLL: posture_faint = True
                    
                    if posture_faint:
                        faint_posture_counter += 1
                        if faint_posture_counter > 8: # Confirm posture slump
                            reason = "FAINT (POSTURE COLLAPSE)"
                            is_alert = True
                            color = (0, 0, 255)
                    else:
                        faint_posture_counter = 0

                # --- C. DISPLAY & LOG ---
                cv2.polylines(frame, [np.array(leftHull)], True, color, 1)
                cv2.polylines(frame, [np.array(rightHull)], True, color, 1)

                if is_alert:
                    cv2.putText(frame, f"ALERT: {reason}!", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                    
                    # Log only once per event state change
                    if not drowsy_state:
                        events.append([timestamp, f"{current_ear:.2f}", reason])
                        drowsy_state = True
                else:
                    drowsy_state = False
                    # Show Debug Info
                    cv2.putText(frame, f"EAR: {current_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    if eye_closure_start:
                        dur = time.time() - eye_closure_start
                        cv2.putText(frame, f"Closed: {dur:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            break
    else:
        cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Faint Detection V10.0 (Fail-Safe)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if events:
    df = pd.DataFrame(events, columns=["Timestamp", "EAR", "Event"])
    df.to_csv(CSV_FILE, index=False)