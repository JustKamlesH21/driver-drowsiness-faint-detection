import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from scipy.spatial import distance
import math
import threading           
import geocoder            
from twilio.rest import Client  
import pygame              
import winsound
import os
from dotenv import load_dotenv            

# ===============================
# CONFIGURATION
# ===============================
CSV_FILE = "log.csv"

# --- 1. TWILIO SMS KEYS ---
load_dotenv()
TWILIO_SID = os.getenv("ACCOUNT_SID") 
TWILIO_AUTH = os.getenv("AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILO_PHONE")               
TO_PHONE = '+918089520113'                                    

# --- 2. DEMO LOCATION ---
USE_EXACT_LOCATION = True
DEMO_LAT = 12.644236275216052  
DEMO_LNG = 77.43739419349555

# --- 3. TIME THRESHOLDS ---
TIME_TIER_1_DROWSY = 1.5   
TIME_TIER_2_SLEEP = 5.5    
TIME_TIER_3_FAINT = 12.0   

# --- 4. GEOFENCE LIMITS ---
LIMIT_PITCH_DOWN = 15.0  
LIMIT_PITCH_UP = 20.0    
LIMIT_ROLL = 25.0        

# ===============================
# UTILS
# ===============================
pygame.mixer.init()
try:
    alert_sound = pygame.mixer.Sound("alarm.wav") 
except:
    alert_sound = None 

try:
    client = Client(TWILIO_SID, TWILIO_AUTH)
    sms_enabled = True
except:
    print("[WARNING] Twilio Keys Invalid. SMS disabled.")
    sms_enabled = False

# ===============================
# HELPER FUNCTIONS
# ===============================
def send_emergency_sms(reason):
    def _send():
        try:
            if USE_EXACT_LOCATION:
                lat, lng = DEMO_LAT, DEMO_LNG
                loc_type = "Fixed"
            else:
                g = geocoder.ip('me')
                lat, lng = g.latlng if g.latlng else (0, 0)
                loc_type = "IP"

            maps_link = f"http://googleusercontent.com/maps.google.com/?q={lat},{lng}"
            
            body_msg = (f"SOS: Driver Unconscious!\n"
                        f"Cause: {reason}\n"
                        f"Loc: {maps_link}")
            
            if sms_enabled:
                message = client.messages.create(body=body_msg, from_=TWILIO_PHONE, to=TO_PHONE)
                print(f"[IOT] SMS Sent! SID: {message.sid}")
            else:
                print(f"[IOT-SIM] SMS Triggered: {body_msg}")

        except Exception as e:
            print(f"[ERROR] SMS Failed: {e}")
            
    threading.Thread(target=_send).start()

def play_alarm():
    if alert_sound:
        alert_sound.play()
    else:
        winsound.Beep(3000, 1000) 

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
    
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])

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
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
cap = cv2.VideoCapture(0)
events = []

drowsy_state = False
eye_closure_start = None
faint_posture_counter = 0

sms_sent_flag = False      
last_alarm_time = 0        

# Calibration Variables
calib_state = 0 
calib_timer = 0
calib_pitch_vals = []
calib_roll_vals = []
calib_ear_vals = []   # NEW: To store open-eye values

# Calculated Thresholds (Defaults, updated after calibration)
ear_threshold_dynamic = 0.22 
zero_pitch = 0
zero_roll = 0

start_time = time.time()
WARMUP_SECONDS = 3.0 


while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    timestamp = time.strftime("%H:%M:%S")
    h, w, _ = frame.shape

    if time.time() - start_time < WARMUP_SECONDS:
        cv2.putText(frame, "SYSTEM STARTING...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Faint Detection V16.0", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            leftEAR, leftHull = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
            rightEAR, rightHull = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
            current_ear = (leftEAR + rightEAR) / 2.0
            
            pitch, yaw, roll = get_head_pose(landmarks, w, h)

            # ==========================================
            # STEP 1: CALIBRATE (AUTO-TUNE EAR)
            # ==========================================
            if calib_state == 0:
                calib_timer += 1
                cv2.putText(frame, "KEEP EYES OPEN & LOOK STRAIGHT", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Collect Data
                calib_pitch_vals.append(pitch)
                calib_roll_vals.append(roll)
                calib_ear_vals.append(current_ear)
                
                cv2.rectangle(frame, (50, 220), (int(50 + (calib_timer * 4)), 240), (0, 255, 255), -1)

                if calib_timer > 60: 
                    # 1. Set Head Zero
                    zero_pitch = np.mean(calib_pitch_vals)
                    zero_roll = np.mean(calib_roll_vals)
                    
                    # 2. Set EAR Threshold (The Fix!)
                    avg_open_ear = np.mean(calib_ear_vals)
                    # Set threshold to 80% of YOUR open eye value
                    # (e.g., if you are 0.25, threshold becomes 0.20)
                    ear_threshold_dynamic = avg_open_ear * 0.80
                    # Safety Clamp: Don't let it go excessively low or high
                    ear_threshold_dynamic = max(0.16, min(ear_threshold_dynamic, 0.28))
                    
                    calib_state = 1
                    print(f"[INFO] Calibrated. OpenEAR:{avg_open_ear:.2f} -> Thresh:{ear_threshold_dynamic:.2f}")

            # ==========================================
            # STEP 2: MONITORING
            # ==========================================
            elif calib_state == 1:
                reason = ""
                is_alert = False
                tier_level = 0 
                color = (0, 255, 0)
                
                # --- A. EYES CHECK ---
                if current_ear < ear_threshold_dynamic:
                    if eye_closure_start is None:
                        eye_closure_start = time.time()
                    
                    closed_duration = time.time() - eye_closure_start
                    
                    # 3-TIER LOGIC
                    if closed_duration > TIME_TIER_3_FAINT: # > 12.0s
                        reason = f"FAINT (UNCONSCIOUS) {closed_duration:.1f}s"
                        is_alert = True
                        tier_level = 3
                        color = (0, 0, 255) 
                    elif closed_duration > TIME_TIER_2_SLEEP: # > 5.5s
                        reason = "Partially FAINT (SLEEP)"
                        is_alert = True
                        tier_level = 2
                        color = (0, 0, 255) 
                    elif closed_duration > TIME_TIER_1_DROWSY: # > 1.5s
                        reason = "DROWSY WARNING"
                        is_alert = True
                        tier_level = 1
                        color = (0, 165, 255) 
                else:
                    eye_closure_start = None 
                
                # --- B. POSTURE CHECK ---
                if tier_level < 3:
                    d_pitch = pitch - zero_pitch
                    if d_pitch > 180: d_pitch -= 360
                    if d_pitch < -180: d_pitch += 360
                    d_roll = roll - zero_roll
                    if d_roll > 180: d_roll -= 360
                    if d_roll < -180: d_roll += 360

                    posture_faint = False
                    if d_pitch > LIMIT_PITCH_DOWN: posture_faint = True
                    elif d_pitch < -LIMIT_PITCH_UP: posture_faint = True
                    elif abs(d_roll) > LIMIT_ROLL: posture_faint = True
                    
                    if posture_faint:
                        faint_posture_counter += 1
                        if faint_posture_counter > 8:
                            reason = "FAINT (POSTURE COLLAPSE)"
                            is_alert = True
                            tier_level = 3  
                            color = (0, 0, 255)
                    else:
                        faint_posture_counter = 0

                # --- C. ACTIONS ---
                cv2.polylines(frame, [np.array(leftHull)], True, color, 1)
                cv2.polylines(frame, [np.array(rightHull)], True, color, 1)

                if is_alert:
                    cv2.putText(frame, f"{reason}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                    
                    # HARDWARE TRIGGER
                    if tier_level == 2:
                        if time.time() - last_alarm_time > 1.0: 
                            play_alarm()
                            last_alarm_time = time.time()

                    elif tier_level == 3:
                        if time.time() - last_alarm_time > 1.0: 
                            play_alarm()
                            last_alarm_time = time.time()
                        
                        if not sms_sent_flag:
                            print("[IOT] TIER 3 EMERGENCY! Sending SMS...")
                            send_emergency_sms(reason)
                            sms_sent_flag = True 

                    if not drowsy_state:
                        events.append([timestamp, f"{current_ear:.2f}", reason])
                        drowsy_state = True
                else:
                    drowsy_state = False
                    # Debug Text: Show Live EAR vs Threshold
                    # This helps you see WHY it thinks eyes are closed
                    status_txt = f"EAR: {current_ear:.2f} (Limit: {ear_threshold_dynamic:.2f})"
                    cv2.putText(frame, status_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    if eye_closure_start:
                        dur = time.time() - eye_closure_start
                        cv2.putText(frame, f"Eyes: {dur:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            break
    else:
        cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Faint Detection V16.0", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if events:
    df = pd.DataFrame(events, columns=["Timestamp", "EAR", "Event"])
    df.to_csv(CSV_FILE, index=False)