# Driver Drowsiness & Faint Detection System

A Windows-based real-time webcam monitoring system that detects driver drowsiness and fainting using MediaPipe Face Mesh, eye aspect ratio (EAR), head pose, and audio/SMS alerting.

## 🚀 Project Overview

This repository includes a working prototype for detecting unsafe driver states such as:
- `DROWSY WARNING`
- `SLEEP` / partial fainting
- `FAINT (UNCONSCIOUS)`

The system uses:
- `MediaPipe` face mesh for landmark detection
- `OpenCV` for video capture and overlay rendering
- `NumPy` / `SciPy` for numeric and geometric calculations
- `pygame` or `winsound` for local audible alarms
- `Twilio` for optional SMS emergency notifications
- `geocoder` for optional location lookup
- `.env` configuration for safely storing API credentials

## 📁 Key Files

- `main.py` - primary application with dynamic calibration, real-time drowsiness/faint detection, audio alarm, and optional SMS alerting.
- `drowsiness_system.py` - earlier version / alternative faint detection script with head pose calibration and event logging.
- `log.csv` - runtime output log file created by `main.py`.
- `faint_drowsiness_log.csv` - output produced by `drowsiness_system.py`.
- `ear_env/` - local Python virtual environment included in this workspace.
- `face_landmarker.task` - MediaPipe task file (project asset).
- `run.txt` - activation instructions and notes.

## ✅ Requirements

This project has been tested on Windows with Python `3.10`.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

### Included requirements

- `mediapipe==0.10.9`
- `opencv-python==4.13.0.90`
- `numpy==2.2.6`
- `pandas==2.3.3`
- `scipy==1.15.3`
- `geocoder==1.38.1`
- `twilio==9.10.0`
- `pygame==2.6.1`
- `python-dotenv==1.2.2`

## ⚙️ Setup Instructions

1. Open a terminal in the project root folder.
2. Activate the Python virtual environment if you want to use the provided one:

```powershell
.\ear_env\Scripts\activate
```

3. Install required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Create a `.env` file in the repository root with the following keys:

```dotenv
ACCOUNT_SID=your_twilio_account_sid
AUTH_TOKEN=your_twilio_auth_token
TWILO_PHONE=+1234567890
```

> If Twilio credentials are missing or invalid, SMS alerting will be disabled automatically and the system will continue using local audio alerts.

5. Ensure `alarm.wav` exists in the project root for the `pygame` sound alarm. If not, `winsound.Beep` will be used as fallback.

## ▶️ Running the System

Launch the main detection system with:

```powershell
python main.py
```

- Press `q` inside the OpenCV window to quit.
- The system uses camera index `0` by default.

## 🧠 How It Works

### `main.py`

- Detects face landmarks using MediaPipe Face Mesh.
- Uses eye aspect ratio (EAR) to measure blink/eye closure.
- Computes head pose to detect pitch/roll and faint-like postures.
- Calibrates for each user during startup to set safe head and EAR thresholds.
- Sends emergency SMS via Twilio for severe events.
- Logs events to `log.csv`.

### `drowsiness_system.py`

- A simpler fallback detection script.
- Uses calibration steps to set straight and downward head pose.
- Emits alerts when eyes close or head drops past a trigger.
- Saves results into `faint_drowsiness_log.csv`.

## 🎯 Detection Thresholds

The main system uses the following built-in threshold values in `main.py`:

- `TIME_TIER_1_DROWSY = 1.5` seconds
- `TIME_TIER_2_SLEEP = 5.5` seconds
- `TIME_TIER_3_FAINT = 12.0` seconds
- `LIMIT_PITCH_DOWN = 15.0` degrees
- `LIMIT_PITCH_UP = 20.0` degrees
- `LIMIT_ROLL = 25.0` degrees
- `EAR_THRESHOLD_DYNAMIC` is calibrated automatically during startup with a fallback default of `0.22`

## 📝 Customization

You can fine-tune detection behavior in `main.py` by editing constants:

- `TIME_TIER_1_DROWSY`
- `TIME_TIER_2_SLEEP`
- `TIME_TIER_3_FAINT`
- `LIMIT_PITCH_DOWN`
- `LIMIT_PITCH_UP`
- `LIMIT_ROLL`
- `USE_EXACT_LOCATION`
- `DEMO_LAT` / `DEMO_LNG`

## 📌 Notes

- `winsound` is used as a fallback alarm on Windows when `alarm.wav` is missing.
- If no face is found, the system displays `NO FACE DETECTED`.
- The first few seconds are reserved for warm-up and calibration.

## 💡 Recommended Improvements

If you want to extend the project, consider adding:
- a GUI for configuration and status display
- support for multiple camera devices
- safety zone overlay and driver ID authentication
- cloud-based alert logging or remote dashboard

## 📞 Troubleshooting

- If the webcam does not initialize, verify camera permissions and that no other app is using the camera.
- If `mediapipe` fails to import, reinstall with `pip install mediapipe==0.10.9`.
- If SMS is not sent, confirm Twilio credentials and check the `TO_PHONE` destination number.

---

Thank you for using this drowsiness/faint detection system. Stay safe and feel free to extend it further with custom alerting, logging, or driver-monitoring features.
