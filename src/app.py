from __future__ import annotations

from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
import base64
from datetime import datetime, timezone
import json
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

from liveness import run_blink_liveness

# ---------------- APP CONFIG ----------------
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# Academic-project configuration. Replace with an environment variable in a
# production deployment.
app.secret_key = os.environ.get("BIOMETRIC_APP_SECRET", "secure_biometric_voting_key")

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_model.yml")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD CASCADE ----------------
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade: {CASCADE_PATH}")

# ---------------- MEDIAPIPE FACE MESH ----------------
# MediaPipe 0.10.21 is intentionally used for this project because it keeps
# the stable Face Mesh solution API used by this implementation.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.50,
    min_tracking_confidence=0.50,
)

# ---------------- CAMERA / MODEL STATE ----------------
camera = None
recognizer = None
camera_lock = threading.Lock()
model_lock = threading.Lock()

# ---------------- SYSTEM STATE ----------------
system_state = "IDLE"
verification_in_progress = False

# The /video endpoint and verification endpoint both access the same
# VideoCapture object. Serializing reads prevents concurrent camera.read()
# calls from corrupting the stream on some webcams/Windows drivers.
overlay_lock = threading.Lock()
overlay = {
    "text": "System ready",
    "state": "idle",
}

# --------- FACE MATCH SETTINGS ---------
THRESHOLD = 55
NUM_READINGS = 5

# Recent liveness verification can be used immediately by registration.
LIVENESS_VALID_SECONDS = 60

# ---------------- LOGIN CONFIG ----------------
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


def is_logged_in():
    return "user" in session


def set_overlay(text: str, state: str = "checking") -> None:
    """Update the text drawn over the live camera feed."""
    with overlay_lock:
        overlay["text"] = text
        overlay["state"] = state


def get_overlay():
    with overlay_lock:
        return dict(overlay)


def read_camera_frame():
    """Safely read one frame from the shared camera."""
    global camera

    with camera_lock:
        if camera is None or not camera.isOpened():
            return False, None
        return camera.read()


def release_camera():
    global camera

    with camera_lock:
        if camera is not None:
            try:
                camera.release()
            finally:
                camera = None


def mark_liveness_verified():
    session["liveness_verified_at"] = time.time()
    session.modified = True


def has_recent_liveness():
    verified_at = session.get("liveness_verified_at")
    if not verified_at:
        return False
    return (time.time() - float(verified_at)) <= LIVENESS_VALID_SECONDS


# ================= DATASET & MODEL SYNC =================
def dataset_has_faces():
    if not os.path.exists(DATASET_DIR):
        return False

    for folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path) or not folder.startswith("user_"):
            continue
        image_files = [
            filename for filename in os.listdir(folder_path)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if image_files:
            return True
    return False


def get_voter_proof(voter_id):
    proof_file = os.path.join(DATASET_DIR, f"user_{voter_id}", "proof.json")
    if os.path.exists(proof_file):
        try:
            with open(proof_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def save_voter_proof(voter_id, timestamp):
    person_dir = os.path.join(DATASET_DIR, f"user_{voter_id}")
    os.makedirs(person_dir, exist_ok=True)
    proof_file = os.path.join(person_dir, "proof.json")
    proof_data = {
        "timestamp": timestamp,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(proof_file, "w", encoding="utf-8") as f:
        json.dump(proof_data, f, indent=2)


def reset_model():
    global recognizer
    with model_lock:
        recognizer = None
        if os.path.exists(MODEL_PATH):
            try:
                os.remove(MODEL_PATH)
            except OSError:
                pass
    print("Model reset: no usable dataset/model")


def load_model_if_valid():
    global recognizer

    with model_lock:
        if dataset_has_faces() and os.path.exists(MODEL_PATH):
            try:
                candidate = cv2.face.LBPHFaceRecognizer_create()
                candidate.read(MODEL_PATH)
                recognizer = candidate
                return True
            except cv2.error as exc:
                print(f"Model load failed: {exc}")
                recognizer = None
                return False
        recognizer = None
        return False


load_model_if_valid()

# ================= LOGIN ROUTES =================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session.clear()
            session["user"] = username
            return redirect(url_for("dashboard"))

        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/logout")
def logout():
    release_camera()
    session.clear()
    set_overlay("System ready", "idle")
    return redirect(url_for("login"))

# ================= CAMERA CONTROL =================
@app.route("/start_camera")
def start_camera():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 403

    global camera

    with camera_lock:
        if camera is None or not camera.isOpened():
            new_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not new_camera.isOpened():
                new_camera.release()
                # Fallback for systems where CAP_DSHOW is unavailable.
                new_camera = cv2.VideoCapture(0)

            if not new_camera.isOpened():
                new_camera.release()
                camera = None
                return jsonify({
                    "status": "error",
                    "message": "Could not open webcam. Check camera permissions/device."
                }), 500

            # Keep the existing UI proportions but request a reasonable capture size.
            new_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            new_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera = new_camera

    set_overlay("Camera ready", "idle")
    return jsonify({"status": "Camera started"})


@app.route("/stop_camera")
def stop_camera():
    release_camera()
    session.pop("liveness_verified_at", None)
    set_overlay("Camera stopped", "idle")
    return jsonify({"status": "Camera stopped"})

# ---------------- VIDEO STREAM ----------------
def gen_frames():
    while True:
        ok, frame = read_camera_frame()
        if not ok or frame is None:
            break

        current_overlay = get_overlay()
        text = current_overlay["text"]
        state = current_overlay["state"]

        if text:
            if state == "passed":
                text_color = (40, 180, 80)
            elif state == "failed":
                text_color = (40, 70, 220)
            elif state == "checking":
                text_color = (0, 165, 255)
            else:
                text_color = (255, 255, 255)

            cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 630), 55), (15, 23, 42), -1)
            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                text_color,
                2,
                cv2.LINE_AA,
            )

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/video")
def video():
    if not is_logged_in():
        return "", 403

    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ================= SECURE VOTER CHECK =================
@app.route("/check_voter")
def check_voter():
    global recognizer, verification_in_progress, system_state

    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 403

    if verification_in_progress:
        return jsonify({
            "state": "BUSY",
            "message": "A verification is already in progress.",
            "color": "orange"
        }), 409

    if camera is None or not camera.isOpened():
        return jsonify({
            "state": "IDLE",
            "message": "Camera not started.",
            "color": "red"
        })

    verification_in_progress = True
    system_state = "LIVENESS_CHECK"
    session.pop("liveness_verified_at", None)

    try:
        # -------- STEP 1: ACTIVE LIVENESS --------
        liveness_result = run_blink_liveness(
            read_frame=read_camera_frame,
            face_mesh=face_mesh,
            set_status=set_overlay,
        )

        if not liveness_result["success"]:
            system_state = "LIVENESS_FAILED"
            return jsonify({
                "state": "LIVENESS_FAILED",
                "message": liveness_result["message"],
                "color": "red",
                "reason": liveness_result["reason"]
            })

        mark_liveness_verified()
        system_state = "FACE_CHECK"
        set_overlay("Liveness verified - matching face...", "checking")

        # -------- STEP 2: FACE RECOGNITION --------
        load_model_if_valid()

        ok, frame = read_camera_frame()
        if not ok or frame is None:
            return jsonify({
                "state": "IDLE",
                "message": "Camera error after liveness check.",
                "color": "red"
            }), 500

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({
                "state": "CHECKING",
                "message": "Liveness passed, but no face is visible. Try again.",
                "color": "orange"
            })

        if len(faces) > 1:
            return jsonify({
                "state": "MULTIPLE_FACES",
                "message": "Multiple faces detected. Only one voter may be in front of the camera.",
                "color": "red"
            })

        # ---- CASE 1: NO MODEL -> NEW VOTER ----
        if recognizer is None:
            system_state = "NEW_VOTER"
            set_overlay("Liveness verified - new voter", "passed")
            return jsonify({
                "state": "NEW_VOTER",
                "message": "Liveness verified. New voter detected. Registration required.",
                "color": "green"
            })

        # ---- CASE 2: MULTI-FRAME FACE MATCH ----
        confidences = []
        labels_seen = []

        for _ in range(NUM_READINGS):
            ret, frame = read_camera_frame()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) != 1:
                continue

            x, y, w, h = faces[0]
            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))

            try:
                with model_lock:
                    lbl, conf = recognizer.predict(face_img)
            except cv2.error:
                lbl, conf = None, None

            if lbl is not None and conf is not None:
                labels_seen.append(int(lbl))
                confidences.append(float(conf))

        if not confidences:
            set_overlay("Face matching failed - try again", "failed")
            return jsonify({
                "state": "CHECKING",
                "message": "Face unclear. Keep your face centered and try again.",
                "color": "orange"
            })

        avg_confidence = sum(confidences) / len(confidences)
        most_common_label = max(set(labels_seen), key=labels_seen.count)

        # -------- FINAL DECISION --------
        if avg_confidence < THRESHOLD:
            proof = get_voter_proof(most_common_label)
            system_state = "DUPLICATE"
            set_overlay(
                f"Voter matched - ID {most_common_label}",
                "failed"
            )
            return jsonify({
                "state": "DUPLICATE",
                "message": f"Voter already registered (ID {most_common_label}).",
                "color": "red",
                "voter_id": most_common_label,
                "proof": proof,
                "liveness_verified": True,
                "confidence": round(avg_confidence, 2)
            })

        system_state = "NEW_VOTER"
        set_overlay("Liveness verified - new voter", "passed")
        return jsonify({
            "state": "NEW_VOTER",
            "message": "Liveness verified. New voter detected. Registration required.",
            "color": "green",
            "liveness_verified": True,
            "confidence": round(avg_confidence, 2)
        })

    finally:
        verification_in_progress = False

# ================= REGISTER FACE =================
@app.route("/register_face")
def register_face():
    global recognizer, system_state

    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 403

    if not has_recent_liveness():
        return jsonify({
            "message": "Liveness verification is required before registration. Click Check Voter first.",
            "success": False,
            "state": "LIVENESS_REQUIRED"
        }), 403

    if camera is None or not camera.isOpened():
        return jsonify({"message": "Camera not started", "success": False}), 400

    ret, frame = read_camera_frame()
    if not ret or frame is None:
        return jsonify({"message": "Camera error", "success": False}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"message": "No face detected", "success": False}), 400

    if len(faces) > 1:
        return jsonify({
            "message": "Multiple faces detected. Only one voter may be registered at a time.",
            "success": False
        }), 400

    x, y, w, h = faces[0]
    face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))

    existing_ids = []
    for folder in os.listdir(DATASET_DIR):
        if folder.startswith("user_"):
            try:
                existing_ids.append(int(folder.split("_")[1]))
            except (IndexError, ValueError):
                continue

    new_id = max(existing_ids) + 1 if existing_ids else 1

    person_dir = os.path.join(DATASET_DIR, f"user_{new_id}")
    os.makedirs(person_dir, exist_ok=True)

    image_path = os.path.join(person_dir, "1.jpg")
    if not cv2.imwrite(image_path, face_img):
        return jsonify({"message": "Could not save face image", "success": False}), 500

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_voter_proof(new_id, timestamp)

    # ---- RETRAIN MODEL ----
    faces_data = []
    labels = []
    folders = sorted(
        [
            folder for folder in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, folder)) and folder.startswith("user_")
        ],
        key=lambda name: int(name.split("_")[1])
    )

    for folder in folders:
        folder_path = os.path.join(DATASET_DIR, folder)
        voter_id = int(folder.split("_")[1])

        for img in os.listdir(folder_path):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder_path, img)
                img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_gray is not None:
                    faces_data.append(img_gray)
                    labels.append(voter_id)

    if not faces_data:
        return jsonify({"message": "No face images available for training", "success": False}), 500

    try:
        new_recognizer = cv2.face.LBPHFaceRecognizer_create()
        new_recognizer.train(faces_data, np.array(labels, dtype=np.int32))
        new_recognizer.save(MODEL_PATH)
        with model_lock:
            recognizer = new_recognizer
    except cv2.error as exc:
        return jsonify({
            "message": f"Model training failed: {exc}",
            "success": False
        }), 500

    session.pop("liveness_verified_at", None)
    system_state = "IDLE"
    set_overlay("Registration completed", "passed")

    return jsonify({
        "message": "Face registered successfully after liveness verification.",
        "state": "IDLE",
        "voter_id": new_id,
        "timestamp": timestamp,
        "success": True,
        "liveness_verified": True
    })

# ================= VOTER IMAGE / RESET =================
@app.route("/get_voter_image/<int:voter_id>")
def get_voter_image(voter_id):
    if not is_logged_in():
        return "", 403

    img_path = os.path.join(DATASET_DIR, f"user_{voter_id}", "1.jpg")
    if os.path.exists(img_path):
        try:
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            return jsonify({"image": f"data:image/jpeg;base64,{img_data}"})
        except OSError:
            return jsonify({"error": "Could not read image"}), 500

    return jsonify({"error": "Image not found"}), 404


@app.route("/reset")
def reset_system():
    global system_state, verification_in_progress

    release_camera()
    session.pop("liveness_verified_at", None)
    system_state = "IDLE"
    verification_in_progress = False
    set_overlay("System ready", "idle")

    return jsonify({
        "message": "System ready. Waiting for next voter.",
        "state": "IDLE",
        "camera": "OFF"
    })


@app.route("/force_stop_camera", methods=["POST"])
def force_stop_camera():
    global system_state, verification_in_progress

    release_camera()
    session.pop("liveness_verified_at", None)
    system_state = "IDLE"
    verification_in_progress = False
    set_overlay("Camera stopped", "idle")
    return jsonify({"status": "Camera stopped"})


@app.route("/liveness_info")
def liveness_info():
    if not is_logged_in():
        return jsonify({"error": "Unauthorized"}), 403

    return jsonify({
        "active": verification_in_progress,
        "liveness_verified": has_recent_liveness(),
        "state": system_state,
        "overlay": get_overlay()
    })


# ============================================================================
# STEP 2: VOICE VERIFICATION (ADDED MODULE)
# The original face/liveness routes above are intentionally left unchanged.
# ============================================================================
from voice_biometrics import (
    VOICE_CHALLENGE_TEXT,
    VOICE_ENROLLMENT_SAMPLES,
    VOICE_SAMPLE_RATE,
    VOICE_SIMILARITY_THRESHOLD,
    VOICE_REFERENCE_MIN_THRESHOLD,
    VOICE_DIFFERENT_SPEAKER_MARGIN,
    decode_wav_bytes,
    encode_wav_bytes,
    load_voice_profile,
    profile_similarity,
    profile_similarity_scores,
    save_voice_profile,
    transcript_matches,
    transcript_similarity,
    voice_embedding,
)

FACE_TO_VOICE_WINDOW_SECONDS = 600
VOICE_PROFILE_FILENAME = "voice_profile.npz"
VOICE_REFERENCE_FILENAME = "voice_reference.wav"


def _voice_profile_path(voter_id: int) -> str:
    return os.path.join(DATASET_DIR, f"user_{int(voter_id)}", VOICE_PROFILE_FILENAME)


def _voice_reference_path(voter_id: int, sample_number: int = 1) -> str:
    if sample_number == 1:
        filename = VOICE_REFERENCE_FILENAME
    else:
        filename = f"voice_reference_{int(sample_number)}.wav"
    return os.path.join(DATASET_DIR, f"user_{int(voter_id)}", filename)


def _is_registered_demo_voter(voter_id: int) -> bool:
    """Check that the existing project registry has a face and registration proof."""
    person_dir = os.path.join(DATASET_DIR, f"user_{int(voter_id)}")
    return (
        os.path.isfile(os.path.join(person_dir, "1.jpg"))
        and os.path.isfile(os.path.join(person_dir, "proof.json"))
    )


def _registered_demo_voter_ids():
    voter_ids = []
    try:
        folders = os.listdir(DATASET_DIR)
    except OSError:
        return voter_ids
    for folder in folders:
        if not folder.startswith("user_"):
            continue
        suffix = folder[5:]
        if suffix.isdigit() and _is_registered_demo_voter(int(suffix)):
            voter_ids.append(int(suffix))
    return sorted(set(voter_ids))


def _face_step_is_current(voter_id: int) -> bool:
    verified_id = session.get("face_verified_voter_id")
    verified_at = session.get("face_verified_at")
    if verified_id is None or verified_at is None:
        return False
    try:
        return (
            int(verified_id) == int(voter_id)
            and (time.time() - float(verified_at)) <= FACE_TO_VOICE_WINDOW_SECONDS
        )
    except (TypeError, ValueError):
        return False


@app.after_request
def _remember_successful_face_match_for_step_two(response):
    """Bind the existing step-one success result to its voter ID without changing it."""
    if request.endpoint == "check_voter":
        result = response.get_json(silent=True)
        if (
            response.status_code == 200
            and isinstance(result, dict)
            and result.get("state") == "DUPLICATE"
            and result.get("liveness_verified") is True
            and result.get("voter_id") is not None
        ):
            session["face_verified_voter_id"] = int(result["voter_id"])
            session["face_verified_at"] = time.time()
            session.pop("voice_verified_voter_id", None)
            session.pop("voice_verified_at", None)
        else:
            session.pop("face_verified_voter_id", None)
            session.pop("face_verified_at", None)
            session.pop("voice_verified_voter_id", None)
            session.pop("voice_verified_at", None)
    return response


@app.route("/voice-verification/<int:voter_id>")
def voice_verification_page(voter_id):
    if not is_logged_in():
        return redirect(url_for("login"))
    if not _face_step_is_current(voter_id):
        return redirect(url_for("dashboard"))

    if not _is_registered_demo_voter(voter_id):
        return redirect(url_for("dashboard"))

    has_reference_voice = False
    profile_path = _voice_profile_path(voter_id)
    if os.path.isfile(profile_path):
        try:
            # Validate the format/model ID before showing the recording controls.
            # This is metadata validation only; the model itself loads on first use.
            load_voice_profile(profile_path)
            has_reference_voice = True
        except (OSError, ValueError):
            has_reference_voice = False

    return render_template(
        "voice_verification.html",
        voter_id=voter_id,
        phrase=VOICE_CHALLENGE_TEXT,
        has_reference_voice=has_reference_voice,
    )


@app.route("/voice-enroll")
def voice_enrollment_page():
    if not is_logged_in():
        return redirect(url_for("login"))
    requested_id = request.args.get("voter_id", type=int)
    if requested_id is not None and not _is_registered_demo_voter(requested_id):
        requested_id = None
    return render_template(
        "voice_enrollment.html",
        voter_id=requested_id,
        voter_ids=_registered_demo_voter_ids(),
        phrase=VOICE_CHALLENGE_TEXT,
    )


@app.route("/api/enroll-voice", methods=["POST"])
def api_enroll_voice():
    if not is_logged_in():
        return jsonify({"state": "UNAUTHORIZED", "message": "Please log in first."}), 403

    voter_id_text = request.form.get("voter_id", "").strip()
    if not voter_id_text.isdigit() or int(voter_id_text) <= 0:
        return jsonify({"state": "INVALID_VOTER", "message": "Enter a valid voter ID."}), 400
    voter_id = int(voter_id_text)
    if not _is_registered_demo_voter(voter_id):
        return jsonify({
            "state": "NOT_REGISTERED",
            "message": f"Voter ID {voter_id} is not present in the existing face-registration dataset."
        }), 404

    transcripts = request.form.getlist("transcript")
    audio_files = request.files.getlist("audio")
    if len(audio_files) != VOICE_ENROLLMENT_SAMPLES or len(transcripts) != VOICE_ENROLLMENT_SAMPLES:
        return jsonify({
            "state": "SAMPLES_REQUIRED",
            "message": f"Record all {VOICE_ENROLLMENT_SAMPLES} voice samples before saving this voter's profile."
        }), 400

    embeddings = []
    canonical_recordings = []
    for index, (transcript, audio_file) in enumerate(zip(transcripts, audio_files), start=1):
        if audio_file is None or not audio_file.filename:
            return jsonify({"state": "NO_AUDIO", "message": f"Voice sample {index} is missing. Please record it again."}), 400
        text_score = transcript_similarity(transcript)
        if not transcript_matches(transcript):
            return jsonify({
                "state": "SENTENCE_MISMATCH",
                "message": f"Voice sample {index} did not match the displayed sentence. No profile was changed; please record all three samples again.",
                "transcript": transcript,
                "sample_number": index,
                "text_score": round(text_score, 3),
            }), 400
        try:
            samples = decode_wav_bytes(audio_file.read())
            embedding = voice_embedding(samples)
            canonical_wav = encode_wav_bytes(samples, VOICE_SAMPLE_RATE)
        except ValueError as exc:
            return jsonify({"state": "INVALID_AUDIO", "message": f"Sample {index}: {exc}"}), 400
        except RuntimeError as exc:
            app.logger.exception("Speaker model unavailable during enrollment")
            return jsonify({"state": "VOICE_MODEL_UNAVAILABLE", "message": str(exc)}), 503
        except Exception:
            app.logger.exception("Voice enrollment failed for sample %s", index)
            return jsonify({"state": "VOICE_PROCESSING_ERROR", "message": f"Could not process sample {index}. Please record it again."}), 500
        embeddings.append(embedding)
        canonical_recordings.append(canonical_wav)

    person_dir = os.path.join(DATASET_DIR, f"user_{voter_id}")
    profile_path = _voice_profile_path(voter_id)
    pending_profile_path = profile_path + ".pending"
    try:
        os.makedirs(person_dir, exist_ok=True)
        # Validate and stage the new profile BEFORE replacing any current files.
        # If the three recordings do not look like the same speaker, the prior
        # valid profile is preserved and this enrollment is rejected.
        save_voice_profile(pending_profile_path, embeddings)
        for index, wav_bytes in enumerate(canonical_recordings, start=1):
            target_path = _voice_reference_path(voter_id, index)
            temp_path = target_path + ".tmp"
            with open(temp_path, "wb") as wav_out:
                wav_out.write(wav_bytes)
            os.replace(temp_path, target_path)
        os.replace(pending_profile_path, profile_path)
    except (OSError, ValueError, RuntimeError) as exc:
        try:
            if os.path.exists(pending_profile_path):
                os.remove(pending_profile_path)
        except OSError:
            pass
        app.logger.exception("Could not save voice profile for voter %s", voter_id)
        return jsonify({"state": "SAVE_FAILED", "message": f"The voice profile could not be saved: {exc}"}), 400 if isinstance(exc, ValueError) else 500

    return jsonify({
        "state": "ENROLLED",
        "message": f"Speaker profile saved for voter ID {voter_id} from {VOICE_ENROLLMENT_SAMPLES} separate voice samples.",
        "voter_id": voter_id,
        "reference_file": f"dataset/user_{voter_id}/{VOICE_PROFILE_FILENAME}",
        "sample_files": [f"dataset/user_{voter_id}/{os.path.basename(_voice_reference_path(voter_id, i))}" for i in range(1, VOICE_ENROLLMENT_SAMPLES + 1)],
        "sample_count": VOICE_ENROLLMENT_SAMPLES,
        "text_score": round(sum(transcript_similarity(t) for t in transcripts) / len(transcripts), 3),
    })


@app.route("/api/verify-voice", methods=["POST"])
def api_verify_voice():
    if not is_logged_in():
        return jsonify({"state": "UNAUTHORIZED", "message": "Please log in first."}), 403

    voter_id_text = request.form.get("voter_id", "").strip()
    if not voter_id_text.isdigit() or int(voter_id_text) <= 0:
        return jsonify({"state": "INVALID_VOTER", "message": "The voter ID is invalid."}), 400
    voter_id = int(voter_id_text)

    if not _face_step_is_current(voter_id):
        return jsonify({
            "state": "FACE_STEP_EXPIRED",
            "message": "The face verification session expired. Please return to Step 1 and verify again."
        }), 403
    if not _is_registered_demo_voter(voter_id):
        return jsonify({
            "state": "NOT_REGISTERED",
            "message": "No registered demo voter record was found for this ID."
        }), 404

    profile_path = _voice_profile_path(voter_id)
    if not os.path.isfile(profile_path):
        return jsonify({
            "state": "VOICE_NOT_ENROLLED",
            "message": f"This voter has no current speaker profile. Enroll {VOICE_ENROLLMENT_SAMPLES} voice samples for this voter, then repeat Step 1."
        }), 409

    transcript = request.form.get("transcript", "")
    text_score = transcript_similarity(transcript)
    if not transcript_matches(transcript):
        session.pop("voice_verified_voter_id", None)
        session.pop("voice_verified_at", None)
        return jsonify({
            "state": "SENTENCE_MISMATCH",
            "message": "The spoken sentence does not match the required sentence. Please try again.",
            "transcript": transcript,
            "text_score": round(text_score, 3),
        }), 400

    audio_file = request.files.get("audio")
    if audio_file is None or not audio_file.filename:
        return jsonify({"state": "NO_AUDIO", "message": "No microphone recording was received."}), 400

    try:
        live_samples = decode_wav_bytes(audio_file.read())
        live_embedding = voice_embedding(live_samples)
        target_profile = load_voice_profile(profile_path)
        target_reference_scores = profile_similarity_scores(target_profile, live_embedding)
        similarity = float(np.mean(target_reference_scores))
        weakest_reference_score = float(np.min(target_reference_scores))

        # Closed-set cross-check: where other demo voters have their own enrolled
        # speaker profiles, the target must also be closer than those profiles.
        closest_other_score = None
        closest_other_id = None
        for other_voter_id in _registered_demo_voter_ids():
            if other_voter_id == voter_id:
                continue
            other_profile_path = _voice_profile_path(other_voter_id)
            if not os.path.isfile(other_profile_path):
                continue
            try:
                other_profile = load_voice_profile(other_profile_path)
                other_score = profile_similarity(other_profile, live_embedding)
            except (OSError, ValueError):
                # A damaged other profile is not allowed to crash a verification.
                continue
            if closest_other_score is None or other_score > closest_other_score:
                closest_other_score = other_score
                closest_other_id = other_voter_id
    except ValueError as exc:
        session.pop("voice_verified_voter_id", None)
        session.pop("voice_verified_at", None)
        return jsonify({"state": "INVALID_AUDIO_OR_PROFILE", "message": str(exc)}), 400
    except RuntimeError as exc:
        session.pop("voice_verified_voter_id", None)
        session.pop("voice_verified_at", None)
        app.logger.exception("Speaker model unavailable during voice verification")
        return jsonify({"state": "VOICE_MODEL_UNAVAILABLE", "message": str(exc)}), 503
    except OSError:
        app.logger.exception("Could not load voice profile for voter %s", voter_id)
        return jsonify({"state": "REFERENCE_READ_FAILED", "message": "The stored voice profile could not be read. Re-enroll this voter."}), 500
    except Exception:
        app.logger.exception("Voice verification failed")
        return jsonify({"state": "VOICE_PROCESSING_ERROR", "message": "Could not compare the speaker profile. Please try again."}), 500

    percent = max(0.0, min(100.0, similarity * 100.0))
    other_percent = None if closest_other_score is None else max(0.0, min(100.0, closest_other_score * 100.0))
    clears_other_margin = (
        closest_other_score is None
        or similarity >= closest_other_score + VOICE_DIFFERENT_SPEAKER_MARGIN
    )
    # Require every enrolled take to support the decision. This prevents one or
    # two unusually high scores from hiding a weak match to the saved speaker.
    all_references_match = weakest_reference_score >= VOICE_REFERENCE_MIN_THRESHOLD
    voice_match = (
        similarity >= VOICE_SIMILARITY_THRESHOLD
        and all_references_match
        and clears_other_margin
    )

    if voice_match:
        session["voice_verified_voter_id"] = voter_id
        session["voice_verified_at"] = time.time()
        return jsonify({
            "state": "ELIGIBLE_TO_PROCEED",
            "message": "Both verification steps passed. The demo registry record is verified; you may proceed in the demo voting workflow.",
            "voter_id": voter_id,
            "sentence_match": True,
            "voice_match": True,
            "voice_similarity": round(percent, 2),
            "reference_similarities": [round(float(score) * 100.0, 2) for score in target_reference_scores],
            "weakest_reference_similarity": round(weakest_reference_score * 100.0, 2),
            "reference_threshold": round(VOICE_REFERENCE_MIN_THRESHOLD * 100.0, 2),
            "threshold": round(VOICE_SIMILARITY_THRESHOLD * 100.0, 2),
            "other_voter_similarity": None if other_percent is None else round(other_percent, 2),
            "text_score": round(text_score, 3),
        })

    session.pop("voice_verified_voter_id", None)
    session.pop("voice_verified_at", None)
    if similarity < VOICE_SIMILARITY_THRESHOLD:
        mismatch_reason = "The sentence matched, but the average speaker similarity was below the strict acceptance threshold."
    elif not all_references_match:
        mismatch_reason = "The sentence matched, but the voice did not match all three enrolled reference recordings consistently."
    else:
        mismatch_reason = f"The recording was closer to another enrolled voter profile (ID #{closest_other_id}) or did not clear the required speaker-separation margin."
    return jsonify({
        "state": "VOICE_MISMATCH",
        "message": mismatch_reason + " You are not verified to proceed. Re-record the reference samples if enrollment quality was poor, then try again.",
        "voter_id": voter_id,
        "sentence_match": True,
        "voice_match": False,
        "voice_similarity": round(percent, 2),
        "reference_similarities": [round(float(score) * 100.0, 2) for score in target_reference_scores],
        "weakest_reference_similarity": round(weakest_reference_score * 100.0, 2),
        "reference_threshold": round(VOICE_REFERENCE_MIN_THRESHOLD * 100.0, 2),
        "threshold": round(VOICE_SIMILARITY_THRESHOLD * 100.0, 2),
        "other_voter_similarity": None if other_percent is None else round(other_percent, 2),
        "text_score": round(text_score, 3),
    }), 401


@app.route("/voice-status")
def voice_status():
    """Return the result of the latest demo voice verification for the session."""
    if not is_logged_in():
        return jsonify({"state": "UNAUTHORIZED"}), 403
    voter_id = session.get("voice_verified_voter_id")
    verified_at = session.get("voice_verified_at")
    verified = False
    if voter_id is not None and verified_at is not None:
        try:
            verified = (time.time() - float(verified_at)) <= FACE_TO_VOICE_WINDOW_SECONDS
        except (TypeError, ValueError):
            verified = False
    return jsonify({"verified": verified, "voter_id": voter_id if verified else None})


if __name__ == "__main__":
    # Disable Flask's reloader because it can initialize camera/ML resources twice.
    app.run(debug=True, use_reloader=False)
