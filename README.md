# Biometric Voting System with Face, Liveness, and Voice Verification

An academic/demo web application that demonstrates a two-step biometric verification flow:

1. **Face recognition and blink-based liveness verification**
2. **Voice challenge verification**, checking both the required sentence and the speaker's voice against an enrolled reference profile

> **Important:** This is an educational prototype, not an official election system. Biometric verification can produce false accepts or false rejects and must not be used to decide real-world voting eligibility or for other high-stakes decisions.

## Features

- Flask web interface with a demo officer login.
- Face registration and recognition using OpenCV and a trained face model.
- Blink-based liveness check using MediaPipe Face Mesh.
- Two-step flow: a successful Step 1 result can route the same registered demo voter to voice verification.
- Voice enrollment with **three recordings** for each registered demo voter.
- Exact challenge-sentence validation after case and punctuation normalization.
- Speaker comparison using SpeechBrain's pretrained **ECAPA-TDNN** model.
- Conservative matching rules that score all three reference recordings, check enrollment consistency, and compare against other enrolled profiles when available.
- Adaptive microphone-level checks to better handle quieter microphones.
- Windows setup script and dependency checker.

## Technology Stack

- **Language:** Python
- **Web framework:** Flask
- **Face processing:** OpenCV / OpenCV-contrib
- **Liveness:** MediaPipe Face Mesh
- **Voice speaker embeddings:** SpeechBrain ECAPA-TDNN
- **Audio/data processing:** NumPy, SciPy, SoundFile, PyTorch, Torchaudio
- **Frontend:** HTML, CSS, JavaScript

## Project Structure

```text
Biometric_Voting_System_Liveness/
├── README.md
├── Run.txt
├── VOICE_VERIFICATION_GUIDE.md
├── CHANGES.md
├── requirements.txt
├── setup_windows.bat
├── check_setup.py
├── dataset/
│   ├── VOICE_SAMPLE_INSTRUCTIONS.txt
│   └── user_<id>/
│       ├── 1.jpg
│       ├── proof.json
│       ├── voice_profile.npz
│       ├── voice_reference.wav
│       ├── voice_reference_2.wav
│       └── voice_reference_3.wav
├── models/
│   ├── face_model.yml
│   ├── haarcascade_eye.xml
│   ├── haarcascade_frontalface_default.xml
│   └── speaker_ecapa/        # Pretrained model cache, populated on first use
├── src/
│   ├── app.py
│   ├── liveness.py
│   ├── recognize_and_vote.py
│   ├── register_face.py
│   ├── train_model.py
│   └── voice_biometrics.py
├── templates/
└── static/
```

The exact files within `dataset/user_<id>/` depend on whether that demo voter has completed voice enrollment. Do not combine different people's voice recordings under one voter ID.

## Requirements

- Windows 10/11, 64-bit
- **Python 3.10 x64** and the Windows `py` launcher
- A webcam
- A microphone
- Chrome or Microsoft Edge with camera and microphone permissions enabled
- Internet access for the first download of the pretrained ECAPA-TDNN model

Python 3.10 x64 is recommended because the pinned dependencies are selected for compatibility with the current MediaPipe-based project.

## Installation on Windows

### Recommended: automatic setup

1. Extract the ZIP to a normal folder. Do not run the application from inside the ZIP.
2. Open the extracted project folder.
3. Double-click **`setup_windows.bat`** and wait for dependency installation and checks to finish.
4. If setup reports success, open PowerShell in that same project folder.
5. Start the app with:

   ```powershell
   .venv\Scripts\python.exe src\app.py
   ```

6. Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in Chrome or Edge.

The setup script creates a project-local `.venv`, installs `requirements.txt`, and runs `check_setup.py` to check imports, including `cv2.face`.

### Manual setup

Run these commands from the project root in PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe check_setup.py
.venv\Scripts\python.exe src\app.py
```

Use the same `.venv` Python executable for setup and launch. Avoid installing `opencv-python` alongside `opencv-contrib-python`; both provide the `cv2` package and can conflict. This project uses `opencv-contrib-python` because the face recognizer requires `cv2.face`.

## Demo Login

The source currently contains the following demo credentials:

- **Username:** `admin`
- **Password:** `admin123`

These credentials are for local demonstration only. Change the authentication implementation and secrets before exposing the app to any network. Do not treat the demo login as production security.

## How to Use the Application

### 1. Register demo voters

1. Log in and open the dashboard.
2. Register each participant's face individually, with only one person in front of the camera.
3. Complete the face/liveness flow and follow the dashboard's registration prompts.
4. Confirm that each participant has their own voter ID, such as `user_1`, `user_2`, and so on.

The voice enrollment page only accepts voter IDs that exist in the project's face-registration dataset.

### 2. Enroll a voice profile

1. Open **Voice Enrollment** from the dashboard.
2. Select the voter ID that belongs to the person whose voice is being enrolled.
3. That same person must record the displayed sentence three times, with short pauses between recordings, in a quiet room.
4. Wait for enrollment to finish and confirm the success message.
5. Repeat for each demo voter, using each person's own ID.

The required challenge sentence is:

> I confirm my identity for voter verification, and I am speaking naturally to confirm that this voice belongs to me.

A saved profile typically consists of `voice_profile.npz` and three reference WAV files in the corresponding `dataset/user_<id>/` folder. Older profiles from a previous voice model/version may be incompatible; re-enroll them through the enrollment page.

### 3. Verify a voter

1. Start the camera and complete **Step 1: face recognition and blink-liveness**.
2. After a successful registered-face result, proceed to **Step 2: voice verification**.
3. Speak the entire displayed sentence in your normal voice.
4. The app checks the transcript and independently compares the speaker embedding against the enrolled reference profile.
5. A matching sentence alone is not sufficient. The speaker checks must also pass before the demo session is marked verified.

For a useful demo test, enroll a friend or family member under their own voter ID, then test that person's voice against a different enrolled ID. Use recordings only with the participants' permission.

## Voice Matching Rules

The current source uses these conservative starting thresholds:

| Check | Current rule |
|---|---:|
| Mean speaker similarity across all three references | At least `0.55` |
| Similarity to each individual reference recording | At least `0.45` |
| Margin over the closest other enrolled voice profile, if available | At least `0.12` |
| Enrollment consistency | All three takes must be sufficiently similar to each other |

These are cosine-similarity scores, **not percentages of certainty or probabilities that a person is genuine**. Results depend on the microphones, room noise, speaking style, reference quality, and how similar speakers sound. Strict settings can reject the correct person; relaxing them too much may admit a different person. Calibrate only with a suitable collection of consented genuine-speaker and different-speaker trials.

If the speaker model cannot load, the application is intended to fail closed rather than accept a matching sentence by itself.

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`

This usually means the app was started with a different Python environment than the one that has the project dependencies. Run `setup_windows.bat`, then launch with:

```powershell
.venv\Scripts\python.exe src\app.py
```

Do not use a different global Python or an old `venv`. If the error persists, run `.venv\Scripts\python.exe check_setup.py` and follow the reported missing dependency.

### `cv2.face` is missing

Check that `opencv-contrib-python` is installed inside `.venv` and that `opencv-python` has not also been installed into the same environment. Re-run setup if necessary.

### SpeechBrain / ECAPA model download fails

The pretrained model downloads from Hugging Face on the first voice enrollment or verification. Ensure internet access is available and retry. This version asks SpeechBrain to copy cached files instead of creating symbolic links, which avoids the common Windows `WinError 1314` symlink-permission failure. A Hugging Face warning about its general cache symlinks may still appear; by itself, that warning is not necessarily fatal.

If a previous model download was interrupted and loading continues to fail, close the app and remove the incomplete project cache folder `models/speaker_ecapa/`, then restart the app and retry while online.

### “Too little clear speech was detected”

- Allow microphone access for the local site in Chrome/Edge.
- Check the selected Windows input microphone.
- Move roughly 15–20 cm from the microphone and speak at a natural, clear volume.
- Use a quiet room and say the complete sentence without long pauses.
- Retry the recording. This message means audio quality checks rejected the recording; it does not by itself mean the speaker identity failed.

### Sentence passes but voice identity fails

This is the expected result when the transcript matches but the speaker comparison does not. Re-enroll only if the correct person's reference samples are poor or inconsistent, and ensure all three enrollment recordings are from that same person. Do not enroll your own voice under another person's ID or mix speakers in one profile.

### Old voice profile is rejected

Re-enroll the voter from the Voice Enrollment page with three fresh recordings using the current version. Do not edit `.npz` profiles manually.

## Privacy and Safety

- Store voice samples only with informed permission from the participants.
- Voice references and face images are biometric data; keep the dataset private and exclude it from public repositories or shared ZIPs unless all participants explicitly consent.
- Do not use this academic prototype to approve or deny real voters, identify people in public, or make legal, financial, medical, or other high-impact decisions.
- A fixed sentence plus a speaker embedding does not fully protect against replayed or synthetically generated audio.
- Before any real deployment, the system would require a security review, privacy/legal review, liveness/spoof-resistance testing, measured false-accept and false-reject rates, secure credential handling, and appropriate human oversight.

## Additional Documentation

- [`Run.txt`](Run.txt) — short Windows setup and run instructions.
- [`VOICE_VERIFICATION_GUIDE.md`](VOICE_VERIFICATION_GUIDE.md) — voice enrollment, thresholds, Windows model-loading notes, and voice troubleshooting.
- [`CHANGES.md`](CHANGES.md) — changes made to the current version.
