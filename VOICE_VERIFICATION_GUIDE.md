# Step 2 — Conservative ECAPA-TDNN Voice Verification

## What changed

The previous speaker check used a 256-dimensional Resemblyzer profile and averaged the **two best** matches from three recordings. This could hide a weak or inconsistent score. The current version uses SpeechBrain's pretrained ECAPA-TDNN speaker embedding model (192-dimensional embeddings), a stronger speaker-verification baseline documented by SpeechBrain. The model is downloaded once from Hugging Face on first use and cached under `models/speaker_ecapa/`. On Windows, this version explicitly requests `LocalStrategy.COPY` for SpeechBrain model files to avoid `OSError: [WinError 1314] A required privilege is not held by the client`. You do not need to run as Administrator or enable Developer Mode. The Hugging Face cache may print a symlink warning on Windows; that warning alone is not a failure.

Step 2 keeps two separate requirements:

1. **Sentence:** the browser transcript must match the complete displayed sentence after case and punctuation normalization.
2. **Speaker identity:** the server compares the audio embedding to **all three** enrolled reference embeddings. Acceptance requires the average score to clear a conservative cutoff and every individual reference score to clear its own minimum. If other voter profiles are enrolled, the target must also clear a separation margin over the closest other profile.

The three enrolled takes are checked against each other. If they do not look sufficiently like the same speaker, enrollment is rejected instead of creating a questionable profile. Voice identity is not accepted if the model is unavailable; there is no sentence-only fallback.

## Recommended Windows setup

Use Python 3.10 x64 for compatibility with the existing MediaPipe code.

1. Extract the project ZIP.
2. Run `setup_windows.bat`.
3. Start the app with:

   ```bat
   .venv\Scripts\python.exe src\app.py
   ```

4. Open `http://127.0.0.1:5000/` in Chrome or Edge.

The setup installs `opencv-contrib-python`, which includes `cv2.face`. If `ModuleNotFoundError: No module named 'cv2'` occurs, it means the app was started using a Python environment without the project's requirements. Run `setup_windows.bat`, then use the exact `.venv\Scripts\python.exe` command above.

The first voice enrollment/verification downloads the pretrained ECAPA model. Internet access is required for that first download. Later launches should use the cached model.

## Enroll/test your friends

1. Register each demo participant's face with a distinct voter ID.
2. Open Voice Enrollment and select the ID belonging to the person whose voice is being enrolled.
3. The same person records the full displayed sentence three times, with short pauses between recordings, in a quiet room using the same microphone.
4. If enrollment says the three takes are inconsistent, redo all three takes with one speaker. Never combine your voice and a friend's voice in the same ID.
5. Complete Step 1, then speak the sentence again on Step 2.
6. Test a different person speaking the same sentence against the enrolled ID. It should fail if the ECAPA scores and per-reference checks are below threshold. Enroll your sister under her own ID as well for the optional other-profile separation check.

## Current conservative decision settings

- Mean similarity across the three target references: at least `0.55` (55%).
- Each individual target reference similarity: at least `0.45` (45%).
- Difference from the closest other enrolled profile: at least `0.12` (12 percentage points).
- Enrollment consistency: minimum pairwise cosine score of `0.40` and mean pairwise score of `0.50`.

These are **starter thresholds**, not universal values or a guarantee. They intentionally lean toward rejecting uncertain matches. Different microphones, background noise, distance, illness, or an unfamiliar speaking style may reject the real user. Adjust thresholds only after collecting consented local recordings and measuring both genuine-speaker and different-speaker trials; do not raise them just to force legitimate tests to pass.

## Limitations and privacy

The pretrained ECAPA-TDNN architecture is trained for speaker embeddings, but its published benchmark results do not guarantee performance for this small custom dataset or sibling comparisons. No threshold can guarantee the correct result every time. A fixed sentence and speaker embedding also do not fully prevent replay or synthetic voice attacks. This is a student/demo application, not production biometric authentication and must not be used to approve or deny real voters. Store only consented samples and do not distribute reference WAV files.

### Fix for “Too little clear speech was detected”

The audio quality gate now adapts its speech threshold to the actual microphone level, instead of using a single fixed frame-volume cutoff. Browser automatic gain control is enabled to help quieter microphones. If the message still appears, use Chrome/Edge on `http://127.0.0.1:5000`, permit microphone access, select the intended microphone in Windows, move closer (about 15–20 cm), and record the whole sentence without long pauses. The sentence/transcript check and speaker-identity check remain separate.
