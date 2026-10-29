# Project Changes

## Existing face and liveness stage
- The original face recognition, active blink-liveness, registration logic, trained face model, Haar cascades, and dashboard remain intact.
- Changes to `src/app.py` are confined to the Step 2 voice-verification section and voice enrollment save/decision logic; the Step 1 flow itself was not intentionally altered.

## Step 2 voice verification
- Replaced Resemblyzer 256-D embeddings with SpeechBrain's pretrained ECAPA-TDNN 192-D speaker embedding model (`speechbrain/spkrec-ecapa-voxceleb`).
- The old “mean of best two out of three” profile score is removed. The verifier scores all three reference recordings individually and uses their mean plus a minimum-per-reference condition.
- Current decision requires mean target score >= 0.55, every reference score >= 0.45, and a 0.12 score margin over the closest other enrolled voter when another current profile exists.
- Enrollment requires three separate takes and rejects a profile if its three recordings are not internally consistent enough to look like one speaker.
- New voice profiles are validated/staged before replacing the existing profile. Inconsistent enrollment is rejected.
- No silent fallback is allowed: if the speaker model is not installed or available, the voice check fails instead of accepting a matching transcript.
- Added Windows setup batch file and dependency checker, including installation/verification of `opencv-contrib-python` to solve missing `cv2` errors.
- Updated setup and troubleshooting guidance.

## Verification limits
- Python syntax, helper logic, archive integrity, and dependency metadata are checked during packaging. A live evaluation using the user's and sister's recordings cannot be performed in this build environment.
- No pretrained speaker model can guarantee perfect outcomes in every recording condition. Thresholds require testing and calibration on multiple consented genuine and impostor recordings. Educational prototype only; never use for real voting eligibility.


## Windows ECAPA model-loading hotfix
- SpeechBrain model loading now requests `LocalStrategy.COPY`, avoiding the Windows `WinError 1314` caused by attempted symbolic-link creation in the project model cache.
- The Hugging Face symlink-cache warning is documented as non-fatal; it is distinct from the SpeechBrain savedir symlink that previously crashed enrollment.
- If the prior attempt created a partial model directory and still reports `WinError 1314`, the loader clears that incomplete local folder and gives a Windows-specific recovery message.
