"""Check that runtime packages and the OpenCV-contrib face module are installed."""
from __future__ import annotations
import importlib
import sys

REQUIRED = ["flask", "numpy", "cv2", "mediapipe", "torch", "torchaudio", "speechbrain", "soundfile"]
print(f"Python: {sys.version.split()[0]} ({sys.executable})")
if sys.version_info[:2] != (3, 10):
    print("WARNING: this project is pinned for Python 3.10; another version may be incompatible.")
failed = []
for name in REQUIRED:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "installed")
        print(f"[OK] {name}: {version}")
    except Exception as exc:
        failed.append(name)
        print(f"[FAIL] {name}: {exc}")

if "cv2" not in failed:
    import cv2
    try:
        cv2.face.LBPHFaceRecognizer_create()
        print("[OK] cv2.face.LBPHFaceRecognizer_create")
    except Exception as exc:
        failed.append("cv2.face")
        print(f"[FAIL] cv2.face: {exc}")

if failed:
    print("\nMissing/broken dependencies:", ", ".join(failed))
    print("Run setup_windows.bat or install requirements.txt inside the SAME virtual environment used to start app.py.")
    raise SystemExit(1)

print("\nAll imports passed. The ECAPA model weights are downloaded on first voice enrollment/verification and require internet access once.")
