"""Active liveness detection for the biometric voting prototype.

This module uses MediaPipe Face Mesh landmarks to detect a real blink.
It does not require any external electronics; the existing webcam is enough.
"""

from __future__ import annotations

import time
from statistics import median
from typing import Callable, Dict, Optional, Tuple

import cv2
import mediapipe as mp


# MediaPipe Face Mesh landmark indices for the eye contours.
# The points are ordered so EAR can be calculated from them.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

# Baseline/open-eye and blink detection parameters.
CALIBRATION_FRAMES = 12
CALIBRATION_TIMEOUT_SECONDS = 3.0
LIVENESS_TIMEOUT_SECONDS = 8.0
MIN_OPEN_FRAMES_BEFORE_BLINK = 2
CLOSED_FRAMES_REQUIRED = 2
REOPEN_FRAMES_REQUIRED = 2


def euclidean_distance(p1, p2) -> float:
    """Return the 2D distance between two MediaPipe landmarks."""
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


def eye_aspect_ratio(landmarks, indices: Tuple[int, int, int, int, int, int]) -> float:
    """Calculate Eye Aspect Ratio (EAR) from six eye landmarks."""
    p1, p2, p3, p4, p5, p6 = (landmarks[index] for index in indices)

    horizontal = euclidean_distance(p1, p4)
    if horizontal <= 1e-8:
        return 0.0

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_average_ear(face_landmarks) -> float:
    """Return the average EAR for both eyes."""
    landmarks = face_landmarks.landmark
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    return (right_ear + left_ear) / 2.0


def _result(
    success: bool,
    message: str,
    blink_detected: bool = False,
    reason: str = "",
) -> Dict[str, object]:
    return {
        "success": success,
        "message": message,
        "blink_detected": blink_detected,
        "reason": reason,
    }


def run_blink_liveness(
    read_frame: Callable[[], Tuple[bool, Optional[object]]],
    face_mesh,
    set_status: Optional[Callable[[str, str], None]] = None,
    timeout_seconds: float = LIVENESS_TIMEOUT_SECONDS,
) -> Dict[str, object]:
    """Run a short active liveness challenge based on one natural blink.

    The function first calibrates the person's open-eye EAR, then looks for a
    short drop in EAR followed by reopening. A static photograph normally has
    a stable EAR and therefore cannot complete the challenge.
    """

    def status(message: str, state: str = "checking") -> None:
        if set_status:
            set_status(message, state)

    status("Liveness check: look at the camera", "checking")

    calibration_values = []
    calibration_start = time.monotonic()

    while (
        len(calibration_values) < CALIBRATION_FRAMES
        and time.monotonic() - calibration_start < CALIBRATION_TIMEOUT_SECONDS
    ):
        ok, frame = read_frame()
        if not ok or frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            status("Liveness check: keep your face visible", "checking")
            continue

        ear = calculate_average_ear(result.multi_face_landmarks[0])
        if 0.10 < ear < 0.50:
            calibration_values.append(ear)
            status(
                f"Liveness check: calibrating ({len(calibration_values)}/{CALIBRATION_FRAMES})",
                "checking",
            )

    if len(calibration_values) < CALIBRATION_FRAMES:
        status("Liveness failed: face/eyes were not clear enough", "failed")
        return _result(
            False,
            "Liveness failed. Keep your face straight and ensure both eyes are visible.",
            reason="calibration_failed",
        )

    baseline_ear = median(calibration_values)
    # Adaptive threshold: the person's own open-eye ratio is the reference.
    # The floor prevents an unusually low baseline from making the threshold useless.
    closed_threshold = max(0.14, baseline_ear * 0.70)

    status("Liveness check: blink once", "checking")

    start = time.monotonic()
    had_open_eyes = False
    closed_frames = 0
    reopened_frames = 0

    while time.monotonic() - start < timeout_seconds:
        ok, frame = read_frame()
        if not ok or frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            status("Liveness check: face not visible", "checking")
            closed_frames = 0
            reopened_frames = 0
            continue

        # Only one face should participate in this biometric check.
        if len(result.multi_face_landmarks) > 1:
            status("Liveness check: only one person should be in the camera", "failed")
            return _result(
                False,
                "Liveness failed. Only one person should be visible in the camera.",
                reason="multiple_faces",
            )

        ear = calculate_average_ear(result.multi_face_landmarks[0])

        if ear >= closed_threshold:
            if not had_open_eyes:
                had_open_eyes = True
            elif closed_frames >= CLOSED_FRAMES_REQUIRED:
                reopened_frames += 1
                status("Liveness check: eyes reopening...", "checking")
                if reopened_frames >= REOPEN_FRAMES_REQUIRED:
                    status("Liveness verified", "passed")
                    return _result(
                        True,
                        "Liveness verified. Real person detected.",
                        blink_detected=True,
                        reason="blink_detected",
                    )
            else:
                reopened_frames = 0
        else:
            if had_open_eyes:
                closed_frames += 1
                reopened_frames = 0
                status("Liveness check: blink detected, open your eyes", "checking")

    status("Liveness failed: blink was not detected", "failed")
    return _result(
        False,
        "Liveness failed. Blink once naturally and try again.",
        reason="blink_not_detected",
    )


__all__ = [
    "run_blink_liveness",
    "eye_aspect_ratio",
    "calculate_average_ear",
]
