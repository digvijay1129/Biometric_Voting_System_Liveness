"""Conservative speaker verification for the two-step biometric demo.

Sentence recognition only verifies the words that were spoken. This module uses
SpeechBrain's pretrained ECAPA-TDNN speaker model to compare vocal identity.
It deliberately fails closed when audio quality, enrollment consistency, model
availability, or speaker similarity is uncertain.

Educational prototype only: no threshold guarantees perfect recognition in every
room, microphone, language, or speaker population. Never use this demo for real
voting or other high-stakes identity decisions.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import threading
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

VOICE_SAMPLE_RATE = 16000
VOICE_CHALLENGE_TEXT = (
    "I am a person, I am speaking clear "
    "to prove that this is my voice."
)
VOICE_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
VOICE_MODEL_ID = "speechbrain-ecapa-voxceleb-192-v1"
EXPECTED_EMBEDDING_SIZE = 192
VOICE_ENROLLMENT_SAMPLES = 3

# Conservative starting cutoffs, intentionally favoring rejection over a false
# accept. They still need validation against recordings from the intended users.
VOICE_SIMILARITY_THRESHOLD = 0.55          # mean similarity to all 3 references
VOICE_REFERENCE_MIN_THRESHOLD = 0.45        # EACH reference must clear this
VOICE_DIFFERENT_SPEAKER_MARGIN = 0.12       # target vs closest other enrolled user
ENROLLMENT_PAIR_MIN_SIMILARITY = 0.40       # consistency of all 3 enrollment takes
ENROLLMENT_MEAN_SIMILARITY = 0.50

MIN_VOICE_SECONDS = 3.0
MAX_VOICE_SECONDS = 15.0
# Microphone gain varies widely across laptops and browsers. Use lower base
# levels and an adaptive speech-frame threshold instead of a fixed 0.008 RMS
# gate, which falsely rejects valid quiet speech on many built-in microphones.
MIN_PEAK_AMPLITUDE = 0.008
MIN_RMS_AMPLITUDE = 0.0015
MIN_ACTIVE_FRAME_RATIO = 0.10
MAX_CLIPPED_SAMPLE_RATIO = 0.03

_encoder = None
_encoder_lock = threading.Lock()
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "speaker_ecapa"


def normalize_text(value: str) -> str:
    """Normalize punctuation/case so ASR punctuation differences are ignored."""
    value = (value or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return " ".join(value.split())


def transcript_similarity(transcript: str, expected: str = VOICE_CHALLENGE_TEXT) -> float:
    """Word-level similarity for UI diagnostics; never used as speaker evidence."""
    from difflib import SequenceMatcher
    spoken = normalize_text(transcript)
    wanted = normalize_text(expected)
    if not spoken or not wanted:
        return 0.0
    return SequenceMatcher(None, spoken, wanted).ratio()


def transcript_matches(transcript: str, expected: str = VOICE_CHALLENGE_TEXT) -> bool:
    """Require the full challenge sentence after case/punctuation normalization."""
    return normalize_text(transcript) == normalize_text(expected)


def decode_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    """Decode uncompressed 16-bit PCM WAV into mono float32 at 16 kHz."""
    if not wav_bytes:
        raise ValueError("No audio data was received. Please record your voice again.")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("Audio must be an uncompressed PCM WAV recording.")
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError("The microphone recording is not a valid WAV file.") from exc

    if sample_width != 2:
        raise ValueError("The recording must use 16-bit PCM audio. Try Chrome or Edge.")
    if channels not in (1, 2):
        raise ValueError("The recording must be mono or stereo audio.")
    if not 8000 <= sample_rate <= 48000:
        raise ValueError("The recording sample rate is unsupported.")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels == 2:
        if samples.size % 2:
            samples = samples[:-1]
        samples = samples.reshape(-1, 2).mean(axis=1)
    if samples.size == 0:
        raise ValueError("The recording is empty. Please record your voice again.")

    if sample_rate != VOICE_SAMPLE_RATE:
        new_count = max(1, int(round(samples.size * VOICE_SAMPLE_RATE / sample_rate)))
        old_positions = np.arange(samples.size, dtype=np.float64) / sample_rate
        new_positions = np.arange(new_count, dtype=np.float64) / VOICE_SAMPLE_RATE
        samples = np.interp(new_positions, old_positions, samples).astype(np.float32)

    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def encode_wav_bytes(samples: np.ndarray, sample_rate: int = VOICE_SAMPLE_RATE) -> bytes:
    """Encode mono float samples as canonical 16-bit PCM WAV."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    pcm = np.round(np.clip(samples, -1.0, 0.999969) * 32768.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


def _get_encoder():
    """Load SpeechBrain's pretrained ECAPA model exactly once; no weak fallback."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is not None:
            return _encoder
        # This warning is about Hugging Face's own cache optimization. It is not
        # fatal on Windows; SpeechBrain's separate savedir copies below explicitly
        # use LocalStrategy.COPY so Windows does not need symlink privileges.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            import torch
            from speechbrain.inference.speaker import SpeakerRecognition
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as exc:
            raise RuntimeError(
                "The ECAPA speaker model dependencies are missing. Run "
                "setup_windows.bat or install requirements.txt in the project's "
                "Python 3.10 environment, then restart Flask."
            ) from exc

        try:
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # The first use downloads the pretrained model from Hugging Face. Later
            # runs use this local cache. CPU mode avoids requiring a CUDA GPU.
            _encoder = SpeakerRecognition.from_hparams(
                source=VOICE_MODEL_SOURCE,
                savedir=str(MODEL_CACHE_DIR),
                run_opts={"device": "cpu"},
                # Important on Windows: the SpeechBrain default is SYMLINK,
                # which raises WinError 1314 unless Developer Mode/admin rights
                # are enabled. COPY uses ordinary files instead.
                local_strategy=LocalStrategy.COPY,
            )
            _encoder.eval()
        except Exception as exc:
            _encoder = None
            # A previous build may have left an incomplete model directory after
            # failing to create a symlink. Clear only that model's local cache when
            # Windows reports the specific symlink privilege error; Hugging Face's
            # download cache is separate and is retained.
            if getattr(exc, "winerror", None) == 1314:
                shutil.rmtree(MODEL_CACHE_DIR, ignore_errors=True)
                raise RuntimeError(
                    "Windows blocked symbolic-link creation while loading ECAPA. "
                    "This project now requests copy mode; the incomplete local model "
                    "folder was cleared. Restart the app and try again with internet access. "
                    "You do not need to run VS Code as Administrator or enable Developer Mode."
                ) from exc
            raise RuntimeError(
                "The pretrained ECAPA voice model could not be loaded. On the first "
                "run, check internet access and retry. If the model cache is damaged, "
                "delete models/speaker_ecapa and start the app again."
            ) from exc
    return _encoder


def _check_audio_quality(samples: np.ndarray) -> None:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    duration = samples.size / float(VOICE_SAMPLE_RATE)
    if duration < MIN_VOICE_SECONDS:
        raise ValueError("The recording is too short. Read the entire sentence naturally.")
    if duration > MAX_VOICE_SECONDS:
        raise ValueError("The recording is too long. Record only the displayed sentence.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("The recording contains invalid audio values. Please record again.")

    absolute = np.abs(samples)
    peak = float(np.max(absolute)) if absolute.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)) + 1e-12)) if absolute.size else 0.0
    if peak < MIN_PEAK_AMPLITUDE or rms < MIN_RMS_AMPLITUDE:
        raise ValueError("The audio is too quiet. Move closer to the microphone and try again.")
    clipped_ratio = float(np.mean(absolute >= 0.995))
    if clipped_ratio > MAX_CLIPPED_SAMPLE_RATIO:
        raise ValueError("The recording is distorted or clipping. Lower the input volume and retry.")

    # Reject recordings that are mostly silence without assuming a fixed mic
    # amplitude. The old fixed threshold (frame RMS >= 0.008) was too high for
    # many laptop/headset microphones: the transcript could be correct while
    # almost every valid speech frame failed this gate.
    frame_size = int(0.020 * VOICE_SAMPLE_RATE)
    usable = samples[: (samples.size // frame_size) * frame_size]
    if usable.size:
        frame_rms = np.sqrt(np.mean(usable.reshape(-1, frame_size) ** 2, axis=1) + 1e-12)
        noise_floor = float(np.percentile(frame_rms, 20))
        speech_level = float(np.percentile(frame_rms, 90))

        # Keep the gate above the estimated background floor, but adapt it to
        # the recording's own speech level. This avoids rejecting quiet speech
        # just because its frame RMS never reaches the old 0.008 constant.
        active_threshold = max(0.001, noise_floor * 2.0, speech_level * 0.12)
        active_ratio = float(np.mean(frame_rms >= active_threshold))
        dynamic_range = speech_level / max(noise_floor, 1e-5)

        # A transcript is independently required to equal the challenge text;
        # here we only screen out near-silence/steady background noise.
        if active_ratio < MIN_ACTIVE_FRAME_RATIO or dynamic_range < 1.25:
            raise ValueError(
                "Too little clear speech was detected. The microphone signal is too quiet "
                "or contains mostly background noise. Move closer to the microphone, "
                "enable microphone access, speak the full sentence naturally, and try again."
            )


def voice_embedding(samples: np.ndarray) -> np.ndarray:
    """Return an L2-normalized 192-D ECAPA-TDNN speaker embedding."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    _check_audio_quality(samples)
    try:
        import torch
        encoder = _get_encoder()
        audio_tensor = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
        with torch.inference_mode():
            encoded = encoder.encode_batch(audio_tensor)
        vector = np.asarray(encoded.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError("The pretrained ECAPA model could not process this recording.") from exc

    if vector.size != EXPECTED_EMBEDDING_SIZE or not np.all(np.isfinite(vector)):
        raise RuntimeError("The speaker model returned an invalid embedding. Reinstall its dependencies.")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("No usable speaker features were detected. Please record again.")
    return (vector / norm).astype(np.float32)


def _normalize_embedding_matrix(embeddings: Iterable[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(list(embeddings), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape != (VOICE_ENROLLMENT_SAMPLES, EXPECTED_EMBEDDING_SIZE):
        raise ValueError(f"Exactly {VOICE_ENROLLMENT_SAMPLES} valid speaker samples are required.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < 1e-8) or not np.all(np.isfinite(matrix)):
        raise ValueError("A reference voice sample was invalid. Please enroll the voter again.")
    return matrix / norms


def validate_enrollment_embeddings(embeddings: Iterable[np.ndarray]) -> np.ndarray:
    """Reject an enrollment set whose recordings do not look like one speaker."""
    matrix = _normalize_embedding_matrix(embeddings)
    pairwise = matrix @ matrix.T
    off_diagonal = pairwise[np.triu_indices(VOICE_ENROLLMENT_SAMPLES, k=1)]
    min_pair = float(np.min(off_diagonal))
    mean_pair = float(np.mean(off_diagonal))
    if min_pair < ENROLLMENT_PAIR_MIN_SIMILARITY or mean_pair < ENROLLMENT_MEAN_SIMILARITY:
        raise ValueError(
            "The three recordings are not consistent enough to confirm one speaker. "
            "Record all three samples again with the same person's voice, using a quiet room and the same microphone."
        )
    return matrix.astype(np.float32)


def save_voice_profile(path: str | Path, embeddings: Iterable[np.ndarray]) -> None:
    """Save three validated embeddings, model ID, and metadata without pickle."""
    matrix = validate_enrollment_embeddings(embeddings)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as profile_file:
        np.savez_compressed(
            profile_file,
            embeddings=matrix,
            model_id=np.asarray(VOICE_MODEL_ID),
            sample_count=np.asarray(VOICE_ENROLLMENT_SAMPLES, dtype=np.int32),
        )
    temp_path.replace(path)


def load_voice_profile(path: str | Path) -> np.ndarray:
    """Load and strictly validate a profile created by this model version."""
    try:
        with np.load(Path(path), allow_pickle=False) as profile:
            model_id = str(profile["model_id"].item())
            embeddings = np.asarray(profile["embeddings"], dtype=np.float32)
            sample_count = int(profile["sample_count"].item())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("The saved speaker profile is missing or damaged. Enroll this voter again.") from exc

    if model_id != VOICE_MODEL_ID:
        raise ValueError("This profile uses an older voice model. Please enroll this voter again.")
    if sample_count != VOICE_ENROLLMENT_SAMPLES:
        raise ValueError("The saved profile is incomplete. Re-enroll this voter with all three samples.")
    if embeddings.ndim != 2 or embeddings.shape != (VOICE_ENROLLMENT_SAMPLES, EXPECTED_EMBEDDING_SIZE):
        raise ValueError("This profile uses an older or incompatible voice model. Open Voice Enrollment and record all three samples again.")
    matrix = _normalize_embedding_matrix(embeddings)
    # Validate stored profiles again in case the file was copied/altered.
    pairwise = matrix @ matrix.T
    off_diagonal = pairwise[np.triu_indices(VOICE_ENROLLMENT_SAMPLES, k=1)]
    if float(np.min(off_diagonal)) < ENROLLMENT_PAIR_MIN_SIMILARITY or float(np.mean(off_diagonal)) < ENROLLMENT_MEAN_SIMILARITY:
        raise ValueError("The stored voice references are inconsistent. Re-enroll this voter.")
    return matrix


def profile_similarity_scores(reference_embeddings: np.ndarray, live_embedding: np.ndarray) -> np.ndarray:
    """Return live-vs-each-reference cosine scores; no best-of-N shortcut."""
    references = np.asarray(reference_embeddings, dtype=np.float32)
    live = np.asarray(live_embedding, dtype=np.float32).reshape(-1)
    if references.ndim != 2 or references.shape[1] != EXPECTED_EMBEDDING_SIZE or live.size != EXPECTED_EMBEDDING_SIZE:
        raise ValueError("The stored profile or live speaker embedding has an invalid shape.")
    live_norm = float(np.linalg.norm(live))
    ref_norms = np.linalg.norm(references, axis=1)
    if live_norm < 1e-8 or np.any(ref_norms < 1e-8) or not np.all(np.isfinite(references)) or not np.all(np.isfinite(live)):
        raise ValueError("Could not compare the voice profile. Please re-enroll and retry.")
    scores = (references @ (live / live_norm)) / ref_norms
    return np.clip(scores.astype(np.float32), -1.0, 1.0)


def profile_similarity(reference_embeddings: np.ndarray, live_embedding: np.ndarray) -> float:
    """Mean cosine similarity across ALL references, rather than best references."""
    scores = profile_similarity_scores(reference_embeddings, live_embedding)
    return float(np.mean(scores))
