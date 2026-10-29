"""Fast unit tests for voice-profile logic; do not download/load the neural model."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import voice_biometrics as vb  # noqa: E402


class VoiceBiometricsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1201)
        base = rng.normal(size=vb.EXPECTED_EMBEDDING_SIZE).astype(np.float32)
        base /= np.linalg.norm(base)
        self.base = base
        self.same_speaker = []
        for _ in range(3):
            sample = base + 0.025 * rng.normal(size=base.shape).astype(np.float32)
            sample /= np.linalg.norm(sample)
            self.same_speaker.append(sample)
        other = rng.normal(size=vb.EXPECTED_EMBEDDING_SIZE).astype(np.float32)
        other /= np.linalg.norm(other)
        self.other = other

    def test_sentence_check_is_exact_after_normalization(self):
        self.assertTrue(vb.transcript_matches(vb.VOICE_CHALLENGE_TEXT.upper()))
        self.assertFalse(vb.transcript_matches("I confirm my identity"))

    def test_consistent_profile_saves_and_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice_profile.npz"
            vb.save_voice_profile(path, self.same_speaker)
            loaded = vb.load_voice_profile(path)
            self.assertEqual(loaded.shape, (3, vb.EXPECTED_EMBEDDING_SIZE))
            scores = vb.profile_similarity_scores(loaded, self.same_speaker[0])
            self.assertEqual(scores.shape, (3,))
            self.assertGreater(float(scores.min()), 0.85)

    def test_mixed_speaker_enrollment_is_rejected(self):
        with self.assertRaises(ValueError):
            vb.validate_enrollment_embeddings([self.same_speaker[0], self.same_speaker[1], self.other])

    def test_all_references_affect_score(self):
        clean_score = vb.profile_similarity(np.stack(self.same_speaker), self.same_speaker[0])
        inconsistent = np.stack([self.same_speaker[0], self.same_speaker[1], -self.base])
        weaker_score = vb.profile_similarity(inconsistent, self.same_speaker[0])
        self.assertLess(weaker_score, clean_score)

    def test_wav_encode_decode_round_trip(self):
        audio = np.sin(np.linspace(0, 10 * np.pi, vb.VOICE_SAMPLE_RATE * 4)).astype(np.float32) * 0.1
        wav = vb.encode_wav_bytes(audio)
        restored = vb.decode_wav_bytes(wav)
        self.assertEqual(restored.size, audio.size)
        self.assertTrue(np.all(np.isfinite(restored)))

    def test_quiet_but_clear_speech_is_not_rejected_by_fixed_energy_gate(self):
        # Simulate four seconds of quiet speech-like energy with pauses. Peak is
        # above the minimum, while each active 20 ms frame's RMS stays below the
        # old fixed 0.008 threshold that caused the user's repeated error.
        sr = vb.VOICE_SAMPLE_RATE
        frame_size = int(0.020 * sr)
        t = np.arange(frame_size, dtype=np.float32) / sr
        tone = 0.009 * np.sin(2 * np.pi * 180 * t).astype(np.float32)
        frames = []
        for i in range(int(4 / 0.020)):
            active = (i % 20) < 13
            frames.append(tone if active else np.zeros_like(tone))
        quiet_speech = np.concatenate(frames)
        vb._check_audio_quality(quiet_speech)

    def test_steady_background_tone_is_not_accepted_as_clear_speech(self):
        sr = vb.VOICE_SAMPLE_RATE
        t = np.arange(sr * 4, dtype=np.float32) / sr
        steady_tone = 0.009 * np.sin(2 * np.pi * 180 * t).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "Too little clear speech"):
            vb._check_audio_quality(steady_tone)


if __name__ == "__main__":
    unittest.main(verbosity=2)
