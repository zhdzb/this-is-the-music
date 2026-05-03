"""
Make a short "guichu" style flute melody from a tiny sample.

Install dependencies:
    pip install librosa soundfile numpy imageio-ffmpeg

Optional, for extracting audio from mp4/mov files:
    Install ffmpeg and make sure the `ffmpeg` command is available,
    or install the Python package `imageio-ffmpeg`.

Run:
    python make_flute_melody.py

How to tune BASE_NOTE:
    BASE_NOTE means the real pitch of your original sample.
    If the generated result sounds too high overall, set BASE_NOTE higher, e.g. C4 -> D4.
    If the generated result sounds too low overall, set BASE_NOTE lower, e.g. C4 -> A3.

How to edit melody:
    Each item is ("NOTE", seconds), for example ("C4", 0.22).
    Use "REST" for silence.
    If you have numbered notation in C major:
        1 2 3 4 5 6 7 = C D E F G A B
    You can call jianpu_to_melody("1 2 3 5 3 2 1", duration=0.22, octave=4).
"""

from __future__ import annotations

import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


# =========================
# User settings
# =========================

SAMPLE_PATH = "flute_sample.wav"
OUTPUT_PATH = "output_guanyu_style.wav"

# The detected/original pitch of the input sample. Adjust this by ear.
BASE_NOTE = "C4"

TOTAL_DURATION = 30.0
FIT_FULL_MELODY = True
SR = 44100

# This script uses seconds directly by default. BPM is kept here in case you
# want to convert beats to seconds with beat_to_seconds().
BPM = 136

# Silence inserted after every non-rest note.
GAP_SECONDS = 0.018

# Whole melody transposition in semitones. 0 = unchanged, 12 = up one octave.
TRANSPOSE = 0

# "fast" is recommended for guichu-style chopped samples.
# Use "librosa" if you prefer slower, higher-quality pitch shifting.
PITCH_SHIFT_MODE = "fast"

# Small note-to-note gain randomness. Set to 0.0 for fully uniform notes.
RANDOM_VOLUME_RANGE = 0.08

# Short fade avoids pops/clicks at note boundaries.
FADE_SECONDS = 0.008
LONG_NOTE_FADE_SECONDS = 0.025

# Crossfade loop padding makes long sustained notes less choppy.
LOOP_CROSSFADE_SECONDS = 0.10

# Trim leading/trailing silence from the input sample.
TRIM_TOP_DB = 28

# If SAMPLE_PATH does not exist, try video/audio files in this folder.
BASE_FOLDER = "base"

# Video extensions that will be extracted with ffmpeg if possible.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# Optional click track for video alignment.
ENABLE_CLICK_TRACK = False
CLICK_GAIN = 0.18
CLICK_INTERVAL_SECONDS = 0.5


melody = [
    # fa sou la dou 【shi la】 sou fa
    ("F#4", 0.50),
    ("G#4", 0.50),
    ("A4", 0.50),
    ("C#4", 0.50),
    ("B4", 0.25),
    ("A4", 0.25),
    ("G#4", 0.50),
    ("F#4", 0.50),

    # 高八度fa dou dou 高八度mi 【ruai dou】
    ("F#5", 0.50),
    ("C#4", 1.00),   # dou dou 合并延长
    ("E5", 0.50),
    ("D4", 0.25),
    ("C#4", 0.25),

    # shi shi shi shi, shi dou ruai 高八度fa 【高八度mi ruai】
    ("B4", 2.00),    # shi shi shi shi 合并延长
    ("B4", 0.50),
    ("C#4", 0.50),
    ("D4", 0.50),
    ("F#5", 0.50),
    ("E5", 0.25),
    ("D4", 0.25),

    # dou shi dou fa shi la [sou fa] sou sou sou sou
    ("C#4", 0.50),
    ("B4", 0.50),
    ("C#4", 0.50),
    ("F#4", 0.50),
    ("B4", 0.50),
    ("A4", 0.50),
    ("G#4", 0.25),
    ("F#4", 0.25),
    ("G#4", 2.00),   # sou sou sou sou 合并延长

    # fa sou la 低ruai ruai dou shi
    ("F#4", 0.50),
    ("G#4", 0.50),
    ("A4", 0.50),
    ("D3", 0.50),    # 低ruai
    ("D4", 0.50),    # ruai
    ("C#4", 0.50),
    ("B4", 0.50),

    # shi 高八度还原sou 高八度fa 高八度mi 高八度fa
    ("B4", 0.50),
    ("G5", 0.50),    # 高八度还原sou
    ("F#5", 0.50),
    ("E5", 0.50),
    ("F#5", 0.50),

    # shi 高八度mi sou mi fa 还原sou la shi 还原dou ruai 高八度mi 升ruai--
    ("B4", 0.50),
    ("E5", 0.50),
    ("G#4", 0.50),
    ("E4", 0.50),
    ("F#4", 0.50),
    ("G4", 0.50),    # 还原sou
    ("A4", 0.50),
    ("B4", 0.50),
    ("C4", 0.50),    # 还原dou
    ("D4", 0.50),
    ("E5", 0.50),
    ("D#4", 1.50),   # 升ruai--

    # [dou shi] 高八度fa------
    ("C#4", 0.25),
    ("B4", 0.25),
    ("F#5", 2.50),   # 高八度fa------
]


# =========================
# Helpers
# =========================

NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}

JIANPU_TO_NOTE = {
    "1": "C",
    "2": "D",
    "3": "E",
    "4": "F",
    "5": "G",
    "6": "A",
    "7": "B",
    "0": "REST",
    "-": "REST",
}


def beat_to_seconds(beats: float, bpm: float = BPM) -> float:
    return 60.0 * beats / bpm


def note_to_midi(note: str) -> int:
    """Convert note name such as C4, D#4, Bb3 to MIDI number."""
    note = note.strip().upper()
    if note == "REST":
        raise ValueError("REST has no MIDI pitch")

    if len(note) < 2:
        raise ValueError(f"Invalid note name: {note!r}")

    if len(note) >= 3 and note[1] in {"#", "B"}:
        name = note[:2]
        octave_text = note[2:]
    else:
        name = note[:1]
        octave_text = note[1:]

    if name not in NOTE_TO_SEMITONE:
        raise ValueError(f"Invalid note name: {note!r}")

    try:
        octave = int(octave_text)
    except ValueError as exc:
        raise ValueError(f"Invalid octave in note name: {note!r}") from exc

    return 12 * (octave + 1) + NOTE_TO_SEMITONE[name]


def semitone_difference(base_note: str, target_note: str) -> int:
    return note_to_midi(target_note) - note_to_midi(base_note) + TRANSPOSE


def jianpu_to_melody(
    text: str,
    duration: float = 0.22,
    octave: int = 4,
    rest_duration: float | None = None,
) -> list[tuple[str, float]]:
    """
    Convert numbered notation to a melody list in C major.

    Example:
        jianpu_to_melody("1 2 3 5 3 2 1 0", duration=0.22, octave=4)

    Supported tokens:
        1..7 = notes in C major
        0 or - = rest
        +1 = one octave higher C
        -1 = one octave lower C
        1:0.44 = custom duration
    """
    result: list[tuple[str, float]] = []
    rest_duration = duration if rest_duration is None else rest_duration

    for raw_token in text.replace(",", " ").split():
        token = raw_token.strip()
        if not token:
            continue

        token_duration = duration
        if ":" in token:
            token, duration_text = token.split(":", 1)
            token_duration = float(duration_text)

        octave_offset = 0
        while token.startswith("+"):
            octave_offset += 1
            token = token[1:]
        while len(token) > 1 and token.startswith("-"):
            octave_offset -= 1
            token = token[1:]

        if token not in JIANPU_TO_NOTE:
            raise ValueError(f"Invalid jianpu token: {raw_token!r}")

        note_name = JIANPU_TO_NOTE[token]
        if note_name == "REST":
            result.append(("REST", rest_duration if ":" not in raw_token else token_duration))
        else:
            result.append((f"{note_name}{octave + octave_offset}", token_duration))

    return result


def find_fallback_sample() -> Path | None:
    folder = Path(BASE_FOLDER)
    if not folder.exists():
        return None

    candidates = [
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a", *VIDEO_EXTENSIONS}
    ]
    if not candidates:
        return None

    preferred_words = ("flute", "dizi", "笛", "笛子")
    for candidate in candidates:
        if any(word in candidate.stem.lower() for word in preferred_words):
            return candidate

    return sorted(candidates, key=lambda p: p.name)[0]


def get_ffmpeg_command() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    if imageio_ffmpeg is not None:
        return imageio_ffmpeg.get_ffmpeg_exe()
    return None


def extract_audio_with_ffmpeg(input_path: Path, sr: int) -> Path:
    ffmpeg = get_ffmpeg_command()
    if ffmpeg is None:
        raise RuntimeError(
            "This input is a video file, but ffmpeg was not found. "
            "Install ffmpeg, run `pip install imageio-ffmpeg`, or export the sample as WAV first."
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="flute_sample_"))
    output_path = temp_dir / "extracted_audio.wav"
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def resolve_sample_path() -> Path:
    sample_path = Path(SAMPLE_PATH)
    if sample_path.exists():
        return sample_path

    fallback = find_fallback_sample()
    if fallback is not None:
        print(f"SAMPLE_PATH not found, using fallback sample: {fallback}")
        return fallback

    raise FileNotFoundError(
        f"Could not find {SAMPLE_PATH!r}, and no usable sample was found in {BASE_FOLDER!r}."
    )


def load_audio_file(path: Path, sr: int) -> np.ndarray:
    audio, source_sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if int(source_sr) != sr:
        audio = librosa.resample(audio, orig_sr=int(source_sr), target_sr=sr)
    return np.asarray(audio, dtype=np.float32)


def trim_silence_fast(audio: np.ndarray, top_db: float = TRIM_TOP_DB) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-8:
        return audio

    threshold = peak * (10.0 ** (-top_db / 20.0))
    indices = np.flatnonzero(np.abs(audio) > threshold)
    if indices.size == 0:
        return audio

    pad = int(round(0.015 * SR))
    start = max(0, int(indices[0]) - pad)
    end = min(audio.size, int(indices[-1]) + pad)
    return audio[start:end]


def load_sample(path: Path, sr: int) -> np.ndarray:
    input_path = path
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        input_path = extract_audio_with_ffmpeg(path, sr)
        print(f"Extracted audio from video: {path}", flush=True)

    try:
        audio = load_audio_file(input_path, sr)
    except Exception:
        audio, _ = librosa.load(str(input_path), sr=sr, mono=True)
        audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Input audio is empty.")

    trimmed = trim_silence_fast(audio, top_db=TRIM_TOP_DB)
    if trimmed.size > 0:
        audio = trimmed

    audio = remove_dc(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-6:
        raise ValueError("Input audio is silent after trimming.")

    return (audio / peak * 0.9).astype(np.float32)


def remove_dc(audio: np.ndarray) -> np.ndarray:
    return (audio - np.mean(audio)).astype(np.float32)


def fit_to_duration(audio: np.ndarray, duration_seconds: float, sr: int) -> np.ndarray:
    target_len = max(1, int(round(duration_seconds * sr)))
    return fit_to_length(audio, target_len)


def fit_to_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    target_len = max(1, int(target_len))
    if audio.size == target_len:
        return audio.astype(np.float32)
    if audio.size > target_len:
        return audio[:target_len].astype(np.float32)

    repeats = math.ceil(target_len / max(1, audio.size))
    tiled = np.tile(audio, repeats)
    return tiled[:target_len].astype(np.float32)


def loop_with_crossfade(audio: np.ndarray, target_len: int, sr: int) -> np.ndarray:
    """Extend a short sample by looping it with crossfaded joins."""
    target_len = max(1, int(target_len))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size >= target_len:
        return audio[:target_len].astype(np.float32)
    if audio.size < 8:
        return fit_to_length(audio, target_len)

    fade_len = int(round(LOOP_CROSSFADE_SECONDS * sr))
    fade_len = min(fade_len, audio.size // 3, target_len // 4)
    if fade_len <= 8:
        return fit_to_length(audio, target_len)

    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = 1.0 - fade_in
    result = audio.copy()

    while result.size < target_len:
        piece = audio.copy()
        overlap = min(fade_len, result.size, piece.size)
        joined = np.empty(result.size + piece.size - overlap, dtype=np.float32)
        joined[: result.size - overlap] = result[: result.size - overlap]
        joined[result.size - overlap : result.size] = (
            result[-overlap:] * fade_out[-overlap:] + piece[:overlap] * fade_in[-overlap:]
        )
        joined[result.size:] = piece[overlap:]
        result = joined

    return result[:target_len].astype(np.float32)


def apply_fade(audio: np.ndarray, sr: int, fade_seconds: float = FADE_SECONDS) -> np.ndarray:
    if audio.size == 0:
        return audio

    fade_len = int(round(fade_seconds * sr))
    fade_len = min(fade_len, audio.size // 2)
    if fade_len <= 1:
        return audio.astype(np.float32)

    faded = audio.astype(np.float32).copy()
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    faded[:fade_len] *= fade_in
    faded[-fade_len:] *= fade_out
    return faded


def resample_linear(audio: np.ndarray, new_len: int) -> np.ndarray:
    new_len = max(1, int(new_len))
    if audio.size == new_len:
        return audio.astype(np.float32)
    if audio.size <= 1:
        return np.full(new_len, audio[0] if audio.size else 0.0, dtype=np.float32)

    old_positions = np.linspace(0.0, 1.0, audio.size, dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, new_len, dtype=np.float32)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def pitch_shift_fast(sample: np.ndarray, steps: int, duration: float, sr: int) -> np.ndarray:
    target_len = max(1, int(round(duration * sr)))
    factor = 2.0 ** (steps / 12.0)
    source_len = max(1, int(math.ceil(target_len * factor)))
    source = loop_with_crossfade(sample, source_len, sr)
    shifted = resample_linear(source, target_len)
    return shifted.astype(np.float32)


def make_note(
    sample: np.ndarray,
    target_note: str,
    duration: float,
    sr: int,
    note_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> np.ndarray:
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration!r}")

    if target_note.strip().upper() == "REST":
        return np.zeros(int(round(duration * sr)), dtype=np.float32)

    steps = semitone_difference(BASE_NOTE, target_note)
    target_len = max(1, int(round(duration * sr)))
    cache_key = (target_note.strip().upper(), target_len)

    # Give pitch_shift enough source material, then trim/pad after shifting.
    if note_cache is not None and cache_key in note_cache:
        note = note_cache[cache_key].copy()
    else:
        if PITCH_SHIFT_MODE.lower() == "librosa":
            work = fit_to_duration(sample, max(duration, 0.5), sr)
            shifted = librosa.effects.pitch_shift(work, sr=sr, n_steps=steps)
            note = fit_to_duration(shifted, duration, sr)
        else:
            note = pitch_shift_fast(sample, steps, duration, sr)
        fade_seconds = LONG_NOTE_FADE_SECONDS if duration >= 0.75 else FADE_SECONDS
        note = apply_fade(note, sr, fade_seconds=fade_seconds)
        if note_cache is not None:
            note_cache[cache_key] = note.copy()

    if RANDOM_VOLUME_RANGE > 0:
        gain = random.uniform(1.0 - RANDOM_VOLUME_RANGE, 1.0 + RANDOM_VOLUME_RANGE)
        note = note * gain

    return note.astype(np.float32)


def validate_melody(items: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    checked: list[tuple[str, float]] = []
    for index, item in enumerate(items):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Melody item #{index} must be a tuple like ('C4', 0.22).")

        note, duration = item
        if not isinstance(note, str):
            raise ValueError(f"Melody item #{index} has invalid note: {note!r}")

        try:
            duration = float(duration)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Melody item #{index} has invalid duration: {duration!r}") from exc

        if duration <= 0:
            raise ValueError(f"Melody item #{index} duration must be positive.")

        if note.strip().upper() != "REST":
            note_to_midi(note)

        checked.append((note.strip().upper(), duration))

    if not checked:
        raise ValueError("Melody is empty.")

    return checked


def make_click_track(length: int, sr: int) -> np.ndarray:
    click = np.zeros(length, dtype=np.float32)
    interval = max(1, int(round(CLICK_INTERVAL_SECONDS * sr)))
    click_len = int(round(0.018 * sr))
    tone_t = np.arange(click_len, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 1600 * tone_t).astype(np.float32)
    tone *= np.linspace(1.0, 0.0, click_len, dtype=np.float32)

    for start in range(0, length, interval):
        end = min(length, start + click_len)
        click[start:end] += tone[: end - start] * CLICK_GAIN

    return click


def build_melody_audio(sample: np.ndarray, melody_items: list[tuple[str, float]], sr: int) -> np.ndarray:
    melody_duration = sum(duration for _, duration in melody_items)
    target_duration = max(TOTAL_DURATION, melody_duration) if FIT_FULL_MELODY else TOTAL_DURATION
    target_len = int(round(target_duration * sr))
    gap = np.zeros(int(round(GAP_SECONDS * sr)), dtype=np.float32)
    chunks: list[np.ndarray] = []
    current_len = 0
    melody_index = 0
    note_cache: dict[tuple[str, int], np.ndarray] = {}

    while current_len < target_len:
        note, duration = melody_items[melody_index % len(melody_items)]
        note_audio = make_note(sample, note, duration, sr, note_cache)
        chunks.append(note_audio)
        current_len += note_audio.size

        if note != "REST" and GAP_SECONDS > 0 and current_len < target_len:
            chunks.append(gap)
            current_len += gap.size

        melody_index += 1

    output = np.concatenate(chunks) if chunks else np.zeros(target_len, dtype=np.float32)
    output = output[:target_len]

    if ENABLE_CLICK_TRACK:
        output = output + make_click_track(output.size, sr)

    return normalize(output)


def normalize(audio: np.ndarray, peak_target: float = 0.92) -> np.ndarray:
    audio = remove_dc(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-8:
        return audio.astype(np.float32)
    return (audio / peak * peak_target).astype(np.float32)


def main() -> None:
    random.seed()

    sample_path = resolve_sample_path()
    melody_items = validate_melody(melody)

    print(f"Loading sample: {sample_path}")
    sample = load_sample(sample_path, SR)

    melody_duration = sum(duration for _, duration in melody_items)
    target_duration = max(TOTAL_DURATION, melody_duration) if FIT_FULL_MELODY else TOTAL_DURATION
    print(f"Generating about {target_duration:.1f}s of audio...")
    output = build_melody_audio(sample, melody_items, SR)

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    sf.write(str(output_path), output, SR, subtype="PCM_16")

    print(f"Done: {output_path.resolve()}")
    print(f"Sample rate: {SR} Hz, duration: {output.size / SR:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
