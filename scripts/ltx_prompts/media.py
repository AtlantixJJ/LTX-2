"""Media conversion helpers for LTX captioning."""
import subprocess
from pathlib import Path

def _run_ffmpeg(args: list[str]) -> None:
    """Run the ffmpeg binary bundled with ``imageio-ffmpeg`` (a dependency)."""
    import imageio_ffmpeg  # noqa: PLC0415

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *args]
    subprocess.run(cmd, check=True, capture_output=True)


def _extract_audio_wav(src: Path, dest: Path) -> None:
    """Extract the audio track to a 16 kHz mono PCM WAV (matches pretraining).
    Raises ``CalledProcessError`` when the video has no audio stream.
    """
    _run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)])


def _transcode_cfr(src: Path, dest: Path) -> None:
    """Re-encode the video to a constant frame rate so the server's frame sampler can
    read every requested index (raw / variable-frame-rate videos over-report frames)."""
    _run_ffmpeg(["-i", str(src), "-fps_mode", "cfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(dest)])


