"""
Video Utilities

Post-processing helpers for browser-compatible video output.
"""

import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def reencode_for_browser(input_path: str) -> str:
    """
    Re-encode a video file to H.264 MP4 so it plays in every browser.

    OpenCV writes with mp4v (MPEG-4 Part 2) or XVID — neither plays
    natively in Chrome/Firefox/Safari.  This function calls ffmpeg to
    transcode to H.264 with the moov atom at the front (faststart),
    which is the only codec/container combination all major browsers
    support without plugins.

    Args:
        input_path: Path to the OpenCV-written video (any codec/container).

    Returns:
        Path to the browser-ready MP4 (replaces input if successful,
        otherwise returns the original path unchanged so playback still
        works in local players).
    """
    if not os.path.exists(input_path):
        return input_path

    base = os.path.splitext(input_path)[0]
    output_path = base + "_h264.mp4"

    try:
        cmd = [
            "ffmpeg",
            "-y",                        # overwrite without asking
            "-i", input_path,
            "-c:v", "libx264",           # H.264 video codec
            "-preset", "fast",           # fast encode, reasonable quality
            "-crf", "23",                # quality (18=best, 28=worst; 23 is default)
            "-c:a", "aac",               # AAC audio (safe default)
            "-movflags", "+faststart",   # move moov atom to front for streaming
            "-pix_fmt", "yuv420p",       # broadest browser compatibility
            output_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=300,  # 5-minute cap for long videos
        )

        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")[-500:]
            logger.warning(f"ffmpeg re-encode failed (rc={result.returncode}): {err}")
            return input_path  # fall back to original

        # Replace original with the H.264 version
        os.replace(output_path, input_path if input_path.endswith(".mp4") else base + ".mp4")
        final_path = input_path if input_path.endswith(".mp4") else base + ".mp4"

        logger.info(f"Re-encoded for browser: {final_path}")
        return final_path

    except FileNotFoundError:
        logger.warning("ffmpeg not found — video may not play in browser")
        return input_path
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg re-encode timed out")
        if os.path.exists(output_path):
            os.remove(output_path)
        return input_path
    except Exception as e:
        logger.warning(f"Re-encode error: {e}")
        return input_path
