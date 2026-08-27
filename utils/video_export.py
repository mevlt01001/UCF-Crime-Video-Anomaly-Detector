"""Shared FFmpeg export and output verification for clip tools."""
import math
import subprocess
from pathlib import Path


class VideoExportError(ValueError):
    code = "INVALID_OUTPUT"


def export_video(video_path, output_path, start, end, *, exact=False):
    import cv2

    output_path = Path(output_path)
    if output_path.resolve() == Path(video_path).resolve():
        raise VideoExportError("Kaynak video kendi üzerine kaydedilemez.")
    if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
        raise VideoExportError("Geçersiz kesit aralığı.")
    command = ["ffmpeg", "-nostdin", "-n" if exact else "-y",
               "-ss", str(start), "-i", str(video_path), "-t", str(end - start)]
    if exact:
        command += ["-map", "0:v:0", "-map", "0:a?", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac",
                    "-movflags", "+faststart"]
    else:
        command += ["-c:v", "copy", "-c:a", "copy"]
    subprocess.run(command + [str(output_path)], check=True, capture_output=True,
                   text=True, timeout=300)
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise VideoExportError("FFmpeg tamamlandı ancak geçerli boyutta dosya oluşmadı.")
    capture = cv2.VideoCapture(str(output_path))
    try:
        opened = capture.isOpened()
        readable, _ = capture.read() if opened else (False, None)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        capture.release()
    if not readable:
        raise VideoExportError("Çıktı geçerli bir video olarak okunamadı.")
    if exact:
        duration = frames / fps if fps > 0 else float("nan")
        tolerance = max(0.15, 2 / fps) if fps > 0 else 0
        if not math.isfinite(duration) or abs(duration - (end - start)) > tolerance:
            raise VideoExportError("Arşiv klibinin süresi istenen aralıkla uyuşmuyor.")
