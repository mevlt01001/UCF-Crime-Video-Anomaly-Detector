from decord import VideoReader

def get_frame_id(video_path: str, number: int, unit: str) -> int:

    unit_map= {
        "saat": 3600,
        "dakika": 60,
        "saniye": 1
    }

    if not unit in unit_map:
        print("Kullanılan birim anlaşılamadı.")

    second = unit_map[unit]

    seconds = int(number * second)

    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()

    frame_id = int(round(seconds * fps))

    return frame_id

if __name__ == "__main__":
    result = get_frame_id(video_path= "videos/test_video.mp4", number= "5", unit="saniye")
    print(result)