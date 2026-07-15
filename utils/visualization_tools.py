import numpy as np
import matplotlib.pyplot as plt

def plot_anomaly_timeline(scores, # [1000]
                          scores_org, # [1000]
                          segments,
                          video_seconds: float, 
                          threshold: float = 0.3, 
                          save_path: str = "anomaly_segmentation_plot.png"
                          ):
    
    time_axis = np.linspace(0, video_seconds, 1000)
    
    plt.figure(figsize=(15, 6), dpi=120)
    plt.plot(time_axis, scores_org, color="#b6b971", alpha=0.5, linewidth=1.5, label='Ham Anomali Skoru (Orijinal)')
    plt.plot(time_axis, scores, color='#1f77b4', linewidth=2.5, label='Anomali Skoru (Smoothed)')
    plt.axhline(y=threshold, color='black', linestyle='--', linewidth=1.5, label=f'Eşik Değeri ({threshold})')

    for idx, seg in enumerate(segments):
        start = seg["start_time"]
        end = seg["end_time"]
        
        label = 'Anomali Segment' if idx == 0 else None
        plt.axvspan(start, end, color='red', alpha=0.2, label=label)
        
        plt.text(start, 1.02, f"{start}s", color='darkred', fontsize=9, ha='center')
        plt.text(end, 1.02, f"{end}s", color='darkred', fontsize=9, ha='center')

    plt.title("Video Anomali Skoru", fontsize=14, fontweight='bold')
    plt.xlabel("Zaman (Saniye)", fontsize=12)
    plt.ylabel("Anomali Olasılığı", fontsize=12)
    plt.ylim(-0.05, 1.1)
    plt.xlim(0, video_seconds)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()