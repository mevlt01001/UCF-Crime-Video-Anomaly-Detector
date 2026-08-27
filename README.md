# UCF Crime Video Anomaly Detector

Güvenlik kamerası videolarındaki olağandışı zaman aralıklarını tespit eden ve bu aralıkları bir görsel-dil modeliyle (VLM) açıklayan video analiz projesidir.

## Sistem nasıl çalışır?

```text
Video
  ↓
Kısa ve örtüşen kliplere ayırma
  ↓
S3D ile görsel özellik çıkarımı
  ↓
SegmentRankingModel ile anomali skoru
  ↓
Eşik üstündeki zaman aralıklarını birleştirme
  ↓
Anomali aralıklarından VLM için kare örnekleme
  ↓
Olayı doğal dille açıklama
```

Anomali modeli olay türünü doğrudan sınıflandırmaz; normalden sapan zaman aralıklarını skorlar. İstenirse bu aralıklar VLM tarafından açıklanır veya ayrı bir video klibi olarak kaydedilir.

## Temel özellikler

- S3D tabanlı video özellik çıkarımı
- Normal/anormal segment skorlaması
- Anomali zaman çizelgesi ve grafik çıktısı
- Belirli zaman aralıklarında VLM soru-cevap
- Planner → Executor → Tools → Reviewer agent akışı
- Ana sohbet hafızası ile agent çalışma context'inin ayrılması
- Paralel VLM çağrıları için bağımsız context
- Kaynak video zamanını koruyan dinamik VLM encode FPS
- FFmpeg ile video kesiti kaydetme
- CUDA, Apple MPS ve CPU inference desteği

## Gereksinimler

- Python 3.10
- Python paketleri: `requirements.txt`
- Klip kaydetme özelliği için FFmpeg
- Model servisleri için EVREN API erişimi
- Anomali skorlayıcı checkpoint dosyası

> Python 3.9 kullanmayın. Projede `str | None` gibi Python 3.10 sözdizimi kullanılmaktadır.

## Kurulum

### 1. Sanal ortam

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Aktif Python sürümünü kontrol edin:

```bash
python --version
which python
```

### 2. FFmpeg

FFmpeg yalnızca video kesitini kalıcı MP4 dosyası olarak kaydeden `save_video_segment` aracı için gereklidir. Python paketi olmadığı için `requirements.txt` ile kurulmaz.

macOS:

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

Kontrol:

```bash
ffmpeg -version
ffprobe -version
```

### 3. Ortam değişkenleri

Örnek dosyayı kopyalayarak proje kökünde `.env` oluşturun:

```bash
cp .env.example .env
```

Ardından kendi servis ve model değerlerinizi düzenleyin. Temel yapı:

```dotenv
EVREN_API_KEY="..."
EVREN_URL="https://.../v1"
EVREN_LLM_MODEL="llm-fast"
EVREN_VLM_MODEL="vlm"

AS_MODEL_NAME="s3d"
AS_CLIP_SIZE=32
AS_OVERLAP=16
AS_FPS=20
AS_WIDTH=224
AS_HEIGHT=224
AS_BATCH=5
AS_FC_CHECKPOINT="Checkpoint/best_loss_fold_3.pt"

VLM_SOURCE_SAMPLE_FPS=5
VLM_MAX_FRAMES=128
VLM_MIN_FRAMES=8
VLM_MAX_EDGE=448
VLM_DIMENSION_ALIGNMENT=32
VLM_DEFAULT_ENCODE_FPS=5
VLM_MIN_ENCODE_FPS=0.25
VLM_MAX_ENCODE_FPS=30
```

`.env` dosyasını veya gerçek API anahtarlarını Git'e eklemeyin.

## Çalıştırma

```bash
source .venv/bin/activate
python lab.py
```

Arayüz:

```text
http://127.0.0.1:7860
```

Port kullanımda ise eski Gradio sürecini kapatın veya farklı bir port yapılandırın.

## Lab sekmeleri

### LLM

Metin modelini ve isteğe bağlı konuşma geçmişini test eder.

### VLM

Görüntü veya videodan seçilen karelerle görsel soru-cevap yapar.

### Analyzer

Videoyu doğrudan anomali modelinden geçirir; zaman aralıklarını ve skor grafiğini üretir.

### Agent

Kullanıcı isteğini planlar, gerekli araçları çalıştırır ve sonucu Reviewer ile kontrol eder.

## Agent araçları

| Araç | Görev |
|---|---|
| `run_abnormal_event_segmenter` | Videodaki anormal zaman aralıklarını ve skorlarını bulur. |
| `analyze_video_with_vlm` | Belirli zaman aralığını VLM ile açıklar. |
| `save_video_segment` | Belirtilen aralığı FFmpeg ile MP4 olarak kaydeder. |
| `get_video_info` | Video süresi, FPS ve çözünürlük bilgisini döndürür. |

Tool sonuçları ortak bir JSON zarfı kullanır:

```json
{
  "ok": true,
  "data": {},
  "warnings": [],
  "error": null
}
```

Tool yetenekleri kendi açıklama ve parametre şemalarında tanımlıdır. Planner bu
kataloğu kayıtlı toollardan otomatik üretir; system prompt içinde ayrı bir tool
listesi tutulmaz.

Genel bir video analizi isteğinde varsayılan akış önce anomalileri bulmak, ardından bulunan segmentleri VLM ile açıklamaktır. Kullanıcı doğrudan bir zaman aralığı sorarsa segmenter çalıştırılmadan VLM kullanılabilir.

## VLM zaman ve kare yönetimi

Bir segment için ihtiyaç duyulan kare sayısı:

```text
segment süresi × VLM_SOURCE_SAMPLE_FPS
```

Sonuç `VLM_MAX_FRAMES` ile sınırlandırılır. Geçici MP4'ün encode FPS'i kaynak süreyi koruyacak şekilde hesaplanır:

```text
encode FPS = gönderilen kare sayısı / kaynak segment süresi
```

Örnek:

```text
20 saniye × 5 FPS = 100 kare
100 kare / 20 saniye = 5 encode FPS
```

VLM'e videonun tamamı değil, istenen veya segmenter tarafından bulunan zaman aralığından örneklenen kareler gönderilir. Segmentasyon çıktıları olay bağlamı için kısa bir başlangıç/bitiş dolgusu içerebilir.

## Çıktılar

Analyzer ve Agent çıktıları varsayılan olarak aşağıdaki dizinde toplanır:

```text
_stuff/lab_runs/<video_adı>/
```

Örnek grafik:

```text
segmentation_graph.png
```

`save_video_segment` çıktısı, verilen `output_filename` konumuna kaydedilir.

## Klip kaydetme testleri

Manuel kabul senaryoları için:

[SAVE_VIDEO_SEGMENT_TEST_SCENARIOS.md](SAVE_VIDEO_SEGMENT_TEST_SCENARIOS.md)

## Model eğitimi

Segment skorlayıcı eğitim girişi:

```bash
python segment_ranking_model_train.py
```

Eğitim kodu önceden çıkarılmış normal/anormal `.pt` özellik dosyalarını kullanır ve mevcut haliyle CUDA bekler. Veri dizinleri ile hiperparametreler çalıştırmadan önce `segment_ranking_model_train.py` içinde kontrol edilmelidir.

## Önemli dosyalar

```text
lab.py                         Gradio test arayüzü
utils/agents.py                LangGraph agent akışı
utils/prompts.py               Planner, Executor ve Reviewer talimatları
utils/tools.py                 Agent araçları
utils/video_analyzer_model.py  Video analiz ve segmentasyon modeli
utils/fc_model.py              SegmentRankingModel ve eğitim kodu
utils/video_process.py         Video/kare işleme yardımcıları
utils/vlm.py                   VLM istemcisi ve geçici MP4 hazırlama
utils/llm.py                   Metin LLM yöneticisi
```

## Platform notları

- CUDA varsa inference için CUDA seçilir.
- Apple Silicon cihazlarda MPS kullanılabilir.
- CUDA/MPS yoksa CPU kullanılır.
- TensorRT yalnızca NVIDIA CUDA ortamında desteklenir.
- macOS'ta OpenCV ve Decord farklı SDL/FFmpeg kütüphaneleri yüklediği için terminalde yinelenen sınıf uyarıları görülebilir.

## Bilinen sınırlar

- VLM servisinin kabul ettiği maksimum video/kare kapasitesi servis ortamında ayrıca doğrulanmalıdır.
- FFmpeg `-c copy` kullandığı için kesim başlangıcı codec keyframe yapısına bağlı olarak küçük zaman farkları gösterebilir.
- Anomali skoru olayın kesin türü veya hukuki niteliği değildir.
- VLM açıklamaları model çıkarımıdır; kritik güvenlik kararlarında insan doğrulaması gerekir.

## Lisans

Proje [MIT Lisansı](LICENSE) altında sunulmaktadır.
