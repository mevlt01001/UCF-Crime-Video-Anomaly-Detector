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

FFmpeg, `save_video_segment` ve nesne takibindeki kutulu MP4 çıktısı için gereklidir. Yalnız nesne tespiti için gerekli değildir. Python paketi olmadığı için `requirements.txt` ile kurulmaz.

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

### Video Raporu

Video yükleyip **Rapor oluştur** ile aynı planner/executor/tools/reviewer akışını
sohbet geçmişinden bağımsız çalıştırır. Onaylanan JSON ekranda gösterilir ve
indirilebilir; bu sekmede takip mesajı yoktur. Normal Agent sohbeti değişmez.

Alanlar: `ozet`, `olaylar` (`saniye`, `aciklama`), `risk_seviyesi`
(`dusuk`, `orta`, `yuksek`), `eylemler` (şimdilik daima `[]`). Risk, modelin
görsel bulgulara dayalı değerlendirmesidir; anomali skoru veya güvenlik garantisi
değildir. İncelenen kapsam ve belirsizlikler özette belirtilir.

Şema/zaman/kapsam doğrulaması veya reviewer onayı başarısızsa rapor indirmeye
sunulmaz. Dosyalar `_stuff/lab_runs/reports/` altında benzersiz adlarla saklanır;
otomatik dosya temizliği yoktur. Video değiştiğinde ekrandaki önceki sonuç temizlenir.

Modelin eklediği tek bir JSON kod bloğu veya baştaki `json` etiketi doğrulama öncesi
ayıklanır. Kod bloğunun çevresindeki açıklamalar nihai rapora eklenmez; onaylanan
nesne saf JSON olarak sunulur. Yarım JSON veya birden fazla olası rapor kabul edilmez;
şema, zaman, görsel kapsam ve reviewer kontrolleri korunur.

Rapor regresyon testleri (API çağrısı ve video analizi yapmaz):

```bash
python -B -m unittest discover -s tests -p 'test_reporting.py' -v
```

## Agent araçları

| Araç | Görev |
|---|---|
| `run_abnormal_event_segmenter` | Videodaki anormal zaman aralıklarını ve skorlarını bulur. |
| `analyze_video_with_vlm` | Belirli zaman aralığını VLM ile açıklar. |
| `save_video_segment` | Belirtilen aralığı FFmpeg ile MP4 olarak kaydeder. |
| `get_video_info` | Video süresi, FPS ve çözünürlük bilgisini döndürür. |
| `detect_and_track_objects` | Nesneleri ve görünme aralıklarını bulur; isteğe bağlı takip kutulu MP4 üretir. |

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

## Nesne tespiti ve takip (isteğe bağlı kurulum)

Hazır **YOLO11s + ByteTrack** kullanılır; eğitim gerekmez. Mevcut anomali,
VLM, sohbet ve rapor akışı korunur. Plaka OCR ve kategorili arşiv henüz bu
özelliğin parçası değildir; rapordaki `eylemler` şimdilik boş kalır.

Önce temel kurulumu, ardından aşağıdakileri çalıştırın (proje kökünde):

```bash
python -m pip install -r requirements-objects.txt
mkdir -p _stuff/models
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt -o _stuff/models/yolo11s.pt
```

Windows PowerShell'de klasör için `New-Item -ItemType Directory -Force _stuff/models`,
indirme için `curl.exe` kullanın. Mevcut bir ağırlık varsa yeniden indirmek gerekmez.
Model yalnız yerel dosyadan yüklenir; tool çalışırken otomatik paket kurulmaz,
model indirilmez veya video bir servise gönderilmez. Eksik bağımlılık/ağırlık
yalnız bu toolu etkiler. Temel paket sürümleri değiştirilmez.
Kutulu video için PyAV da isteğe bağlı paket listesine dahildir; yeni tool bu
paketi çizim/kodlama aşamasında kullanır. Önceden kurulum yaptıysanız `requirements-objects.txt`
kurulumunu tekrar çalıştırın.

Ultralytics 8.3.225, projenin NumPy 2.0.2 sürümüyle çözümleme kontrolünden
geçtiği için sabitlenmiştir; 8.4.131 bu NumPy sürümünü dışlar.
YOLO11 kod/ağırlıkları AGPL-3.0 / Enterprise koşullarına tabidir;
projenin MIT lisansı bu bağımlılıkların lisansını değiştirmez.
[Resmî model ve lisans bilgisi](https://docs.ultralytics.com/models/yolo11/).

Örnek sohbet istekleri:

- “10–20 saniye arasında arabaların göründüğü aralıkları bul.”
- “Bu kesitteki kişileri takip et, kutulu video oluştur.”
- “Arabaların bulunduğu aralığı ayrı klip olarak kaydet.” (Tespit + mevcut kaydetme toolu.)

Sınıf filtresi COCO'nun İngilizce sınıf adlarıdır (`person`, `car`, `truck` vb.).
Boş liste geçersizdir; filtre verilmezse tüm sınıflar incelenir. Silah/plaka,
renk veya kişi kimliği bu hazır modelin sınıfı değildir. Desteklenmeyen sınıf
istenirse modelin desteklediği liste döner. Agent bunları tool açıklamasından öğrenir.

Çıktılar `_stuff/lab_runs/actions/objects/` altındadır:

- `frames.json`: kaynak kare indeksi/saniyesi, orijinal görüntü piksel koordinatları
  (`xyxy`), sınıf, güven ve takip ID'si (henüz atanmadıysa `null`).
- `intervals.json`: sınıf veya takip ID'si için kesintisiz görünme aralıkları.
  Bitiş hariçtir; tespitin kaybolduğu boşluklar birleştirilmez.
- `annotated.mp4`: yalnız istendiğinde; kutulu video, varsa kaynak ses korunur.
- `manifest.json`: ayarlar, çıktı özeti ve dosya bütünlük bilgisi.

Tool sınıf aralıklarına öncelik vererek en çok 100 aralığı döndürür; daha fazlası varsa `intervals_truncated=true`
ve uyarı verir. Tüm aralıklar dosyada kalır; kesilmiş özet tam liste gibi
sunulmamalıdır. Kutu hareketi kare kare güncellenir; takip ID'si gerçek kimlik
değildir ve örtüşmede değişebilir. Tespit olmaması kesin yokluk kanıtı değildir.
`detection_count` karelerdeki toplam kutu sayısıdır, tekil kişi/araç sayısı değildir.

Her kare, kaynak zaman damgasıyla işlenir. Başlangıç/bitişe uyan ilk/son
kare `sampled_range` içinde raporlanır. Değişken FPS videonun kutulu kopyası
PyAV ile her karenin kendi zaman damgası ve gösterim süresi korunarak yazılır;
FFmpeg bu görüntüyü yeniden kodlamadan varsa kaynak sesi ekler. Çıktının 0. saniyesi
`sampled_range.start_sec` değerine karşılık gelir. Her kare zamanı (1 ms tolerans)
ve son kare süresi doğrulanır; eski `VFR_NORMALIZED` yaklaşımı kullanılmaz.
`output_timing="source_timestamps"`; `output_fps` ortalamadır, sabit FPS zorlaması değildir.

`.env.example` ayarları: `OBJECT_DEVICE=auto` CUDA → MPS → CPU seçer;
`cpu`, `cuda` veya `mps` ile açık seçim yapılabilir. Kullanılamayan açık seçim
hata döner; sessiz cihaz değişimi yapılmaz. Varsayılan model girdisi 640,
raporlanan tespit güven eşiği 0.25, kare başına en çok 100 tespit,
iş başına en çok 18000 kare ve 900 saniye kontrol süresidir. Sınır aşılırsa
aralık kısaltılmalı veya ayar değiştirilmeli; kareler sessizce atlanmaz.
Süre sınırı kareler arasında ve FFmpeg'de uygulanır; tek bir yerel model/decoder
çağrısını işletim sistemi seviyesinde kesen bir watchdog değildir.

Model erişimi ve çıktı üretimi kilitlidir; eşzamanlı işler sıraya girer.
Her iş ayrı okuyucu/takip durumu kullanır. Aynı kaynak dosya kimliği, ağırlık,
ayar, sınıflar ve aralık için tespit ve çizim ayrı önbelleklenir. Önce tespit,
sonra kutulu video istendiğinde YOLO yeniden çalışmaz (`detection_cache_hit=true`),
yalnız çizim/kodlama yapılır. Kutulu video da hazırsa `cache_hit=true` olur.
Yalnız MP4 bozulursa tespit korunur; `frames.json` bozulursa tespit yeniden yapılır.
Çizim başarısız olsa bile sağlam tespit kaydı sonraki denemede kullanılabilir.
Kaynaklara dokunulmaz; başarısız aşamanın geçici dosyaları temizlenir. Tamamlanmış
çıktılar otomatik silinmez. Önceki cache sürümünün dosyaları silinmeden bırakılır,
bu düzeltmeden sonraki ilk çağrıda bir kez yeniden analiz edilir.
Başarısız model yükleme/cihaz aktarımı önceki sağlam model önbelleğini bozmaz.
Bu aşamada arayüz değiştirilmez; dosya yolları tool sonucundadır.

Manuel kabul senaryoları: [OBJECT_TRACKING_TEST_SCENARIOS.md](OBJECT_TRACKING_TEST_SCENARIOS.md).

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
