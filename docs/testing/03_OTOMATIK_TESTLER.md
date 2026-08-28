# Otomatik testler — daha sonra çalıştırma rehberi

**Bu görevde testler çalıştırılmadı.** Yalnız kaynak/sözdizimi ve doküman eşleştirmesi kontrol edildi. Manuel kabul öncelikli; bu testler modelin gerçek video yorumunu doğrulamaz.

## 1. Test grupları

| Dosya | Kontrol edilen | Yerine geçmediği manuel kontrol |
|---|---|---|
| `tests/test_reporting.py` (mevcut) | JSON sarmalları, şema/kapsam, gerçek graph + taklit model yanıtları | gerçek VLM olay doğruluğu |
| `tests/test_ui_backend.py` (mevcut) | HTTP routes, session ayrımı, kilit/iptal, video değiştirme | tarayıcı/gerçek SSE kopması ve görsel arayüz |
| `tests/acceptance/test_report_contracts.py` | görsel kapsam birleşimi, yanlış video, başarısız VLM, eylem kaydı, OCR manifest bağı | kategori/plaka doğruluğu ve dosyanın gerçekten var olması |
| `test_archive_contracts.py` | gerçek arşiv/cache/metadata, tekrar/kategori/bozuk kayıt, temizlik | FFmpeg ve klibin oynatılabilirliği (export taklit) |
| `test_plate_tracking_contracts.py` | koordinat dönüşümü, sınıf filtreleri, zaman alt sınırı, aralık/crop/manifest, OCR çözümleme | gerçek YOLO/ByteTrack/ONNX doğruluğu |
| `test_agent_lifecycle.py` | gerçek node'lar, hedef prompt, kanıtsız raporu reddetme, bütçe, kanonik çıktı | modelin doğru adımı seçmesi ve zincirin kurulması |
| `test_media_contracts.py` | örnekleme bütçesi/offset, normal süre aralığında encode FPS, min kare/hizalama | gerçek decoder, codec ve VFR/video geometri kabulü |
| `test_backend_contracts.py` | gerçek worker/SSE payload üretimi, rapor dosyası, pre-cancel, rapor/sohbet ayrımı | TCP/HTTP SSE, tarayıcı render ve gerçek model |
| `test_tool_boundaries.py` | gerçek tool wrapper'ları, sınır/hata/uyarı, segmenter çağrısı | gerçek video/metadata ve model sonucu |
| `test_model_math.py` | CPU'da küçük FC eval ve loss aritmetiği | eğitim, checkpoint kalitesi, genelleme ve veri ayrımı |
| `gap_report_safety.py` | **hedef kabul şartları; mevcut açıklar nedeniyle başarısızlık beklenen ayrı grup** | bir düzeltmenin yapıldığı iddiası değildir |

`support.py` ortak fixture/loader sağlar. Kaynak fonksiyonlar kopyalanıp yeniden yazılmadı; gerçek modüller yüklenir, model/decoder/FFmpeg gibi dış sınırlar mock edilir. Kaynak koddan AST ile davranış kopyalama yok. Testsiz yalnız AST kontrolü ise teslim kontrolünün bir parçasıdır, çalışma zamanı testi değildir.

Yeni `test_*.py` grubunda testler çalışırken dış TCP bağlantıları yasaklanır; gerçek LLM client oluşturulmaz. Yükleme sırasında normal Python bağımlılıkları import edilir; eksik paket varsa import hatası test geçişi değildir. `test_model_math` küçük rastgele CPU MLP kurar; pretrained model, eğitim veya ağ indirmesi yapmaz. Model ağırlığı ve gerçek API anahtarı gerekmez.

## 2. Önkoşullar

- Proje kökünde Python 3.10+ `.venv`; mevcut temel requirements kurulu olmalı. NumPy, Torch, OpenCV/Decord, LangChain/Pydantic ve mevcut testler için FastAPI/TestClient bağımlılıkları gerekir.
- Yeni testler YOLO, ONNX Runtime, FFmpeg, gerçek video, checkpoint veya EVREN çağrısı gerektirmez. Gerçek model testleri manuel planın ayrı aşamasıdır.
- Çalışan uygulamayı testler yönetmez; silmez/yeniden başlatmaz. Yeni dosya testleri TemporaryDirectory kullanır. Python/Matplotlib gibi kütüphanelerin cache üretimi mümkün olduğundan aşağıdaki komutlarda geçici cache klasörü kullanılır.
- `test_main.py` interaktif CLI'dir, `segment_ranking_model_train.py` importta CUDA eğitimi başlatır: ikisini test keşfine dahil etmeyin.
- **Repo kökünde gelişigüzel `pytest` veya tüm `.py` dosyalarını import eden bir smoke komutu kullanmayın.** Açık discovery dizini/deseni kullanın.

## 3. Zamanı geldiğinde çalıştırılacak komutlar

**Çıktıları paylaşmak için önerilen yol:** proje kökünde aşağıdaki komutlardan birini çalıştırın. Her biri `_stuff/test_runs/` altında ayrı klasöre tam `output.log` ve gerçek çıkış kodunu içeren `summary.json` yazar. Bir grubu çalıştırıp sonucunu değerlendirdikten sonra diğerine geçin. Hazırlayıcı/test çalıştırıcı kullanımı [adım adım rehberde](05_ADIM_ADIM_UYGULAMA.md).

```bash
.venv/bin/python scripts/testing/run_checks.py existing
```

```bash
.venv/bin/python scripts/testing/run_checks.py contracts
```

```bash
.venv/bin/python scripts/testing/run_checks.py gaps
```

```bash
.venv/bin/python scripts/testing/run_checks.py frontend
```

Aşağıdaki doğrudan komutlar alternatif/tek dosya çalıştırma içindir; yukarıdakiyle aynı grupları yeniden çalıştırmanız gerekmez.

Aşağıdakiler yönergedir; bu görevde çalıştırılmadı. Shell yorum satırları kopyalama sorunlarına yol açmaması için komut bloklarında yoktur. Çıktıların başarılı olup olmadığını ayrı değerlendirin; zincirleme komutlar önceki hatayı gizlemesin.

Önce mevcut regresyonlar (22 sayısı geçmiş çalıştırmaya aittir; güncel çıktıyı esas alın):

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
```

Sonra yeni **izole sözleşme** grubu:

```bash
TEST_CACHE_DIR=$(mktemp -d /tmp/ucf-acceptance-cache.XXXXXX)
MPLCONFIGDIR="$TEST_CACHE_DIR" XDG_CACHE_HOME="$TEST_CACHE_DIR" .venv/bin/python -B -m unittest discover -s tests/acceptance -p 'test_*.py' -v
```

Mevcut `tests` keşfi, `tests/acceptance` içinde `__init__.py` olmadığı için yeni grubu içermez (hedef Python 3.10 unittest davranışı). Açık `-s tests/acceptance` komutu gereklidir. Böylece henüz çalıştırılmamış yeni suite eski test sayısının içine gizlenmez.

Tek bir alt grup:

```bash
.venv/bin/python -B -m unittest discover -s tests/acceptance -p 'test_archive_contracts.py' -v
```

**Bilinen açıkları ölçmek için ayrıca**, düzeltme çalışmasına başlarken:

```bash
.venv/bin/python -B -m unittest discover -s tests/acceptance -p 'gap_*.py' -v
```

Bu grup KG01–KG06 için istenen davranışı assert eder. Bugünkü kaynakta FAIL beklenir; `expectedFailure`/`xfail`/skip ile yeşile boyanmadı. Başarısızlıkları “test bozuk” diye silmeyin; traceback'i inceleyin. Bir düzeltme sonrası PASS olursa karşılık gelen manuel senaryo da yeniden çalıştırılmalı. Harici import/runtime hatası, hedef bug'ın yeniden üretildiği anlamına gelmez.

Frontend derleme/statik kontrolleri (tarayıcı testi yerine geçmez):

```bash
npm --prefix frontend run lint
npm --prefix frontend run build
```

Build `frontend/dist` ve TypeScript cache çıktısı üretebilir. Bu komutların çalıştırılması Git add/commit/push anlamına gelmez.

## 4. Bilinen açık testlerinin kapsamı

| Test | Manuel senaryo | İstenen davranış |
|---|---|---|
| KG01 | G03 | Reviewer, tool sonucu yokken bile gerçek hedef video yolunu görür |
| KG02 | G02/R02 | Boş segment + görsel kontrol yok → doğrulanmış normal rapor yok |
| KG03 | G01/M01 | Dikey görüntünün içeriği yatay esnetilmez |
| KG04 | G05 | Sınır/hata final metni “approved” etiketlenmez |
| KG05 | R05 | Çağrı karşılığı olmayan ToolMessage görsel kanıt sayılmaz |
| KG06 | S02 | Agent Analyzer checkpoint olmadan model kurup başarı üretmez |

KG03 test fixture'ı düz yeniden boyutlandırmayı denetler. Gelecekte letterbox uygulanırsa test padding içindeki gerçek içerik oranını ölçmek üzere güncellenmeli; testi yalnız kaldırmak çözüm değildir.

## 5. Henüz otomatik olmayan kritik alanlar

- Gerçek EVREN/VLM nesne yönü, olay/risk/kategori doğruluğu ve plaka uydurma.
- Takip→plaka ROI→OCR→olay bağlama: bağlantı üretim kodunda yok. Olmayan metodu çağıran sahte “uçtan uca geçti” testi yazılmadı.
- Gerçek YOLO/ByteTrack ID değişimi, bütün plaka kırpımlarının görüntü doğruluğu, gerçek ONNX modeli.
- FFmpeg ses/PTS/kare doğrulaması, VFR, codec/rotation; bunun için manuel M/A/O senaryoları gerekir.
- Browser olay sırası, geciken fetch, SSE tekrar bağlanma, UI yeni sohbet/iptal kontrolleri.
- Eğitim veri kümesi ayrımı ve genelleme ölçümü; training script'i bu paket çalıştırmaz.
- Kaynak/model dosyalarının işlem ortasında değişimi ve native çağrı takılması için kapsamlı stres testleri.

Bu eksikler test kapsamı borcudur; mevcut mock testlerin geçmesi bu alanları PASS yapmaz. Her gerçek model çalıştırması görüntülerin EVREN servisine gönderilebileceği, süre/maliyet ve özel veri açısından ayrıca planlanmalıdır.
