# Test kapsamı: neyi hangi kanıtla kabul edeceğiz?

Paket: **84 manuel senaryo** (38 P0, 43 P1, 3 P2), **47 yeni izole regresyon testi**, **6 ayrı açık kabul testi**. Sayılar test fonksiyonlarının statik sayımıdır; çalıştırılma/geçiş sayısı değildir. Önceden bulunan testler ayrıca korunmuştur.

| Manuel aile | Yeni otomatik destek | Hâlâ gerekli gerçek kontrol |
|---|---|---|
| G01–G08 genel rapor | KG01–KG04, AgentLifecycle, ToolBoundaries | yüklenmiş gerçek video, anlam/olay/yön, modelin araç seçimi ve toparlanması |
| U01–U08 oturum/UI | BackendContracts + mevcut test_ui_backend.py | iki browser, geciken network, SSE/reconnect, Gradio global hafıza, native iptal beklemesi |
| R01–R08 rapor | ReportContracts, AgentLifecycle, KG02/KG05 | görsel iddiaların doğruluğu, kaynak videoya bakarak olay zamanı, indirme ekranı |
| E01–E06 eylemler | ReportContracts, ArchiveContracts | gerçekten üretilmiş dosyalar ve modelin eylem seçme gerekçesi |
| Z01–Z08 hedef zincir | **Tam zincir için test uygulaması yok: ürün bağlantıları eksik.** E/R alt sözleşmeleri destekleniyor. | olay–iş–araç–kare–plaka bağları, seç/atla kararı, çok araç ve ID değişimi |
| S01–S06 segmenter | ToolBoundaries, KG06, ModelMath (yalnız FC matematiği) | gerçek S3D/FC checkpoint, son kare kapsamı, cihaz ve grafik |
| M01–M06 medya | MediaContracts, KG03, ToolBoundaries | gerçek decoder/encoder, içerik oranı, VFR/rotation ve ses |
| O01–O06 takip | TrackingContracts | gerçek ByteTrack/YOLO kutuları ve ID, render cache akışı, native süre sınırı |
| P01–P06 plaka | PlateContracts koordinat ve çıktı sözleşmeleri | gerçek modelin plaka recall/precision'ı, her kare kırpımları, dosya temizliği |
| C01–C06 OCR | PlateContracts manifest/decoder/boş-kırpım | gerçek ONNX okunabilirlik, PNG değişim yarışları, hash'siz dosyalar, uzun işler |
| A01–A06 kayıt | ArchiveContracts; export mock | FFmpeg kare/süre/ses, kaynak üstüne yazma koruması, kullanıcı hedef yolu |
| X01–X04 ortam | çalıştırma yönergeleri; kapsamlı otomasyon yok | doğru process/build, .env, opsiyonel kurulum, RAM/disk trendi |
| T01–T03 eğitim/CLI | ModelMath yalnız T02 aritmetiği | kaynak bazında train/val ayrımı, fold kapsamı, CLI gerçek hafıza |
| Q01–Q03 kalite | **Model kalitesi için mock test yeterli değil.** | iki insan inceleyici, etiketli test seti, yanlış-normal/yanlış-kategori/yanlış-plaka ilişki ölçümü |

## Sonuçların birbirinin yerine kullanılmaması

- `ArchiveContracts` PASS: arşiv kayıt ve tekrar kullanma mantığı sahte export çıktısıyla çalıştı. Gerçek MP4 doğru kesildi demek değildir.
- `PlateContracts` PASS: sentetik ONNX çıktısı doğru koordinata/karaktere çevrildi. Gerçek plaka doğru okundu demek değildir.
- `BackendContracts` PASS: gerçek worker sahte graph çıktısını doğru yayımladı. Graph'ın olayı doğru analiz ettiği anlamına gelmez.
- `gap_*.py` FAIL: traceback hedef assert'ten geliyorsa beklenen açığı ortaya koyabilir. Import/kurulum hatası aynı kanıt değildir.
- Bütün mevcut alt araçlar PASS, Z testleri BLOCKED olabilir. Bu durumda “araçlar çalışıyor, zincir henüz hazır değil” sonucu verilir.

## İleride eklenecek otomasyon sırası

1. KG01–KG06 düzeltmeleri için mevcut kabul testlerini yeşile getir; ilgili manuel G/R/S senaryosunu tekrar çalıştır.
2. Karar ve ROI/track ilişki sözleşmesi tasarlandığında Z01/Z03/Z04/Z05 için aynı videoda yanlış plaka bağlamayı reddeden gerçek entegrasyon testleri ekle. Önce sahte “başarılı zincir” yazma.
3. Yerel sentetik videoyla gerçek FFmpeg/PyAV süre/PTS testleri; ardından gerçek YOLO/ONNX fixture seti. Bu gruplar varsayılan offline suite'ten ayrı olsun.
4. Tarayıcı testleri: yeni sohbet, çift tıklama, iptal, SSE kopması, sayfa yenileme, indirme ve iki sekme.
5. Etiketli video kümesiyle gerçek model değerlendirmesi; servis çağrısı sayısı/maliyet ve özel veri aktarımı için açık çalıştırma onayı.
