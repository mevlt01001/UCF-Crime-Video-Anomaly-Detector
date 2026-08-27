# Nesne tespiti ve takip — manuel kabul testleri

Kapsam: YOLO11s + ByteTrack, zaman aralıkları, kutulu MP4. Plaka OCR,
kategorili arşivleme ve rapor eylemleri bu aşamaya dahil değildir.

## Hazırlık

- README'deki isteğe bağlı nesne tespiti kurulumunu tamamlayıp uygulamayı yeniden başlatın.
- Kişi/araç içeren kısa bir video, nesnesiz bir video ve örtüşen nesnelerin olduğu bir video seçin.
- Node trace'te `detect_and_track_objects` çağrısını ve gerçek tool sonucunu kontrol edin;
  yalnız sohbetin “başarılı” demesi yeterli değildir.
- `data.details_path` kareleri, `data.intervals_path` tüm aralıkları,
  `data.annotated_video_path` ise varsa MP4'ü gösterir. Dosyalar yereldir;
  bu aşamada arayüze yeni oynatıcı/indirme alanı eklenmedi.

## Senaryolar ve beklenen sonuçlar

| # | Deneme | Beklenen |
|---|---|---|
| 1 | “0–5 saniyede kişilerin göründüğü aralıkları bul.” | `classes=["person"]`, `ok=true`; sınıf aralıkları kaynak saniyesinde, kutulu video istenmediyse MP4 yok. |
| 2 | Tespit yaptıktan sonra “Aynı kesitte kişileri takip et ve kutulu video oluştur.” | Aynı video/aralık/ayar/sınıflarla `detection_cache_hit=true`, `cache_hit=false`; yeniden YOLO çalışmadan çizim yapılır. MP4 açılır, kutular hareket eden kişileri izler. Orijinal video değişmez. |
| 3 | “2.05–4.55 saniyesini incele.” | Kare zamanları istenen aralık içindedir; geçici klibin 0. saniyesi kaynak videonun 0. saniyesi diye sunulmaz. `sampled_range` gerçek kare sınırlarını gösterir. |
| 4 | Kişi ve araba filtresi kullanın; sonra filtre vermeden deneyin. | İlkinde yalnız istenen sınıflar, ikincide modelin tüm desteklediği sınıflar değerlendirilir. |
| 5 | Doğrudan toola `classes=["gun"]` veya `[]` verin. | `ok=false`, `UNSUPPORTED_OBJECT_CLASS` ve desteklenen sınıflar. Silah var/yok iddiası üretilmez. Agent bu isteği VLM'e yönlendirebilir. |
| 6 | İstenen nesne bulunmayan video. | İşlem tamamlanır: `ok=true`, ilgili aralık listesi boş. Sonuç “model tespit etmedi”; kesin yokluk garantisi değil. |
| 7 | Nesne çıkar, sonra yeniden girer. | Görünmeyen boşluk tek kesintisiz aralık gibi birleştirilmez. Aynı ID korunabilir veya değişebilir; gerçek kişi kimliği iddia edilmez. |
| 8 | İki kişi birbirini örter veya yolları kesişir. | Videoda kutular kontrol edilir. ID değişimleri ölçülür; “takip kusursuz” varsayılmaz. `null`/`?` henüz onaylanmamış ID anlamındadır. |
| 9 | Aynı video/aralık/ayarlarla toolu yeniden çağırın. | `cache_hit=true`, aynı dosya yolları; yeni kopyalar oluşmaz. Agent toolu yeniden çağırmadıysa bu cache testi yapılmış sayılmaz. |
| 10 | Yalnız test çıktısı kopyasında `frames.json` içeriğini değiştirip aynı çağrıyı tekrarlayın. | Bütünlük kontrolü eski çıktıyı reddeder; yeni klasörde yeniden üretir, `cache_hit=false`. |
| 11 | Video süresini aşan başlangıç; sonra yalnız bitişi aşan istek. | İlkinde `TIME_OUT_OF_RANGE`; ikincide video sonunda kırpma ve `END_TIME_CLAMPED` uyarısı. Negatif/ters aralık reddedilir. |
| 12 | `.env` içinde geçici olarak olmayan model yolu kullanın; uygulamayı yeniden başlatın. | `OBJECT_MODEL_NOT_FOUND`; sohbet/metadata gibi diğer işler açılmaya devam eder. Test sonrası ayarı geri alın. |
| 13 | FFmpeg olmayan test ortamında kutulu video isteyin. | `FFMPEG_NOT_FOUND`; başarı veya sahte dosya yolu yok. `render_video=false` ile tespit çalışabilir. |
| 14 | Yazma izni olmayan test çıktı dizini / bozuk video. | `ok=false`, açıklayıcı hata; tamamlanmamış çıktı başarı sayılmaz. Kaynak ve eski sağlam dosyalar korunur. |
| 15 | Geçici `OBJECT_MAX_FRAMES=5` ile daha uzun kesit deneyin. | `FRAME_LIMIT_EXCEEDED`; ilk 5 kareyle sessizce tamamlanmaz. Ayarı test sonunda geri alın. |
| 16 | Çok küçük `OBJECT_TIMEOUT_SEC` ile işlem deneyin. | `TRACKING_TIMEOUT`; kısmi analiz tamamlanmış sayılmaz. Sınır kareler arasında kontrol edilir, tek native çağrıyı zorla durdurmaz. |
| 17 | İki ayrı oturumdan farklı videoları aynı anda gönderin. | İşler sıraya girer; dosyalar/kaynak zamanları karışmaz. Her işin ID alanı bağımsızdır; iki işte de `#1` olması normaldir. |
| 18 | Sesli video ve belirgin değişken FPS video ile kutulu çıktı alın. | Kaynak kare zamanları korunur (`output_timing=source_timestamps`); görüntü/ses olayı aynı anda kalır. Örneğin 0.5 sn'deki kare 2.3 sn'ye kaymaz. Başlangıcı sıfır olmayan kesitte çıkış zamanı = kaynak zamanı − sampled_range.start_sec. |
| 19 | Çok kalabalık video / çok sayıda kısa görünme aralığı. | Tespit sınırında uyarı; 100'den fazla aralıkta `intervals_truncated=true`. Tam liste dosyada, sınıf aralıkları özette önceliklidir. |
| 20 | CPU ve varsa MPS/CUDA ile aynı kısa kesit. | Doğru cihaz `data.device` içinde; geçerli çıktı. Süreler ve model sonuçları birebir aynı olmak zorunda değil. Olmayan açık cihaz seçimi `DEVICE_UNAVAILABLE`. |
| 21 | “Arabaların olduğu aralığı klip olarak kaydet.” | Tespit aralığı mevcut `save_video_segment` aracına iletilir. Klip gerçekten açılır; tespit toolu tek başına klip kaydettiğini söylemez. |
| 22 | Normal sohbet, VLM sorusu, anomali analizi ve JSON raporu. | Mevcut planner–executor–reviewer sırası ve son cevap davranışı korunur; raporda `eylemler` hâlâ `[]`. |
| 23 | Yalnız test için üretilen kutulu MP4'ü bozup aynı çağrıyı tekrar edin. | `detection_cache_hit=true`, `cache_hit=false`; YOLO yeniden çalışmaz, sağlam tespitten yeni MP4 oluşturulur. |
| 24 | Tespit tamamlandıktan sonra çizim/kodlama hatası oluşturup tekrar deneyin. | Başarısız çizim başarı sayılmaz. Tespit JSON'ları korunur; sonraki çizim denemesinde tekrar kullanılır. |
| 25 | Aynı Python sürecinde sağlam model A yükleyin, bozuk model B yüklemeyi deneyin, A'ya dönün. | B hata verir; A kullanılabilir kalır. `NoneType` hatası/restart zorunluluğu oluşmaz. Bu test uygulamayı arada yeniden başlatmadan yapılmalıdır. |

## Başarı ölçütü

Dosyanın açılması, zamanların doğru olması, hataların sahte başarıya dönüşmemesi
ve görevlerin birbirine karışmaması teknik kabul ölçütleridir. Model doğruluğunu
ayrıca gözle işaretlenmiş videolarda değerlendirin: kaçırılan nesne, yanlış kutu
ve ID değişimi sayısını not edin. Tek kısa deneme bütün videolarda doğruluk kanıtı değildir.
