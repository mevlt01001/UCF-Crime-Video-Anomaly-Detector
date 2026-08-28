# Kategorili klip arşivi — manuel testler

Uygulamayı yeniden başlatın. FFmpeg kurulu olsun. Görsel olarak sınıflandırılabilecek
kısa bir olay videosu kullanın. Örnek: “2–5 saniyedeki kesiti kavga/saldırı olarak
yerel arşive kaydet; gerekçeyi gördüğün kanıta göre yaz.” Node trace içinde
`archive_anomaly_clip` çağrısını ve tool JSON’unu kontrol edin; yalnız sohbetin
“kaydedildi” demesi yeterli değildir.

Bu tool olay sınıflandırmaz. Kategoriyi agent seçer. `save_video_segment` ile
karıştırılmamalıdır: arşiv FFmpeg ile yeniden kodlar ve
`_stuff/lab_runs/actions/archive/<kategori>/<anahtar>/clip.mp4` + `metadata.json`
üretir.

Kategoriler: `hirsizlik`, `soygun`, `kavga_saldiri`, `trafik_kazasi`, `is_kazasi`,
`diger`, `belirsiz`.

| Senaryo | Beklenen |
| --- | --- |
| Geçerli aralık + geçerli kategori | `ok=true`, `cache_hit=false`; MP4 açılır; süre istenen aralıkla uyumludur; `metadata.json` aynı kategori/aralık/gerekçeyi taşır. Kaynak video değişmez. |
| Aynı kaynak/aralık/kategori tekrar | Yeni klasör yok; `cache_hit=true`; aynı `output_path`; ilk `explanation` korunur. Agent toolu yeniden çağırmadıysa cache testi yapılmış sayılmaz. |
| Aynı kesit, farklı kategori | `ok=false`, `ARCHIVE_CATEGORY_CONFLICT`; ikinci kategoriye kopya oluşmaz. |
| Listede olmayan kategori | `ok=false`, `INVALID_CATEGORY`. Model “benzer kategoriye kaydettim” diyemez. |
| Boş / 2000 karakteri aşan gerekçe | `ok=false`, `INVALID_ARCHIVE_DESCRIPTION`; klasör oluşmaz. |
| Bitiş video süresini aşıyor | Bitiş kırpılır, `END_TIME_CLAMPED`; başlangıç dışarıdaysa `TIME_OUT_OF_RANGE`. |
| Negatif/ters/sıfır süre | `ok=false`, `INVALID_TIME_RANGE`; çıktı üretilmez. |
| Kaynak yok | `ok=false`, `FILE_NOT_FOUND`; sahte `output_path` yok. |
| FFmpeg yok | `FFMPEG_NOT_FOUND`; başarı veya boş dosya yok. |
| Mevcut kaydın `clip.mp4` veya `metadata.json` dosyasını bozup aynı çağrıyı tekrarlayın | `ARCHIVE_CONFLICT`; üzerine yazılmaz. Eski bozuk dosya durur. |
| Kayıt sırasında kaynak videoyu değiştirin | `SOURCE_CHANGED`; bu işin kısmi klasörü temizlenir; önceki sağlam arşivler korunur. |
| `save_video_segment` ile karşılaştırma | Genel kayıt kullanıcı adıyla/copy kesimle oluşur; arşiv kategori klasöründe yeniden kodlanmış kliptir. Biri diğerinin yerine geçmez. |
| “Bu kesiti dışarı aktar / e-posta at” | Yalnız yerel arşiv; ağ/galeri/indirme alanı eklenmez. |
| Eşzamanlı iki farklı kesit | Kilit sıraya sokar; dosyalar karışmaz. Aynı kesit+kategori ikinci çağrıda cache veya çakışma döner. |
| Normal chat | Mevcut node sırası değişmez. Rapor modunda bu çağrı `eylemler` kaydına girer: [REPORT_ACTIONS_TEST_SCENARIOS.md](REPORT_ACTIONS_TEST_SCENARIOS.md). |

Klip süresini oynatıcıyla kontrol edin. Arşiv klibi istenen aralıkla uyuşmuyorsa
başarı sayılmaz. Bu testler modelin olay türünü doğru bildiğini kanıtlamaz.
