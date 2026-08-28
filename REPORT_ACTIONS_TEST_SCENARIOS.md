# Rapor `eylemler` kaydı — manuel testler

Kapsam: Video Raporu sekmesi, `utils/action_records.py` ve `validate_report`.
Sohbet cevabı bu sözleşmeyi zorlamaz.

Uygulamayı yeniden başlatın. Raporu **Rapor oluştur** ile alın. Nihai JSON’daki
`eylemler` dizisini node trace’teki gerçek tool çağrı/sonuçlarıyla karşılaştırın.

Kayıt biçimi:

`[BASARILI|BASARISIZ] <tool_adı> (<tool_call_id>): <özet>`

Eylem tool’ları: `save_video_segment`, `archive_anomaly_clip`,
`detect_and_track_objects`, `detect_license_plate_regions`,
`read_license_plate_crops`. Segmenter, VLM ve `get_video_info` bu listeye girmez.

OCR kaydı ancak bu görevin başarılı `detect_license_plate_regions` çıktısındaki
`details_path` değeri `read_license_plate_crops` çağrısının `crops_manifest_path`
argümanıysa eklenir. Başarı için tool `ok=true` ve `data.video_path` hedef video
ile aynı gerçek yoldur.

| Senaryo | Beklenen |
| --- | --- |
| Yalnız segmenter + VLM (eylem tool yok) | `eylemler=[]` veya yalnız `[ONERI] ...`. Uydurma `[BASARILI]` yok. Rapor onaylanabilir. |
| Başarılı arşiv | Birebir `[BASARILI] archive_anomaly_clip (<id>): Klip kaydedildi; kategori=...; {start}–{end} sn; dosya=<output_path>`. Dosya vardır. |
| Aynı arşiv cache | Özet “Mevcut kayıt kullanıldı” ile başlar; yeni kopya yok. |
| Başarısız arşiv | `[BASARISIZ] archive_anomaly_clip (<id>): İşlem tamamlanmadı: <kod>`. Görsel kapsam tamamsa rapor üretilebilir. |
| Plaka tespiti raporda çalıştı | `[BASARILI] detect_license_plate_regions (<id>): Plaka bölgesi taraması tamamlandı; kırpım=...; kayıt=<details_path>`. Sıfır kırpımda da `ok=true` ise BASARILI’dır. |
| Tespit + OCR aynı raporda | Önce tespit, sonra OCR; OCR `crops_manifest_path` bu görevin `details_path` değeri. Sıra tool sırasıyladır. OCR özeti `ocr_performed=true` iken işlenen/okunan sayılarını taşır. |
| OCR var, bu görevin tespiti yok / başka klasör | OCR kaydı `eylemler`e girmez. |
| OCR kırpım yok (`ocr_performed=false`, `ok=true`) | `[BASARILI] read_license_plate_crops (<id>): Okunacak kırpım bulunamadı; OCR çalıştırılmadı`. |
| Nesne takibi | `[BASARILI] detect_and_track_objects`; kutu sayısı tekil nesne değildir. Kutulu MP4 varsa yolu özettedir. |
| Başka videoya ait tool sonucu | Hedef videonun listesine girmez. |
| Eşleşmeyen/yarım tool mesajı | Listeye alınmaz. Çalıştırılmamış çağrı BASARILI olamaz. |
| Uydurma başarı / eksik kayıt / sıra değişik | `validate_report` reddeder; rapor indirilmez. |
| `[ONERI] ` ile ek satır | 1–2000 karakter, boş olmayan öneri kabul; tool çalıştırmaz. Gerçek kayıtlar aynen durur. |
| `[ONERI]` boş veya önek yok | Doğrulama hatası; rapor sunulmaz. |
| Normal Agent sohbeti | JSON `eylemler` zorlanmaz. |
| Eylem hatası, görsel analiz eksik | Rapor yine reddedilir (kapsam kuralı). |

Trace kesilmiş olabilir; tam sonucu `details_path` / `output_path` dosyalarından
doğrulayın. Bu testler olay yorumunu kanıtlamaz; kayıt uydurulmadığını kontrol eder.
