# Plaka bölgesi tespiti — manuel testler

Uygulamayı yeniden başlatın. Net plakalı kısa bir video kullanın.
Örnek: “Bu videonun 2–4 saniyesindeki plaka bölgelerini bul ve görüntülerini kırp.
Metnini okuma.” Node trace içinde `detect_license_plate_regions` çağrısını doğrulayın.

| Senaryo | Beklenen |
| --- | --- |
| Net plaka, geçerli aralık | `ok=true`, `crop_count>0`; PNG açılır ve plaka bölgesini içerir. |
| Konum/renk kontrolü | PNG kaynak kareyle aynı renklerde; kutu kaynak piksel koordinatlarında, görüntü dışına taşmaz. |
| Zaman kontrolü | Her `source_sec` istenen başlangıç dahil, bitiş hariç aralıkta; kareler atlanmaz. |
| Birden fazla plaka | Bulunan adayların ayrı PNG yolları olur; tekil araç sayısı iddia edilmez. |
| Aynı plaka birçok karede | Ayrı kırpımlar normal; takip veya tekilleştirme yapıldığı söylenmez. |
| Plaka olmayan/bulanık video | Hata yoksa `ok=true`; sıfır tespitte `NO_PLATE_DETECTED`; kesin yokluk iddiası yapılmaz. |
| 30'dan fazla kırpım | Özet 30 kayıt; `crops_truncated=true`, tümü `details_path` içinde. |
| Bitiş video süresini aşıyor | Bitiş kırpılır, `END_TIME_CLAMPED`; başlangıç dışarıdaysa hata. |
| Negatif/ters/sıfır süre | `ok=false`; çıktı üretilmez. |
| Model eksik / ONNX Runtime eksik | Açık hata; mevcut sohbet/diğer toollar çalışmaya devam eder. |
| Kare veya kırpım sınırı | `PLATE_FRAME_LIMIT` / `PLATE_CROP_LIMIT`; eksik sonuç başarı diye sunulmaz. |
| Çıktı yazılamıyor veya süre doluyor | `ok=false`; bu işin kısmi çıktıları temizlenir; önceki başarılı işler korunur. |
| Tekrarlı/eşzamanlı çağrı | Ayrı çıktı klasörleri; dosya üstüne yazma/başka videoya ait kırpım yok. |
| “Plaka numarası nedir?” | Tespit tool'u metin döndürmez; agent metin için ayrı `read_license_plate_crops` çağrısını kullanabilir. |
| Normal chat ve rapor | Mevcut node sırası/cevap akışı değişmez; rapor `eylemler=[]` kalır. |

Kırpımları gözle kontrol edin: şema doğruluğu modelin her plakayı bulduğunu kanıtlamaz.
