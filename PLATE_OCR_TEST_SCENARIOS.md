# Plaka OCR — manuel testler

Uygulamayı yeniden başlatın. Net plakalı kısa videoda önce bölgeleri kırptırın,
sonra “Bulduğun plaka kırpımlarındaki numaraları oku; emin olmadıklarını belirt.” deyin.
Trace: `read_license_plate_crops`, önceki tespitin `details_path` değerini almalı.

| Senaryo | Beklenen |
| --- | --- |
| Net okunabilir plaka | Görselle `text` eşleşir; güvenli tahminde `status=read`. Gözle doğrulayın. |
| Bulanık/küçük plaka | Doğruluk garanti edilmez; düşük skor `uncertain`, boş/bozuk karakter dizisi `unreadable`; `text=null`. |
| Yüksek güven | `min_slot_confidence` skor olarak sunulur; “kesin doğru” veya kimlik doğrulaması denmez. |
| Tekrar eden plaka | Kare bazında ayrı okumalar; araç sayısı veya aynı araç olduğuna dair kesin iddia yok. |
| 30'dan fazla kırpım | `processed_crop_count` tüm kayıtları kapsar; özet 30, `results_truncated=true`; dosyada tamamı. |
| Tespit tekrarını kontrol | OCR tool'u video okuyucusu/tespit modeli çalıştırmaz; mevcut PNG'leri kullanır. |
| Zaman/koordinat | Her okumanın saniyesi/kutusu ilgili kırpımla aynı. |
| Plaka tespiti sıfır | `ok=true`, `ocr_performed=false`, `NO_CROPS`; uydurma numara yok. |
| Model/yapılandırma eksik | `OCR_MODEL_MISSING`; diğer toollar çalışır. |
| Yanlış model/yapılandırma | Hata döner; rastgele karakterler başarılı çıktı diye sunulmaz. |
| Kırpım silinmiş | `OCR_CROP_MISSING`; kısmi sonuç başarı sayılmaz. |
| PNG değiştirilmiş | Yeni hash'li kayıtta `OCR_CROP_CHANGED` veya boyut/bozulma için `OCR_INVALID_CROP`. |
| Eski hash'siz kayıt | Uyumlu dosyalar okunur; `LEGACY_CROPS_UNVERIFIED` uyarısı. |
| Yanlış dosya yolu/başka klasörde PNG | `INVALID_CROP_MANIFEST`; rastgele yerel dosya OCR'a verilmez. |
| Kırpım/süre sınırı | `OCR_CROP_LIMIT` / `OCR_TIMEOUT`; eksik sonuç başarılı raporlanmaz. |
| Kaynak video silinmiş | PNG ve manifest sağlamsa OCR çalışır; videoyu tekrar istemez. |
| Boş sohbet/normal rapor | Mevcut akış değişmez; raporun `eylemler` alanı bu aşamada hâlâ `[]`. |

Bu testler OCR'ın tüm Türkiye plakalarında doğru olacağını kanıtlamaz; gerçek
gece/gündüz, açı ve çözünürlük çeşitliliğiyle ayrıca doğruluk değerlendirmesi gerekir.
