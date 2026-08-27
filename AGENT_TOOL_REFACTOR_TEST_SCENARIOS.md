# Agent / Tool Refactor — Manuel Kabul Senaryoları

## Hazırlık

- Uygulamayı yeniden başlatın ve sohbeti temizleyin.
- Süresi bilinen bir video kullanın. Aşağıdaki örnekler 17 saniyelik video içindir;
  gerçek süreyi dikkate alın (oynatıcı süreyi yuvarlayabilir).
- Her bağımsız senaryoda sohbeti temizleyin; takip sorularında aynı sohbeti kullanın.
- Nihai cevapla birlikte node trace'i kontrol edin. Araç sırası zorunlu değildir;
  seçilen yolun kullanıcı hedefini karşılaması ve kanıta dayanması önemlidir.
- Tool JSON'u uzun olduğunda arayüz 800 karakterden sonrasını kesebilir. Bu,
  modelin de kesilmiş veri aldığı anlamına gelmez; tam JSON kontrolü için doğrudan
  tool çıktısı incelenmelidir.

## Kullanıcı üzerinden testler

| # | İstek / işlem | Beklenen sonuç |
|---|---|---|
| 1 | “Merhaba” | Gereksiz video analizi yapılmadan doğal cevap. |
| 2 | “Videonun süresi, FPS'i ve çözünürlüğü nedir?” | Metadata toolu seçilir; cevap gerçek değerlerle uyumludur. |
| 3 | “Videodaki anormal zaman aralıklarını bul.” | Segmenter çalışır; sonuçta `data.video`, eşik ve segment listesi bulunur. Boş liste kesin normal anlamına gelmez. |
| 4 | “3–7 saniye arasında ne oluyor?” | Geçerli aralıktan görsel kanıt alınır; bütün videonun incelendiği iddia edilmez. |
| 5 | “40. saniyede ne oluyor?” | Sürenin dışında olduğu açıklanır; olmayan zamana ilişkin gözlem üretilmez. |
| 6 | “40. saniyede anomali var mı?” | “Anomali yok” yerine zamanın videoda bulunmadığı söylenir. Segmenter seçilmişse metadata sonucu dikkate alınır. |
| 7 | “12–25 saniyeyi incele.” | Yalnız geçerli 12–video sonu aralığı incelenir; sınırlandırma açıklanır. Toola taşan bitiş verilirse `END_TIME_CLAMPED` uyarısı döner. |
| 8 | “3–7 saniyeyi refactor_deneme.mp4 olarak kaydet.” | Dosya oluşur ve açılır; başarıda `ok=true`, `error=null`, `data.output_path` bulunur. |
| 9 | “Anormal bölümleri bul ve oralarda ne olduğunu açıkla.” | Anomali tespiti ile görsel açıklama birbirini tamamlar; sonraki araç öncekinin bulduğu aralıkları kullanır. Segment yoksa hayali olay/aralık üretilmez. |
| 10 | Önce analiz, sonra “Bulduğun ilk aralığı kaydet.” | Önceki cevapta mevcut aralık kullanılır; aynı analiz gereksiz yere tekrarlanmaz. |
| 11 | Farklı bir video yükleyip aynı soruyu sorun. | Yeni video hedeflenir; eski videonun sonuçları yeni videoya mal edilmez. |
| 12 | “Videoda konuşulanları yazıya dök.” | Mevcut araçlarda ses çözümleme olmadığı belirtilir; konuşma metni uydurulmaz. |
| 13 | Aynı isteği farklı ifadelerle üç kez sorun. | Plan farklılaşabilir; yetenek, zaman ve kanıt sınırları korunmalıdır. |

## Geliştirici kontrolü — doğrudan tool / node çağrısı

| Durum | Beklenen |
|---|---|
| Dört toolun başarı ve hata çıktıları | JSON alanları `ok`, `data`, `warnings`, `error`; başarıda `error=null`, hatada kod ve mesaj bulunur. |
| VLM: başlangıç süreye eşit veya büyük | `TIME_OUT_OF_RANGE`; uzak VLM çağrısı yapılmaz. |
| VLM/kaydetme: negatif, ters veya eşit aralık | `INVALID_TIME_RANGE`; işlem yapılmaz. |
| VLM servis hatası | `ok=false`; hata başarılı analiz gibi raporlanmaz. |
| Eksik zorunlu parametre / olmayan tool adı | Hata da ortak JSON zarfında dönmeli; akış kontrolsüz kesilmemeli. |
| Tool tur sınırı | Çalıştırılmayan çağrı için ortak JSON hata zarfı ve doğru `tool_call_id` bulunmalı. |
| Reviewer son denemede de cevabı reddeder | Reddedilmiş cevap doğrulanmış gibi kullanıcıya gönderilmemeli. |
| Tool açıklaması / şeması değiştirilip uygulama yeniden başlatılır | Planner/Reviewer kataloğu kayıtlı tanımı yansıtmalı; system promptta elle tool listesi düzenlemek gerekmemeli. |

## İncelemede doğrulanan açıklar (27 Ağustos 2026)

- Reviewer sınırında reddedilmiş son cevap halen geri dönebiliyor.
- Eksik parametre/bilinmeyen tool hataları ve tool tur sınırı mesajı henüz ortak
  JSON sözleşmesini izlemiyor.
- VLM servis hatası halen yanıt içindeki `[VLM HATA]:` metni aranarak ayırt ediliyor;
  hata türü üzerinden aktarım yapılması daha güvenilir olur.

Bu belge beklenen davranışları tanımlar; bütün senaryoların geçtiği anlamına gelmez.
Canlı LLM/VLM ile uçtan uca kabul testleri ekip tarafından ayrıca uygulanmalıdır.
