# Agent Cevap Akışı — Manuel Test Senaryoları

Kapsam: Planner → Executor → Tools → Reviewer, nihai cevap seçimi ve sohbet hafızası.
Yarışma raporu JSON'u ve modellerin doğruluk benchmark'ı bu belgenin dışında.

## Hazırlık ve başarı ölçütü

- Uygulamayı yeniden başlatın; bağımsız testlerden önce **sohbeti silin**.
- Süresi bilinen bir video ve mümkünse iki anomali aralığı üreten örnek kullanın.
- Takip sorularında aynı sohbeti koruyun. Yeni dosya isimleri kullanın; mevcut klipleri ezmeyin.
- Başarılı onayda sohbet cevabı, son tool çağrısı içermeyen executor cevabıdır; reviewer'ın iç `feedback` metni değildir.
- Trace'teki `[reviewer] feedback:` iç değerlendirmedir. Aynı bölümde nihai cevabın da bulunması normaldir.
- Trace önizlemeleri executor'da 600, tool'da 800 karakterde kesilebilir. Tam mesajı doğrulamadan veri/cevap kaybı sonucu çıkarmayın.
- Cümlelerin birebir aynı olması gerekmez; kanıt, kapsam ve sonuç doğru olmalıdır. Kritik kullanıcı testlerini farklı ifadelerle üç kez deneyin.

## Arayüzden uygulanabilen testler

| # | İstek / işlem | Beklenen sonuç |
|---|---|---|
| 1 | Video yüklemeden “Merhaba” deyin. | Doğal cevap; gereksiz tool çağrısı ve “Kullanıcının isteği tamamlandı” gibi iç değerlendirme yok. |
| 2 | “Videonun süresi ve çözünürlüğü nedir?” | Gerçek metadata değerleri cevapta yer alır; yalnız “bilgiler alındı” denmez. |
| 3 | “Videodaki anomalileri bul ve neler olduklarını açıkla.” | Segmenter aralıkları ve VLM açıklamaları kullanıcıya aktarılır. İki segment varsa ikisi de ele alınır veya eksik kalan açıkça belirtilir. |
| 4 | “3–7 saniye arasında ne oluyor?” | Geçerli aralığın görsel açıklaması verilir; tüm videonun incelendiği iddia edilmez. |
| 5 | Süresi 17 saniye olan videoda “40. saniyede ne oluyor?” | Süre dışı olduğu açıklanır; hayali olay veya “anomali yok” sonucu üretilmez. |
| 6 | Aynı videoda “12–25 saniyeyi incele.” | Yalnız geçerli bölüm yorumlanır; video sonunda sınırlandırıldığı cevapta belirtilir. |
| 7 | Segment üretmeyen videoda “Anomalileri açıkla.” | Mevcut eşikte segment bulunmadığı söylenir; kesin normal olduğu veya VLM'nin incelemediği olaylar iddia edilmez. |
| 8 | “3–7 saniyeyi cevap_akis_testi_01.mp4 olarak kaydet.” | Gerçekten oluşan, açılabilir klip için kaydetme sonucu verilir; reviewer'ın denetim notu gösterilmez. |
| 9 | “Videoda konuşulanları yazıya dök.” | Mevcut ses çözümleme yeteneği sınırı açıklanır; konuşma uydurulmaz, gereksiz döngü oluşmaz. |
| 10 | Analizden sonra “İlk bulduğun aralığı iki cümlede özetle.” | Önceki kullanıcı cevabındaki aralık esas alınır; iç denetim notu yanıt yerine kullanılmaz. |
| 11 | Farklı video yükleyip yeni videodaki bir aralığı sorun. | Tool çağrıları yeni dosyaya gider; eski videonun olayları yeni videoya mal edilmez. |
| 12 | Sohbeti silin; “Az önce hangi olayı bulmuştun?” deyin. | Eski sohbet sonucu hatırlanıyormuş gibi aktarılmaz. Yüklenmiş video kalabilir; sohbet temizlemek videoyu kaldırmak değildir. |
| 13 | İki ayrı tarayıcı oturumunda farklı videoları analiz edin. | Oturumların sohbet cevapları ve hedef videoları birbirine karışmaz. |
| 14 | Boş mesaj gönderin. | Yeni analiz/cevap eklenmez; mevcut sohbet korunur. |

## Kontrollü geliştirici senaryoları

Bu yolları yalnız prompt yazarak kesin tetikleyemezsiniz. İzole test ortamında sahte model/tool yanıtı veya debugger kullanın; gerçek servisleri bozmayın, anahtarları değiştirmeyin. Bu belge test kodu içermez.

| # | Hazırlanan durum | Beklenen sonuç |
|---|---|---|
| 15 | Executor gerçek cevabı versin; reviewer onaylayıp `feedback` içinde “Sonuçlar kullanıcıya iletildi” yazsın. | Sohbette executor cevabı, trace'te iç feedback görünür. Kalıcı `lc_messages` içine yalnız kullanıcı mesajı ve nihai cevap eklenir. |
| 16 | Reviewer önce `is_complete=false, route_to=executor`, düzeltmeden sonra onay versin. | Executor geri bildirime göre düzeltir; yalnız düzeltilmiş cevap yayınlanır. |
| 17 | Reviewer önce `route_to=planner` ile kapsamı reddetsin, sonraki değerlendirmede onaylasın. | Planner yeniden planlar, executor uygular; eski taslak yayınlanmaz. |
| 18 | Reviewer arka arkaya iki kez reddetsin. | Mevcut `MAX_REVIEW_LOOPS=2` ile ikinci rette durur; doğrulanmış nihai yanıt hazırlanamadığı bildirilir. Reddedilen taslak veya iç feedback yayınlanmaz. |
| 19 | Reviewer onaylasın ama son mesaj boş, ToolMessage veya tool çağrılı AIMessage olsun. | Onay tek başına yeterli sayılmaz; ilk sefer executor'a dönülür, sınırda güvenli duruş mesajı verilir. Eski cevap/ham tool JSON'u seçilmez. |
| 20 | Son executor içeriği metin ve `reasoning` türündeki ayrı bloklardan oluşsun. | Yalnız metin blokları kullanıcı cevabına alınır. Bu kontrol, düz metnin içine yazılmış değerlendirmeyi otomatik ayıklama garantisi değildir. |
| 21 | Agent sekiz tool turunu doldurup yeni tool çağrısı istesin. | Dokuzuncu tool turu çalışmaz; bekleyen çağrılar doğru `tool_call_id` ile kapatılır. Akış sınırlı sürede biter, yapılmayan iş başarılı gösterilmez. Tur sayısı, tekil tool çağrısı sayısı değildir. |
| 22 | Bir VLM tool'u hata versin; diğer aralık başarılı olsun. | Başarılı aralık ile başarısız aralık ayrılır. Eksik aralık için olay uydurulmaz; bütün analiz tamamlandı denmez. |
| 23 | LLM bağlantı hatası veya reviewer şema ayrıştırma hatası üretin. | Mevcut arayüz `[HATA]` sonucu verir; eski cevap başarı gibi gösterilmez. Uygulama açık kalır; hata kaldırıldıktan sonra yeni istek işlenebilir. |

## Sonuç kaydı

Her test için **Geçti / Kaldı / Uygulanmadı**, kullanılan video/istek, sohbet cevabı ve node trace'i kaydedin. Kontrollü koşul oluşmadıysa testi “Geçti” saymayın.

Bu liste beklenen davranışları tanımlar; canlı testlerin tamamının geçtiği iddiası değildir. Reviewer onayı, görsel yorumun kesin doğru olduğunu kanıtlamaz; olay iddialarını video ve tam tool çıktısıyla ayrıca karşılaştırın.
