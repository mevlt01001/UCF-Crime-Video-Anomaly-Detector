# Manuel test planı — genelden özele

**Durum: tüm senaryolar CALISTIRILMADI.** Bu belge uygulama talimatıdır, test sonuç raporu değildir.

## Hazırlık ve kayıt disiplini

**Nasıl uygulayacağım?** [Adım adım rehberi](05_ADIM_ADIM_UYGULAMA.md) açın. Bölüm 1–2 ilk video ve kanıt kaydını, bölüm 3 G senaryolarını, bölüm 4 U senaryolarını, bölüm 5 R/E/Z senaryolarını, bölüm 6 araç testlerini, bölüm 7 X/T/Q ve sonuç kaydını anlatır. Aşağıdaki her senaryonun kabul koşulu bu uygulama adımlarıyla birlikte kullanılmalıdır. Arayüzden üretilemeyen hata koşulları PASS sayılmaz; kontrollü otomatik test veya izole ortam gerekir.

- Gerçek videoların orijinallerini salt okunur kabul edin; çalışma kopyası kullanın. Kaynak hash'ini işlem öncesi/sonrası kaydedin. Her bozma/silme/izin testi yalnız disposable kopya, geçici çıktı dizini ve ayrı süreç/ortamda yapılır.
- Ana arayüz React için `http://127.0.0.1:8000/`; Gradio karşılaştırması `/gradio`, ayrı `lab.py` çalıştırıldıysa `7860`. Hangi URL/PID/build test ediliyor mutlaka yazın.
- Rapor sekmesinde yalnız video ve Rapor oluştur vardır; adım adım prompt giremezsiniz. Tool'u zorunlu seçtirmek gereken kontroller Sohbet veya izole otomatik test üzerinden yapılır. Bu kontroller raporun otomatik karar mekanizmasının geçtiğini göstermez.
- P0: yanlış güvenlik/olay/kimlik sonucu, bağlam karışması veya raporun çalışmaması. P1: araç/medya/artifact ve işletim doğruluğu. P2: eğitim ve yardımcı girişler. T01 veri ayrımı hatası, model kalitesi iddiası için yine engeldir.
- Her senaryoda yazılı tüm beklentiler karşılanmalı; bir alt durum eksikse ayrı run satırı açın. CSV'de aynı ID ile yeni run kaydı kullanılabilir. Hiç çalıştırılmayan senaryo PASS olamaz.
- Kontrollü hata için üretim servislerini bozmayın, gerçek .env'yi değiştirmeyin. Gerekli sahte servis/ortam kurulmadıysa BLOCKED yazın. Otomatik test dosyaları bu durumların bir kısmını ağsız üretir.

## Test video kataloğu

Bunlar **hazır oldukları iddia edilen dosyalar değil**, hazırlanacak/seçilecek fixture tanımlarıdır. `veri_kaydi.json` içine gerçek yerel yol/hash ve insan etiketi eklenir; özel videolar Git'e girmez.

| Veri | İçerik / amaç | İnsan referansı |
|---|---|---|
| V01 | Kullanıcının WhatsApp Video 2026-08-27 at 22.47.15.mp4 örneği; 22.4 sn, 576×1024, araç yakınlaşması | 0–22.4 sn tamamı izlenmeli; 2–8 ve 16–19 sn ayrıntılı. Çarpışma/niyet peşinen etiketlenmez. |
| V02 | 03:36 başarısız denemesindeki WhatsApp ...03.26.34.mp4 | Gerçek dosya/hash temin edilmeli; ekran görüntüsü video dosyası yerine geçmez. |
| V03 | Sentetik daire/kare ve RGB şeritleri; 9:16, 16:9, 1:1 ve rotation metadata sürümü | Geometri ve renk referansı kesin; olay değerlendirmesinde kullanılmaz. |
| V04 | Normal, kısa, net video; araçsız sürüm de olsun | İnsanlarca belirgin olay görülmeyen kapsam; mutlak güvenlik garantisi değil. |
| V05 | En az üç ayrı olay aralığı ve arada normal bölüm | Her olayın start/end, tür, belirsizliği önceden etiketlenir. |
| V06 | Araçsız net olay + aynı olayın belirsiz/örtülü sürümü | Kategori/görsel gerekçe; suç veya kişi niyeti için yalnız görünür kanıt. |
| V07 | Tek araç, net okunabilir plaka, kısa olay | Araç kutuları ve plaka kırpımı insan etiketi; sentetik plaka kullanılabilir. |
| V08 | En az iki araç; yalnız biri olayla ilgili; gündüz/gece/bulanık varyant | A/B araç ayrımı, görünme aralıkları, belirsiz ilişki kareleri. |
| V09 | Örtüşme, görüntüden çıkış/yeniden giriş ve ID değişimi | Nesne sürekliliği yalnız açık karelerde; görünmeyen arada kesin kimlik yok. |
| V10 | Zaman işaretli CFR/VFR, sesli/sessiz ve sıfır dışı kesit | Kaynak PTS listesi, belirgin ses/görüntü eşleşme noktası. |
| V11 | Klip boyu sınırlarında kısa video, sonda kısmi pencere ve son kare olayı | Gerçek/örneklenmiş kare sayısı; event'in kesin zaman işareti. |
| V12 | Uzun/kalabalık; kare, crop, preview ve tool tur sınırlarını zorlayan video | Uzunluk, beklenen sınır ve bu test için izin verilen CPU/RAM/süre. |
| V13 | Boş, sahte uzantılı, yarım veya test sırasında değiştirilen kopya | Her varyant disposable; gerçek kaynaklar korunur. |

## Her denemede saklanacak kanıt

Run ID; commit; UI/build/URL; işletim sistemi/cihaz; model ve config dosya hash'leri; gizli olmayan etkin ayarlar; video hash/süre/FPS/boyut; session/job; kullanıcı girdisi; **tam trace (parsed tool sonuçları dahil)**; ham/kanonik JSON; çıktı yolu/hash; ilgili kaynak ve çıktı kareleri; süre/RAM; PASS/FAIL/BLOCKED/INCONCLUSIVE gerekçesi.

Log preview'ı kırpılmış olabilir; gerçek kanıt için parsed alanı ve details_path dosyaları gerekir. Anahtarları, base64 videoyu ve gereksiz özel plaka verisini paylaşılan test raporuna koymayın.

## Senaryolar


## 1 — Kritik genel rapor akışı


### MAN-G01 · P0 — Dikey trafik videosunda yön, yakınlaşma ve görüntü oranı

**Veri/önkoşul:** V01 + V03.

**Uygulama:** V01'i orijinal oynatıcıda 0–22.4 sn izleyin; özellikle 2–8 ve 16–19 sn'yi yavaşlatın. Yeni sohbet → yükle → Rapor oluştur. Aynı gözlemleri 2–8 sn sorusuyla Sohbet'te karşılaştırın. V03 geometri referansını VLM girdisinde ayrıca kontrol edin.

**Beklenen / geçiş koşulu:** Araç ön/arka yüzü ve hareket yönü görüntüyle uyuşmalı; güvenli mesafe, çarpışma, kasıt gibi kanıtsız kesinlik olmamalı. Dikey içerik yatay esnetilmemeli. Segment yoksa “kesin normal” denmemeli.

**Kanıt ve başarısızlık değerlendirmesi:** Kaynak kareler, gönderilen n/shape/FPS marker'ı, tüm parsed VLM metni ve JSON. KG03 ve OBS01; mevcut sürümde başarısızlık beklenebilir.


### MAN-G02 · P0 — Sıfır segmentte görsel doğrulama

**Veri/önkoşul:** V01 veya segment üretmeyen V04.

**Uygulama:** Rapor oluşturun. Trace'te segment_count=0 olduğunu doğrulayın. Bundan sonra VLM çalışıp çalışmadığını ve hangi aralıkları kapsadığını kontrol edin; aynı raporu üç bağımsız oturumda deneyin.

**Beklenen / geçiş koşulu:** Boş segmentten doğrudan güvenli/düşük risk raporu çıkmamalı. Kısa video için tamamını kapsayan görsel kontrol veya tamamlanamadı sonucu gerekli. Uzun videoda açıklanmış parçalı kapsam gerekir.

**Kanıt ve başarısızlık değerlendirmesi:** Segmenter uyarısı, VLM çağrı aralıkları ve nihai JSON. Hiç VLM olmadan onay FAIL; KG02.


### MAN-G03 · P0 — Video yüklü ama reviewer “yol eksik” diyor

**Veri/önkoşul:** V02.

**Uygulama:** Yeni sohbet açın; dosya adını ve oynatılabilir önizlemeyi doğrulayın; yalnız Rapor oluştur'a basın. Video yolunu kullanıcı mesajıyla elle eklemeyin. Üç bağımsız denemenin trace'ini saklayın.

**Beklenen / geçiş koşulu:** Tüm düğümler aynı yüklenen dosyayı hedeflemeli. Kullanıcıdan tekrar yol istememeli. İzinli debug kaydı varsa her düğümün prompt'unda hedefi kontrol edin; API anahtarını kaydetmeyin.

**Kanıt ve başarısızlık değerlendirmesi:** 03:36 iziyle karşılaştırma, session/job kimliği, çağrı args. Reviewer'ın yol eksik varsayımı FAIL; KG01. İlk executor atlamasının nedeni ayrı araştırılır.


### MAN-G04 · P0 — Araç çalışmadan başarısızlık/düşük risk taslağı

**Veri/önkoşul:** V02/V04.

**Uygulama:** Rapor isteyin; executor metadata alınamadı diyorsa önceki tool satırlarını sayın. Doğal denemede oluşmazsa kontrollü otomatik testle aynı taslağı üretin; yeniden planlamanın eksik araç adımını yaptığını kontrol edin.

**Beklenen / geçiş koşulu:** Bir tool hatası olmadan dosya bozuk/metadata hatası iddiası yok. Segmentasyon eksikse rapor indirilemez. Düzeltme aynı hedefle aracı çalıştırmalı veya gerçek sınırla durmalı; sonsuz döngü yok.

**Kanıt ve başarısızlık değerlendirmesi:** Executor taslağı ile gerçek çağrı/sonuç eşleştirmesi. Bu hata doğal oluşmadıysa kurtarma alt testi BLOCKED, PASS değil.


### MAN-G05 · P0 — Deneme sınırında yanlış başarı etiketi

**Veri/önkoşul:** V02 veya kontrollü başarısız rapor.

**Uygulama:** İki denetim reddinden sonra trace başlığını, sağ üst durumu, JSON alanını ve indirme bağlantısını birlikte inceleyin.

**Beklenen / geçiş koşulu:** Sonuç başarısız/tamamlanamadı olmalı; “final answer approved” görünmemeli. Eski rapor linki kalmamalı. Sınır mesajı doğrulanmış rapor değildir.

**Kanıt ve başarısızlık değerlendirmesi:** Ekran görüntüsü + reviewer feedback/final_answer + job_error/done; KG04.


### MAN-G06 · P0 — Çok segmentli raporda tüm kapsam

**Veri/önkoşul:** V05.

**Uygulama:** İnsan referansında en az üç ayrı olay aralığı belirleyin. Raporu çalıştırın; segmenter'ın döndürdüğü her aralığı VLM effective_range birleşimiyle karşılaştırın. Son segmenti özellikle kontrol edin.

**Beklenen / geçiş koşulu:** Bulunan tüm aralıklar kapsanmalı. Bir aralık eksik/başarısızsa tam rapor yayımlanmamalı. Kaynak zamanları taşmamalı; olayın başlangıcı bilinmiyorsa kesit başlangıcı olduğu yazılmalı.

**Kanıt ve başarısızlık değerlendirmesi:** Aralık eşleme tablosu ve tam trace; yalnız ilk segmentin anlatılması FAIL.


### MAN-G07 · P0 — Doğrulanmış olay, belirsiz kategori

**Veri/önkoşul:** V06.

**Uygulama:** Rapor oluşturun. Aynı olay için net görüntü ve bulanık/örtülü sürümü karşılaştırın; kullanıcı mesajına suç türünü peşinen yazmayın.

**Beklenen / geçiş koşulu:** Kategori yalnız görsel kanıtla seçilmeli. Tür anlaşılmıyorsa belirsiz; bilinen fakat listede olmayan olay için diger. Anomali skoru suç olasılığı diye kullanılmamalı.

**Kanıt ve başarısızlık değerlendirmesi:** İnsan etiketleri, kategori gerekçesi, arşiv çağrısı varsa category/explanation. Kesin suç/kimlik uydurulursa FAIL.


### MAN-G08 · P0 — Normal videoda gereksiz eylemler

**Veri/önkoşul:** V04.

**Uygulama:** Araç içermeyen normal bir kısa videoya Rapor oluştur deyin. Trace ve eylemler listesini karşılaştırın.

**Beklenen / geçiş koşulu:** Hayali olay/plaka/kimlik ve yapılmamış kayıt yok. Sırf katalogda var diye tüm araçlar çalışmamalı. Eylem değerlendirme kaydı henüz yoksa “hiç çağrılmadı” bunun değerlendirildiğini kanıtlamaz.

**Kanıt ve başarısızlık değerlendirmesi:** Çağrı sayıları, dosya list değişim, JSON. Hedef seç/atla kaydı Z01 altında ayrıca test edilir.


## 2 — Oturum, kullanıcı arayüzü ve iş yaşam döngüsü


### MAN-U01 · P0 — A videosu sonuçları B videosuna taşınmıyor

**Veri/önkoşul:** V01 + V04.

**Uygulama:** A'yı raporlayın; rapor/graph/linki kaydedin. Video seçicinin kilitli olduğunu doğrulayın. Yeni sohbet → B yükle. Analiz öncesi ve sonrası ekranları karşılaştırın.

**Beklenen / geçiş koşulu:** Yeni session, boş geçmiş/rapor/grafik; tüm yeni tool args B'ye ait. Eski job event'i yeni ekrana yazılmamalı. Doğrudan ikinci yükleme aynı session'a yapılırsa 409.

**Kanıt ve başarısızlık değerlendirmesi:** İki session/job ve video yolu; eski linkin görünmesi veya yanlış araç sonucu P0 FAIL.


### MAN-U02 · P0 — İki bağımsız sekmede context ayrımı

**Veri/önkoşul:** V01 + V06.

**Uygulama:** İki yeni sekmede farklı video yükleyin. Birine yalnız onda bulunacak test kelimesi yazın; diğerinde önceki konuşmayı sorun. Raporları eşzamanlı başlatın.

**Beklenen / geçiş koşulu:** Session'lar farklı; prompt/cevap/tool sonucu karışmaz. Aynı track_id=1 iki işte oluşabilir ama tek araç diye birleştirilemez.

**Kanıt ve başarısızlık değerlendirmesi:** Sekme ekranları, session/job ve tool video_path'leri; gerçek kişi verisi yerine test kelimesi kullanın.


### MAN-U03 · P0 — Çift tıklama ve sekmeler arası eşzamanlı başlatma

**Veri/önkoşul:** V05.

**Uygulama:** Rapor oluştur'a hızlı çift tıklayın; ardından Analyzer ve Sohbet sekmelerinden iş başlatmayı deneyin. Aynı session'a ikinci API isteğini geliştirici kontrolüyle gönderin.

**Beklenen / geçiş koşulu:** Arayüz ikinci işi engeller; sunucu 409 döndürür. Tek işlem; önceki işi görünmez bırakıp yenisini izlemeye başlamaz.

**Kanıt ve başarısızlık değerlendirmesi:** Ağ istek sayısı, job_id'ler, CPU/model çağrı sayısı; 409 hata sonrası arayüz kullanılabilir.


### MAN-U04 · P0 — Analyzer iptali

**Veri/önkoşul:** V05.

**Uygulama:** Analyzer'ı başlatın, çalışırken İptal'e basın. Model çağrısı sürerken yeni sohbet/yükleme kontrolünü ve iş bitince çıktıyı izleyin.

**Beklenen / geçiş koşulu:** İptal isteği durumu dürüstçe görünür; native çağrı anında durdu denmez. Sonuç iptalden sonra yayımlanmaz; job_cancelled ve tek terminal done; kilit sonunda açılır.

**Kanıt ve başarısızlık değerlendirmesi:** İptal zamanı, son event, boş/korunan önceki çıktı; yeni Analyzer isteği tekrar çalışmalı.


### MAN-U05 · P0 — Rapor/sohbet iptali ve yan etkiler

**Veri/önkoşul:** V05.

**Uygulama:** VLM çalışırken raporu iptal edin; ayrı denemede arşiv çağrısı başlamışken iptal edin. Sonra yeni sohbet açın.

**Beklenen / geçiş koşulu:** İptal edilen taslak rapor veya sohbet geçmişe eklenmez. Önceden tamamlanmış ya da devam eden yerel arşiv işlemi geri alınmış gibi gösterilmez; iptal rollback değildir.

**Kanıt ve başarısızlık değerlendirmesi:** Trace, yeni sohbet geçmişi, iptal öncesi/sonrası dosya listesi. Dosya üretildiyse kaydı saklayın; otomatik silmeyin.


### MAN-U06 · P0 — SSE kopması ve yeniden bağlanma

**Veri/önkoşul:** V05.

**Uygulama:** İş sürerken tarayıcı geliştirici araçlarından kısa süre offline yapıp geri alın; ayrı denemede sunucuyu yeniden başlatın. Yeni sekmeye geçmek ile aynı SSE bağlantısını yeniden kurmayı ayırın.

**Beklenen / geçiş koşulu:** Sonuç kaybolmamalı/başka işe taşınmamalı; bitmiş iş sonsuza kadar çalışıyor kalmamalı. Sunucu job'ı unutmuşsa açık hata ve yeni sohbet yolu olmalı. Yinelenen event çift sonuç üretmemeli.

**Kanıt ve başarısızlık değerlendirmesi:** Network/SSE kaydı, son durum ve JSON dosyası. EventSource tekrar bağlandı diye sonuç geri alındı varsaymayın.


### MAN-U07 · P0 — Sayfa yenileme ve başlatma hatası

**Veri/önkoşul:** V04.

**Uygulama:** Yükleme sonrası sayfayı yenileyin; sonra sunucu kapalıyken Rapor/Sohbet başlatmayı deneyip yeniden açın.

**Beklenen / geçiş koşulu:** Mevcut tasarımda yenileme yeni session açar ve eski görünmeyen context'i kullanmaz. Fetch hatası sahte kullanıcı/başarı izi bırakmamalı; yeni istek mümkün olmalı. Eski sunucu işi otomatik iptal oldu varsayılmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Önce/sonra session ve UI durumları. “Geçmişi geri yükleme” mevcut özellik değildir.


### MAN-U08 · P0 — Gradio ile React farkı ve eski LLM/VLM hafızası

**Veri/önkoşul:** V01 + V04.

**Uygulama:** React ile /gradio Agent/Rapor sonuçlarını aynı ayarlarla karşılaştırın. Gradio LLM/VLM'de geçmiş açıkken iki tarayıcı oturumu kullanın; diğerinin geçmişini temizleyin.

**Beklenen / geçiş koşulu:** API oturum ayrımı Gradio'da otomatik var kabul edilmez. Diğer kullanıcıya context sızarsa FAIL. Rapor modunda takip sorusu yok; normal sohbet çalışmalı.

**Kanıt ve başarısızlık değerlendirmesi:** Arayüz/URL kesin kaydedilsin. Eski global Gradio yöneticileri için başarısızlık çıkabilir; React sonucu bunu kapatmaz.


## 3 — Rapor kanıtı ve çıktı sözleşmesi


### MAN-R01 · P0 — Saf JSON, tek indirme ve alanlar

**Veri/önkoşul:** V04/V05.

**Uygulama:** Raporu oluşturup JSON'u indirin; ekrandaki nesneyle karşılaştırın. Özet, olaylar, risk ve eylemleri insan referansıyla inceleyin.

**Beklenen / geçiş koşulu:** Yalnız ozet/olaylar/risk_seviyesi/eylemler; olayda saniye/aciklama. Tek tamamlanmış nesne; Markdown veya takip sorusu dosyada yok. Risk yalnız dusuk/orta/yuksek.

**Kanıt ve başarısızlık değerlendirmesi:** İndirilen dosya, ekran JSON'u ve parse sonucu; şema geçişi anlam doğruluğu değildir.


### MAN-R02 · P0 — Görsel kapsam ve kaynak zaman

**Veri/önkoşul:** V05/V10.

**Uygulama:** Segment ve VLM aralıklarını bir tabloda eşleyin. Sıfır dışı başlangıçlı kesitte olay zamanı ile orijinal oynatıcı zamanını karşılaştırın.

**Beklenen / geçiş koşulu:** Klip 0. saniyesi kaynak 0 diye sunulmaz. Bitiş video süresinde kırpılır; görsel kapsamdaki boşluk kapanmış sayılmaz. Boş segment politikası G02 ile birlikte sağlanmalı.

**Kanıt ve başarısızlık değerlendirmesi:** requested/effective/sampled_range + kaynak kare zamanları. VFR için ortalama FPS hesabını tek referans yapmayın.


### MAN-R03 · P0 — Kısmi VLM hatası

**Veri/önkoşul:** V05 + test servisinde kontrollü hata.

**Uygulama:** Bir görsel çağrı başarılı, diğeri 500/timeout olacak kontrollü servis kullanın; gerçek servisi veya üretim ağını bozmayın.

**Beklenen / geçiş koşulu:** Başarılı aralık kaybolmaz ama tüm analiz tamamlandı/riski düşük denmez. Eksik zorunlu görsel kapsam varsa rapor indirilmez; düzeltme yalnız eksik kısmı hedefler.

**Kanıt ve başarısızlık değerlendirmesi:** Hangi çağrı hata verdi, yeniden çağrı sayısı, final durum. Kontrollü servis yoksa BLOCKED.


### MAN-R04 · P0 — Sarmal JSON ve gerçekten bozuk JSON ayrımı

**Veri/önkoşul:** Kontrollü model yanıtı / otomatik test.

**Uygulama:** Tek fenced JSON, json etiketi, açıklama+tek blok; sonra yarım JSON, iki blok, ek alan, NaN/negatif saniye örneklerini kullanın.

**Beklenen / geçiş koşulu:** Bilinen sunum sarmalı temizlenebilir; yarım/çoklu nesne ve geçersiz şema onaylanmaz. Biçim düzeltmesi uğruna başarılı video araçları gereksiz tekrarlanmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Ham taslak, doğrulama hatası ve kanonik JSON. UI'den modelin bozuk üretmesini zorlamak yerine mock alt testi kullanılır.


### MAN-R05 · P0 — Kanıtın gerçek çağrıya ait olması

**Veri/önkoşul:** Kontrollü tool mesajları.

**Uygulama:** Eşleşmeyen tool_call_id, yanlış tool adı, başka video, yalnız içerik alanlarında vlm_response bulunan mesajları doğrulayıcıya verin.

**Beklenen / geçiş koşulu:** Sahte/bağlantısız görsel veya segment mesajları kanıt olamaz. Yalnız hedef videonun gerçek çağrı-sonuç çifti kabul edilmeli.

**Kanıt ve başarısızlık değerlendirmesi:** KG05 gap testi; üretim logunu düzenleyip başarı kanıtı diye sunmayın. Mevcut uygulamada eksik kontrol var.


### MAN-R06 · P0 — Yüksek skor ve gerçek olay riski ayrımı

**Veri/önkoşul:** V04 + V06.

**Uygulama:** Normal fakat hareketli sahne ile açık tehlike içeren sahneyi karşılaştırın; aynı/yakın skor olsa da risk gerekçesini inceleyin.

**Beklenen / geçiş koşulu:** Skor 0.99 diye suç/şiddet kesin denmez; “az segment” diye düşük risk seçilmez. Görsel bulgu, kapsam ve belirsizlik gerekçeye dahil edilir.

**Kanıt ve başarısızlık değerlendirmesi:** Skor grafiği, VLM kanıtı, risk açıklaması; insan referansı gerekir.


### MAN-R07 · P0 — Görünmeyen plaka/kimlik/ses iddiaları

**Veri/önkoşul:** V07/V08.

**Uygulama:** Rapor ve Sohbet'te konuşma dökümü, araç sahibinin kimliği, küçük plakayı kesin okuma gibi istekleri deneyin.

**Beklenen / geçiş koşulu:** Mevcut yetenek sınırı belirtilir. VLM'nin metinsel plaka tahmini OCR olarak sunulmaz; OCR de sahip/kimlik doğrulamaz. Sesli kaynak otomatik transkripsiyon demek değildir.

**Kanıt ve başarısızlık değerlendirmesi:** Çağrılan araçlar ve iddialar; uydurma transkript/kimlik P0 FAIL.


### MAN-R08 · P0 — Rapor ve sohbet geçmişi bağımsızlığı

**Veri/önkoşul:** V04.

**Uygulama:** Önce sohbette yanlış bir olay varsayımı söyleyin; ardından bağımsız Rapor oluşturun. Rapor sonrası sohbet geçmişini kontrol edin.

**Beklenen / geçiş koşulu:** Rapor önceki sohbet önyargısını otomatik kanıt saymaz; tool/reviewer mesajları kullanıcı geçmişine dolmaz. Rapor JSON'u takip sohbeti başlatmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Önce/sonra geçmiş, rapor başlangıç state'i (geliştirici kontrolü), nihai JSON.


## 4 — Eylem seçimi ve gerçek işlem kayıtları


### MAN-E01 · P0 — Gerçek arşiv başarısı kayda giriyor

**Veri/önkoşul:** V06.

**Uygulama:** Görsel olay arşivlendiği bir rapor üretin; otomatik seçim gerçekleşmezse Z01'i BLOCKED işaretleyip ayrı Sohbet tool denemesi yapın.

**Beklenen / geçiş koşulu:** Rapor içinde gerçekleşen archive_anomaly_clip kaydı gerçek call_id, kategori, aralık ve output_path ile birebir olmalı. Sohbette ayrı yapılan eylem o raporda yapılmış gibi yazılmamalı.

**Kanıt ve başarısızlık değerlendirmesi:** Tool args/result + açılan clip.mp4 + metadata + rapor eylemler. Ayrı sohbet başarısı rapor zinciri PASS değildir.


### MAN-E02 · P0 — Eylem hatası, tamamlanmış görsel analizi bozmuyor

**Veri/önkoşul:** V06 + test ortamında FFmpeg/model eksikliği.

**Uygulama:** Görsel analiz başarılıyken arşiv veya OCR adımını kontrollü olarak başarısız yapın.

**Beklenen / geçiş koşulu:** Gerçek başarısız eylem BASARISIZ kaydıyla görünür. Zorunlu analiz tamamsa rapor üretilebilir; eylem hatası “anormallik yok”a dönüştürülemez.

**Kanıt ve başarısızlık değerlendirmesi:** Hata kodu, diğer başarılı adımlar, nihai eylemler; kısmi dosya başarı diye gösterilmez.


### MAN-E03 · P0 — Kayıt uydurma, eksiltme ve sıra değiştirme

**Veri/önkoşul:** Kontrollü taslak.

**Uygulama:** Gerçek başarı satırını silin/değiştirin, olmayan BASARILI ekleyin veya sonuç sırasını ters çevirin; doğrulama testini çalıştırın.

**Beklenen / geçiş koşulu:** Kod taslağı reddeder. İnsan tarafından değiştirilmiş JSON gerçek tool çıktısı yerine geçmez; aynı sonuç mesajı iki kez gelince kayıt çoğalmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Otomatik ReportContracts; manuel raporda trace ile birebir karşılaştırma.


### MAN-E04 · P0 — OCR provenance: aynı raporun manifesti

**Veri/önkoşul:** V07 + V08.

**Uygulama:** A raporundaki tespitin details_path'ini kontrol edin; OCR'un aynı manifesti aldığını doğrulayın. Ayrı B videosunun manifestiyle kontrollü negatif deneme yapın.

**Beklenen / geçiş koşulu:** OCR kaydı yalnız aynı görevin başarılı tespitinden türeyebilir; yanlış videonun plakası A'nın eylemlerine karışmaz. Başarısız tespit sonrası OCR çalıştırılmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Tespit call_id, details_path, OCR crops_manifest_path/source_manifest_path ve data.video_path.


### MAN-E05 · P0 — Öneri, uygulanmış eylem değildir

**Veri/önkoşul:** V06.

**Uygulama:** “Arşivlenen klip insan denetimine gönderilsin” veya başarısız ikinci kesite ilişkin öneri bulunan raporu kontrol edin.

**Beklenen / geçiş koşulu:** ONERI satırı bir tool çalıştırmaz ve yapılmış işlem diye anlatılmaz. Benzer sözcük nedeniyle geçerli öneri reddedilmez; aynı hedefte başarılı işlemin gereksiz tekrarı semantik olarak denetlenir.

**Kanıt ve başarısızlık değerlendirmesi:** Rapor ve çağrı listesi. Önceki arşivle sözcük eşleştirmesi regresyonu; gerçek kayıtlar korunmalı.


### MAN-E06 · P0 — Başarılı boş sonuç ile yapılmamış işlem ayrımı

**Veri/önkoşul:** V04.

**Uygulama:** Plaka tespiti sıfır kırpım döndürsün; aynı görevde OCR çağrısı varsa çıktısını kontrol edin.

**Beklenen / geçiş koşulu:** Tespit teknik olarak başarılı olabilir. OCR için ocr_performed=false ve “OCR çalıştırılmadı” olmalı; BASARILI etiketi plaka okundu anlamına gelmez. Atlanan tool için sahte kayıt yok.

**Kanıt ve başarısızlık değerlendirmesi:** crop_count, processed_crop_count, ocr_performed, NO_CROPS ve eylem özeti.


## 5 — Hedeflenen olay–araç–plaka zinciri (henüz tam uygulanmadı)


### MAN-Z01 · P0 — Her olay için seç/atla/engellendi kararı

**Veri/önkoşul:** V05/V06/V07.

**Uygulama:** Raporu normal butonla başlatın; kullanıcı adımları tek tek söylemesin. Her olay için arşiv/takip/plaka/OCR kararlarını gerekçe ve kanıtla arayın.

**Beklenen / geçiş koşulu:** Karar kaydı her ilgili tool için run/skip/blocked ve önkoşul içerir. eylemler=[] değerlendirme yapıldı anlamına gelmez. Bu yapı yoksa BLOCKED: eksik özellik.

**Kanıt ve başarısızlık değerlendirmesi:** Mevcut şemada ayrı karar kaydı yok; hayali alanları raporda varmış gibi işaretlemeyin.


### MAN-Z02 · P0 — Araçsız olayda gereksiz plaka zinciri yok

**Veri/önkoşul:** V06 araç içermeyen sürüm.

**Uygulama:** Normal rapor oluşturun; kategori/arşiv varsa inceleyin, ardından takip/plaka/OCR çağrılarını kontrol edin.

**Beklenen / geçiş koşulu:** Araç bulunmayan olayda plaka/OCR atlanmalı; neden görünür olmalı. Görüntüdeki insanın yüzünden kimlik çıkarılmaz. Arşiv kararı OCR'a bağlı değildir.

**Kanıt ve başarısızlık değerlendirmesi:** Görsel kanıt + karar kaydı; çağrılmadı kısmı ayrı PASS olabilir, gerekçeli zincir yoksa genel sonuç BLOCKED.


### MAN-Z03 · P0 — Tek ilgili araçta tam bağlantı

**Veri/önkoşul:** V07.

**Uygulama:** Normal rapor butonuyla olay → kategori → arşiv → ilgili araç takip → plaka → OCR sonuçlarını izleyin.

**Beklenen / geçiş koşulu:** Aynı kaynak video/aralık/kare korunur. Kutulu MP4 yalnız görsel çıktı; plaka orijinal kare ve ROI üzerinden işlenir. event/job/track/crop/OCR referansları bağlıdır.

**Kanıt ve başarısızlık değerlendirmesi:** Mevcut plaka tool'u ROI/track girdisi almadığı için tam zincir bugün BLOCKED. Ayrı çağrıların doğru çıkması yeterli değil.


### MAN-Z04 · P0 — Çok araçtan yalnız ilgili olanı bağlama

**Veri/önkoşul:** V08.

**Uygulama:** Önceden ilgili araç A, yanından geçen B olarak insan etiketi oluşturun. Raporu çalıştırın; her plaka sonucunun hangi araçla ilişkilendirildiğini kontrol edin.

**Beklenen / geçiş koşulu:** B'nin plakası A'ya yazılmamalı. Yalnız olay aralığında görünme, olaya dahil olma değildir. Bir kutuya birden fazla plaka adayı varsa ilişki belirsiz kalabilmeli.

**Kanıt ve başarısızlık değerlendirmesi:** Kaynak kare+iki araç kutusu+plaka kutuları; yanlış ilişki P0 FAIL, ilişki özelliği yoksa BLOCKED.


### MAN-Z05 · P0 — Örtüşme ve track_id değişimi

**Veri/önkoşul:** V09.

**Uygulama:** Araçlar kesişsin/örtülsün; bir araç çıkıp tekrar girsin. Aynı aracın track_id değiştiği kareleri ve OCR referanslarını inceleyin.

**Beklenen / geçiş koşulu:** ID değişimi gerçek araç değişimi diye kesinleştirilmez. ID tek başına kalıcı kimlik değildir; çapraz işe taşınmaz. Belirsiz eşleşme zorla doldurulmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Frame-by-frame bağlantı tablosu; görünmeyen aralık için sahte plaka/track yok.


### MAN-Z06 · P0 — Plaka yok, okunamıyor veya model eksik

**Veri/önkoşul:** V08 bulanık/gece sürümü.

**Uygulama:** Aynı olay için plaka bulunamama, OCR uncertain ve OCR model eksikliği durumlarını ayrı ayrı deneyin.

**Beklenen / geçiş koşulu:** Kategori/arşiv/takip sonuçları korunur. No plate, unreadable, blocked OCR ayrılır; candidate_text kesin numara olmaz. Yapılmayan OCR “başarılı okuma” sayılmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Karar/sonuç durumu, hata kodları, mevcut sağlam dosyalar ve JSON.


### MAN-Z07 · P0 — Aynı olayın tekrar raporlanması

**Veri/önkoşul:** V07.

**Uygulama:** Aynı video için raporu iki kez üretin; arşiv, takip, plaka ve OCR dosyalarını ayrı karşılaştırın.

**Beklenen / geçiş koşulu:** Arşiv ve takip cache sözleşmelerini uygular. Plaka/OCR şu anda yeniden üretir; tüm zincir idempotent varsayılmaz. Önceki rapor eylemleri yeni raporda sahte yeni işlem gibi gösterilmez.

**Kanıt ve başarısızlık değerlendirmesi:** İki job, dosya hash/yol listesi, cache_hit ve detection_cache_hit. “LLM önceki cevabı hatırladı” cache testi değildir.


### MAN-Z08 · P0 — Çok olay, çok araç ve çağrı bütçesi

**Veri/önkoşul:** V05 + V08.

**Uygulama:** En az üç olaylı video için arşiv/takip/plaka/OCR gerektiren rapor çalıştırın; tool-node turunu ve tekil çağrı sayısını ayrı sayın.

**Beklenen / geçiş koşulu:** 8 tur sınırında kalan işler yapılmış gibi gösterilmez. Bağımlı OCR, tespit sonucu gelmeden paralel çağrılmaz. Arşiv başarısızlığı bağımsız görsel kanıtı yok etmez.

**Kanıt ve başarısızlık değerlendirmesi:** Olay bazlı tamamlanan/engellenen adımlar, tool sayacı, dosyalar; bütün zincir için açık terminal durum.


## 6 — Anomali modeli ve skorların zamanı


### MAN-S01 · P1 — Analyzer ve rapor tool'u aynı ayarları kullanıyor

**Veri/önkoşul:** V05.

**Uygulama:** Aynı checkpoint/hash, AS_* ve cihazla API Analyzer, Gradio Analyzer ve Agent segmenter'ı çalıştırın. Sıcak/soğuk yüklemeyi ayırın.

**Beklenen / geçiş koşulu:** Ayarlar aynıysa açıklanamayan fark olmamalı; eşik 0.3 ve kaynak süre görünür. Farklı crop/FPS/model yolu varsa fark belgelenir; aynı sanılmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Üç girişin cihaz/checkpoint/clip_size/overlap/FPS/segment sonuçları. Model önbelleği nedeniyle ayar değişiminden sonra sunucuyu yeniden başlatın.


### MAN-S02 · P1 — Checkpoint yok veya uyumsuz

**Veri/önkoşul:** Ayrı test süreci + V04.

**Uygulama:** Geçersiz checkpoint yolu ve hiç checkpoint bulunmayan ortamı ayrı deneyin. API Analyzer ile Agent segmenter'ı karşılaştırın; mevcut gerçek dosyayı silmeyin.

**Beklenen / geçiş koşulu:** Rastgele FC ağırlığıyla başarılı anomali raporu üretilmemeli. Her giriş açık hata vermeli; diğer mümkün işler çalışmaya devam etmeli.

**Kanıt ve başarısızlık değerlendirmesi:** KG06; yalnız yanlış açık yol değil, fallback dosyası da olmayan ortam/mock gerekir. Eğitim başlamamalı.


### MAN-S03 · P1 — Klip boyu sınırları

**Veri/önkoşul:** V11.

**Uygulama:** Örneklenmiş kare sayısı clip_size−1, clip_size, clip_size+stride ve +kalan olacak videolar hazırlayın; klip sayısını ölçün.

**Beklenen / geçiş koşulu:** Çok kısa video açıklayıcı hata. Tam pencereler için 1+floor((N−clip_size)/stride); son kısmi pencere kapsamı gizlenmez. full_video iddiası işlenmeyen son kareler varsa sorgulanır.

**Kanıt ve başarısızlık değerlendirmesi:** Kaynak/örneklenmiş N, pencere indeksleri, analiz kapsamı. Atılan kuyruk bilinen davranış olabilir, güvenlik kapsamı PASS değildir.


### MAN-S04 · P1 — Son karede kısa olay

**Veri/önkoşul:** V11 kuyruk + V05.

**Uygulama:** Olayı yalnız son eksik pencereye denk getirin; aynı olayı videonun ortasına taşıyan referansla karşılaştırın.

**Beklenen / geçiş koşulu:** Son kısım işlenmeden tüm video incelendi denmemeli; skorların toplam süreye yayılması olayı yanlış zamanda göstermemeli.

**Kanıt ve başarısızlık değerlendirmesi:** Son kaynak kareler, score grafiği, segmentler. Model kaçırması ile pencere kapsamı hatasını ayırın.


### MAN-S05 · P1 — Padding, birleştirme ve süre sınırı

**Veri/önkoşul:** V05.

**Uygulama:** Yakın iki olay, uzak iki olay ve 0/video-sonu olaylarını analiz edin. Ham/smoothed grafiği ve aralıkları karşılaştırın.

**Beklenen / geçiş koşulu:** 0≤start<end≤duration; duration=end−start yuvarlama toleransında. Tolerans içindeki boşluklar/padding birleşebilir; tek skor birden fazla olayın kesin aynı olay olduğu anlamına gelmez.

**Kanıt ve başarısızlık değerlendirmesi:** Grafik ve JSON aralıkları, kullanılan tolerance/padding; yanlış saniye veya video dışı aralık FAIL.


### MAN-S06 · P1 — Cihaz ve kapasite hatası

**Veri/önkoşul:** V05 + uygun cihaz.

**Uygulama:** CPU; varsa CUDA/MPS ile ayrı çalıştırın. Kullanılamayan açık cihaz, yanlış AS_DEVICE ve overlap≥clip_size gibi ayarları ayrı test sürecinde deneyin.

**Beklenen / geçiş koşulu:** Geçersiz ayar sessizce başka cihaz/bozuk stride'a dönüşmemeli. MPS fallback performans uyarısı analiz başarısızlığıyla karıştırılmaz. TensorRT yalnız uygun CUDA'da kullanılmalı.

**Kanıt ve başarısızlık değerlendirmesi:** Cihaz, log, süre, hata ve kaynak hash'i. Donanım yoksa ilgili alt durum N/A, başarısız sonuç PASS değil.


## 7 — Medya geometrisi, örnekleme ve zaman


### MAN-M01 · P1 — En-boy, renk ve dönüş bilgisi

**Veri/önkoşul:** V03.

**Uygulama:** Daire, kare, renk şeritleri bulunan dikey/yatay/kare ve dönüş metadata'lı videoyu kaynak oynatıcı, generate_frames ve VLM hazırlığından sonra karşılaştırın.

**Beklenen / geçiş koşulu:** Daire elipse dönüşmemeli; görüntü yönü doğru; RGB/BGR kanalları karışmamalı. Model girdisinde yeniden boyutlandırma oranı/hizalama sınırı açıklanmalı.

**Kanıt ve başarısızlık değerlendirmesi:** Kaynak ve gönderilen kare ölçüleri; JPEG/MP4 marker tek başına içerik geometri kanıtı değil. KG03 alt kontrolü.


### MAN-M02 · P1 — Kaynak aralığı ve örnekleme yoğunluğu

**Veri/önkoşul:** V10.

**Uygulama:** 2–6 sn gibi sıfır dışı aralıkta 5 FPS ve düşük max_frames ile kareleri çıkarın; gerçek ilk/son source_sec'yi kontrol edin.

**Beklenen / geçiş koşulu:** Kare sayısı bütçeyi aşmaz, zaman sırası artar; requested/effective/sampled ayrımı korunur. max_frames dolunca tüm kareler görüldü denmez.

**Kanıt ve başarısızlık değerlendirmesi:** Örneklenen indeks/zaman listesi ve VLM marker; son frame start ile video duration arasındaki küçük fark hata değildir.


### MAN-M03 · P1 — Çok kısa/uzun VLM kesitleri

**Veri/önkoşul:** V11 + V12.

**Uygulama:** Tek kareye yakın kısa kesit ve 128 kareyi aşan uzun aralığı inceletin. n/encode_fps ile geçici klibin süresini hesaplayın.

**Beklenen / geçiş koşulu:** Minimum kare çoğaltması yeni gözlem sayılmaz. FPS clamp nedeniyle süre korunamıyorsa açık uyarı/parçalama gerekir; hız değişimi gerçek hareket diye yorumlanmamalı.

**Kanıt ve başarısızlık değerlendirmesi:** n/encode_fps, istenen süre, model yanıtındaki zamanlar. VLM'nin bir saniyelik olayı örnekleme yüzünden kaçırması ayrıca kaydedilir.


### MAN-M04 · P1 — Geçersiz ve taşan zamanlar

**Veri/önkoşul:** 10 sn V10.

**Uygulama:** Sohbetten negatif, ters, sıfır uzunluk, başlangıcı video sonuna eşit ve yalnız bitişi taşan aralıkları isteyin; NaN/Inf için otomatik test kullanın.

**Beklenen / geçiş koşulu:** Geçersiz başlangıç TIME_OUT_OF_RANGE veya INVALID_TIME_RANGE; yalnız bitiş kırpılır ve END_TIME_CLAMPED görünür. Olmayan zamanda anomali yok denmez.

**Kanıt ve başarısızlık değerlendirmesi:** Tool args, error/warnings, çıktı dosyası yokluğu; yapılandırılmış argüman validasyonu tool gövdesinden önce reddedebilir.


### MAN-M05 · P1 — VFR ve sesli çıktı

**Veri/önkoşul:** V10.

**Uygulama:** 0'dan başlamayan kesitte kutulu video üretin; kaynak ve çıktı kare PTS'lerini, belirgin ses/görüntü işaretini karşılaştırın.

**Beklenen / geçiş koşulu:** Takip çıktısında output_timing=source_timestamps; PTS hatası en çok 1 ms, son süre kontrolü kod sözleşmesinde 2 ms. Ses senkronu gözle de doğrulanır. Ortalama FPS tek başına yeterli değil.

**Kanıt ve başarısızlık değerlendirmesi:** ffprobe PTS, başlangıç offset'i, kaynak/çıktı oynatma; diğer decoder/codec toleransları açıkça kaydedilsin.


### MAN-M06 · P1 — Bozuk/boş dosya ve kaynağın değişmesi

**Veri/önkoşul:** V13.

**Uygulama:** Boş, .mp4 uzantılı metin ve yarım video kopyası yükleyin. Ayrı kontrollü süreçte analiz sürerken yalnız disposable kaynak kopyasını değiştirin.

**Beklenen / geçiş koşulu:** Çalışan decoder yoksa başarılı analiz/rapor yok. Kaynak değişimi doğrulanan tool'larda hata; eski metadata/cache yanlış kullanılmaz. Diğer sağlam dosyalar korunur.

**Kanıt ve başarısızlık değerlendirmesi:** Upload kabulü gerçek video doğrulaması demek değil; downstream hata ve kalıcı çıktı listesi kontrol edilir.


## 8 — Nesne tespiti ve takip


### MAN-O01 · P1 — Sınıf filtresi ve yetenek sınırı

**Veri/önkoşul:** V08/V09.

**Uygulama:** Sohbet: “2–6 saniyede car ve person sınıflarını takip et, kutulu video üret.” Sonra filtresiz, boş ve gun/plate filtrelerini kontrollü tool çağrısıyla deneyin.

**Beklenen / geçiş koşulu:** İstenen destekli sınıflar; boş/desteksiz filtre UNSUPPORTED_OBJECT_CLASS. YOLO plaka metni, silah sınıfı, suç veya gerçek kişi kimliği uydurmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Tool sınıfları, desteklenen liste, frames.json ve render sonucu.


### MAN-O02 · P1 — Kutu/ID kaynak kareye oturuyor

**Veri/önkoşul:** V08/V09.

**Uygulama:** En az ilk/orta/son ve örtüşme karelerini kaynakla karşılaştırın; kutuların nesneyi takip ettiğini kontrol edin.

**Beklenen / geçiş koşulu:** xyxy kaynak piksel uzayında; ID=null ise atanmış kimlik gibi anlatılmaz. detection_count karelerdeki kutu toplamıdır; tekil araç sayısı olarak sunulmaz.

**Kanıt ve başarısızlık değerlendirmesi:** İş/video/track_id ile kare ekran görüntüleri; tracking doğruluğu schema testiyle kanıtlanmaz.


### MAN-O03 · P1 — Görünme aralıkları ve kesilmiş özet

**Veri/önkoşul:** V09 + çok parçalı V12.

**Uygulama:** Nesne çıkıp tekrar girsin; 100'den fazla aralıklı bir örnek için summary ve intervals_path'i karşılaştırın.

**Beklenen / geçiş koşulu:** Görünmeyen boşluklar birleştirilmez. intervals_truncated varsa tüm liste dosyada; sınıf aralıkları özette öncelikli. End hariç sınır korunur.

**Kanıt ve başarısızlık değerlendirmesi:** interval_count, örnek aralıklar ve tam JSON; 100 özet satırı 100 toplam nesne demek değil.


### MAN-O04 · P1 — Tespit ve çizim cache'leri ayrı

**Veri/önkoşul:** V08.

**Uygulama:** Önce render_video=false, sonra aynı ayarlarla true, sonra true tekrar isteyin. Disposable cache'de yalnız MP4'ü, ayrı denemede frames.json'u değiştirin.

**Beklenen / geçiş koşulu:** İlk çizimde detection_cache_hit=true/cache_hit=false; tekrar çizimde ikisi true. MP4 bozuksa tespit korunur; frame verisi bozuksa yeniden tespit. Kaynak değişmez.

**Kanıt ve başarısızlık değerlendirmesi:** Gerçek tool çağrı sayısı, model çağrı sayısı, dosya hash'leri. Cache klasörünü üretimde bozmayın.


### MAN-O05 · P1 — Limit, timeout ve eksik render bağımlılığı

**Veri/önkoşul:** Ayrı test ortamı + V12.

**Uygulama:** OBJECT_MAX_FRAMES=5, çok kısa timeout ve FFmpeg/PyAV eksikliği durumlarını ayrı deneyin.

**Beklenen / geçiş koşulu:** FRAME_LIMIT_EXCEEDED/TRACKING_TIMEOUT veya ilgili bağımlılık hatası; kısmi analiz başarı sayılmaz. Render hatası sağlam tespit cache'ini yok etmez; detection-only uygun ortamda çalışır.

**Kanıt ve başarısızlık değerlendirmesi:** Hata kodları, geçici/sağlam dosyalar ve sonraki başarılı deneme.


### MAN-O06 · P1 — Model cache yükleme hatası ve eşzamanlılık

**Veri/önkoşul:** İki video + disposable ağırlık ayarı.

**Uygulama:** Sağlam A modelini yükleyin, bozuk B dosyasını deneyin, sonra A'ya dönün. İki oturumdan aynı/different video çağrısı gönderin.

**Beklenen / geçiş koşulu:** Başarısız B önceki sağlam cache'i bozmaz. Her iş ayrı tracker; dosyalar ve zamanlar karışmaz. Kilit bekleme/timeout açık; CPU aşırı kullanımından sessiz sonuç bozulması yok.

**Kanıt ve başarısızlık değerlendirmesi:** Model signature/key, iki job sonucu ve çağrı zamanları; model dosyalarını silmeyin.


## 9 — Plaka bölgesi tespiti


### MAN-P01 · P1 — Plaka kutusu ve PNG gerçek içeriği

**Veri/önkoşul:** V07.

**Uygulama:** “2–4 saniyede plaka bölgelerini bul ve kırp; henüz okuma” deyin. İlk/orta/son PNG'leri kaynak kareyle yan yana açın.

**Beklenen / geçiş koşulu:** Plaka bölgesi görünür; kırpım orijinal çözünürlükte, renkler doğru. width=x2−x1, height=y2−y1. OCR yapıldı denmez.

**Kanıt ve başarısızlık değerlendirmesi:** crop_path, bbox_xyxy, source_sec, frame_index, PNG hash ve gözle kontrol.


### MAN-P02 · P1 — Letterbox koordinat dönüşümü

**Veri/önkoşul:** V03 + V07 dikey/yatay.

**Uygulama:** Aynı plakanın görüntü merkezi ve kenarında olduğu örnekleri deneyin; kaynak kutularını ölçün.

**Beklenen / geçiş koşulu:** ONNX 384×384 letterbox padding'i geri alınmalı; kutu kaynak sınırında kırpılır, boş/ters kutu kaydedilmez. Dikey VLM hatasıyla plaka ön işlemesini karıştırmayın.

**Kanıt ve başarısızlık değerlendirmesi:** Kaynak boyut, padding/scale, çıktı kutusu; otomatik sentetik koordinat testiyle desteklenir.


### MAN-P03 · P1 — Birden fazla araç ve tekrar eden plakalar

**Veri/önkoşul:** V08.

**Uygulama:** İki aracın plakasını 2–5 sn aralığında tespit ettirin; aynı plakanın ardışık kare kırpımlarını sayın.

**Beklenen / geçiş koşulu:** Her adayın ayrı PNG'si olur; crop_count tekil plaka/araç sayısı değil. Track bağlantısı yapılmış gibi gösterilmez; bunun kabulü Z04'tür.

**Kanıt ve başarısızlık değerlendirmesi:** Tam crops.json ve kaynak kare; yanlış araç eşleştirme iddiası FAIL.


### MAN-P04 · P1 — Özet 30, tam manifest ve sıfır aday

**Veri/önkoşul:** V07/V04.

**Uygulama:** 30'dan fazla kırpımlı örnek ve plakasız örneği ayrı çalıştırın.

**Beklenen / geçiş koşulu:** crops_truncated doğru, özette ilk 30 ve details_path'te tamamı. Sıfır sonuçta NO_PLATE_DETECTED; kesin plaka yok iddiası değil.

**Kanıt ve başarısızlık değerlendirmesi:** Özet/tam dosya sayımı; OCR'a özet liste değil manifest aktarılır.


### MAN-P05 · P1 — Her kare ve bitiş hariç aralık

**Veri/önkoşul:** V10.

**Uygulama:** 2.05–4.55 sn aralığını verin; her kayıt frame_index/source_sec değerini source PTS ile karşılaştırın.

**Beklenen / geçiş koşulu:** İstenen aralıkta başlayan her kare işlenir; end_sec hariç. Kırpım olmayan kare processed_frame_count'a yine dahil; atlanan kareler gizlenmez.

**Kanıt ve başarısızlık değerlendirmesi:** Frame başlangıçları, processed_frame_count ve crop_count farkı.


### MAN-P06 · P1 — Plaka sınırları, model ve yazma hataları

**Veri/önkoşul:** V12 + izole test süreci.

**Uygulama:** PLATE_MAX_FRAMES/MAX_CROPS/TIMEOUT sınırlarını sırayla düşürün; eksik/uyumsuz ONNX veya yazılamayan disposable çıktı dizini deneyin.

**Beklenen / geçiş koşulu:** PLATE_FRAME_LIMIT/PLATE_CROP_LIMIT/PLATE_TIMEOUT veya açık model/yazma hatası; kısmi klasör temizlenir, önceki sağlam klasör korunur. Tekrarda yeni klasör normaldir; cache garantisi yok.

**Kanıt ve başarısızlık değerlendirmesi:** Önce/sonra dosya envanteri ve hata kodu. Uygun ortam yoksa ilgili alt durum BLOCKED.


## 10 — OCR ve plaka kanıtı


### MAN-C01 · P1 — Okunabilir plaka ve kesinlik sınırı

**Veri/önkoşul:** V07 crops.json.

**Uygulama:** Önce insan PNG'yi okuyup etiketi kaydetsin; sonra “Bu kırpım manifestini oku” deyin. Video sahibini belirlemeyi istemeyin.

**Beklenen / geçiş koşulu:** status=read ise text kırpımla karşılaştırılır; yüksek skor kesin doğruluk olasılığı değildir. Yanlış yüksek güvenli okuma kalite hatası olarak kaydedilir.

**Kanıt ve başarısızlık değerlendirmesi:** text/candidate_text/min_slot_confidence/slot_confidences + insan etiketi; plaka özel verisini paylaşırken maskeleyin.


### MAN-C02 · P1 — Bulanıklık, iç boşluk, yalnız padding

**Veri/önkoşul:** V08 bulanık + sentetik olasılık testleri.

**Uygulama:** Belirsiz/okunamayan kırpımları deneyin; otomatik testte düşük olasılık, iç _ ve tamamı padding çıktısı kullanın.

**Beklenen / geçiş koşulu:** uncertain/unreadable için text=null; candidate_text kesin plaka diye rapora taşınmaz. Yanlış şekil/NaN/normalize olmayan model çıktısı reddedilir.

**Kanıt ve başarısızlık değerlendirmesi:** Kırpım görüntüsü ve ham OCR alanları; belirli bulanıklık her zaman düşük güven üretir varsayılmaz.


### MAN-C03 · P1 — Tam manifest, 30 sınırı ve provenance tekrar tespiti yok

**Veri/önkoşul:** V07 >30 kırpım.

**Uygulama:** Tespitin details_path'ini OCR'a verin; kaynak videonun disposable kopyasını kaldırdıktan sonra yalnız mevcut PNG/manifestle tekrar OCR deneyin.

**Beklenen / geçiş koşulu:** processed_crop_count tam listeye uyar; results_truncated sadece özeti etkiler. OCR video okuyucu/plaka dedektörü çalıştırmaz; kaynak video olmadan sağlıklı kırpım okunabilir.

**Kanıt ve başarısızlık değerlendirmesi:** Tool çağrıları, readings.json sayısı ve kaynak/crop dosyalarının hash'leri.


### MAN-C04 · P1 — Manifest yolu, zaman ve boyut bütünlüğü

**Veri/önkoşul:** Disposable crops.json.

**Uygulama:** Kırpım yolunu başka klasöre yönlendirin; source_sec'yi bitişe eşit yapın; bbox/boyut çelişkisi ve duplicate crop_path üretin.

**Beklenen / geçiş koşulu:** INVALID_CROP_MANIFEST; rastgele yerel PNG okumaz. Hiçbir orijinal veri düzenlenmez; yalnız test kopyası kullanılır.

**Kanıt ve başarısızlık değerlendirmesi:** Negatif varyantın farkı ve hata kodu; otomatik testte geçici dizin kullanılır.


### MAN-C05 · P1 — PNG hash, eksik dosya ve eski kayıt

**Veri/önkoşul:** Disposable plaka klasörü.

**Uygulama:** Bir PNG'yi silin; ayrı kopyada aynı boyutla içeriği değiştirin; başka denemede sha256 alanını kaldırın.

**Beklenen / geçiş koşulu:** Eksik OCR_CROP_MISSING; değişmiş hash OCR_CROP_CHANGED veya boyut bozuksa OCR_INVALID_CROP. Eski hash'siz kayıtta LEGACY_CROPS_UNVERIFIED; sessiz güven iddiası yok.

**Kanıt ve başarısızlık değerlendirmesi:** Kaynak/son hash, PNG boyutu, warnings; eski sağlam kayda dokunmayın.


### MAN-C06 · P1 — Model sözleşmesi, limit ve boş kırpım

**Veri/önkoşul:** Boş manifest + izole test ayarı.

**Uygulama:** OCR model/config eksikliği, yanlış alphabet/shape, crop limit ve timeout deneyin; boş manifesti model olmadan çalıştırın.

**Beklenen / geçiş koşulu:** Geçersiz model/config açık hata; sıfır kırpımda OCR modeli yüklenmeden ocr_performed=false/NO_CROPS. Başarısız işte readings.json başarı diye yayımlanmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Model/config hash'leri, processed/read/uncertain sayıları, hata ve dosyalar.


## 11 — Kategorili arşiv ve genel klip kaydı


### MAN-A01 · P1 — Kategori, gerekçe ve gerçek dosya

**Veri/önkoşul:** V06.

**Uygulama:** Sohbet: “2–5 saniyedeki gözlemi incele, tür belirsizse belirsiz kategorisine gerekçesiyle arşivle.” Sonra rapordaki gerçekleşen eylemle ilişkiyi kontrol edin.

**Beklenen / geçiş koşulu:** Kategori enum içinde; explanation görüntüye dayanır. clip.mp4 ve metadata.json gerçekten açılır, süre uyumlu, kaynak aynı hash'te. Tool kendi kendine sınıflandırmaz.

**Kanıt ve başarısızlık değerlendirmesi:** video_path/category/saved_range/explanation/output_path, kaynak ve çıktı hash'leri.


### MAN-A02 · P1 — Tekrar istek ve ilk gerekçe

**Veri/önkoşul:** V06.

**Uygulama:** Aynı kaynak/aralık/kategoriyle arşiv çağrısını gerçekten iki kez çalıştırın; ikincide gerekçeyi farklı yazın.

**Beklenen / geçiş koşulu:** cache_hit=true, aynı dosya ve ilk explanation korunur. LLM aracı çağırmadan “zaten kaydedildi” derse cache alt testi yürütülmüş sayılmaz.

**Kanıt ve başarısızlık değerlendirmesi:** İki call_id, tek kayıt klasörü ve metadata karşılaştırması.


### MAN-A03 · P1 — Kategori değişimi ve bozuk mevcut arşiv

**Veri/önkoşul:** Disposable arşiv.

**Uygulama:** Aynı kesiti başka kategoriye isteyin; ayrı testte clip/metadata'yı bozup tekrar aynı kategoriye çağırın.

**Beklenen / geçiş koşulu:** ARCHIVE_CATEGORY_CONFLICT veya ARCHIVE_CONFLICT; sessiz üstüne yazma ve ikinci kategoriye kopya yok. Çakışma raporda başarıya dönüşmez.

**Kanıt ve başarısızlık değerlendirmesi:** İlk kaydın korunması ve hata kodu; gerçek olay arşivini bozmayın.


### MAN-A04 · P1 — Arşiv hatasında yalnız yeni geçici çıktı temizliği

**Veri/önkoşul:** İzole export hatası + sağlam eski kayıt.

**Uygulama:** FFmpeg hatası/yazma hatası/süre sınırı/kaynak değişimi sırasında yeni bir aralık kaydetmeyi deneyin.

**Beklenen / geçiş koşulu:** Yalnız başarısız yeni klasör temizlenir; sağlam eski kayıt ve orijinal video kalır. Hata kodu BASARISIZ eylemde görünür.

**Kanıt ve başarısızlık değerlendirmesi:** Önce/sonra dizin listesi, source hash, çağrı-sonuç çifti; mock export dosyası oynatılabilirlik kanıtı değildir.


### MAN-A05 · P1 — Genel save_video_segment arşivle aynı değildir

**Veri/önkoşul:** V10.

**Uygulama:** “2–5 saniyeyi test çıktıları içindeki yeni bir MP4 adına kaydet” deyin; aynı aralığı arşivle karşılaştırın. Çıktı klasörünü önceden oluşturun.

**Beklenen / geçiş koşulu:** Genel kayıt stream-copy kullanır; keyframe öncesi görüntü olabilir. Arşiv yeniden kodlar ve kare hassasiyetli süre hedefler. İkisini aynı hassasiyetle geçti diye işaretlemeyin.

**Kanıt ve başarısızlık değerlendirmesi:** ffprobe, ilk/son kare, kaynak ses. Genel kayıt -y ile mevcut hedefin üstüne yazabildiğinden yalnız yeni disposable yol kullanın.


### MAN-A06 · P1 — Kaynak üstüne yazma ve çıktı yolu

**Veri/önkoşul:** Disposable V10.

**Uygulama:** Çıktı adı boş, olmayan klasör, uzantısız isim ve kaynakla aynı yol durumlarını kontrollü deneyin.

**Beklenen / geçiş koşulu:** Kaynak kendi üstüne yazılmaz; uzantısız isme .mp4 eklenir; geçersiz yol açık hata. Başarısız çağrı “klip kaydedildi” diye raporlanmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Hata kodları, kaynak hash ve dosya varlığı. Yükleme session_id güvenliği bu yerel testin kapsamı dışındadır.


## 12 — Kurulum, servis ve kaynak yönetimi


### MAN-X01 · P1 — Doğru sürüm ve doğru sunucu

**Veri/önkoşul:** Temiz test oturumu.

**Uygulama:** Commit, branch, Python, paket ve frontend build tarihini kaydedin; 8000 React, /gradio ve ayrı 7860'ı ayırt edin. Aynı portta ikinci sunucu başlatmayın.

**Beklenen / geçiş koşulu:** Eski process/build ile yeni kodu test etmeyin. /api/health tek başına Gradio'nun gerçekten mount edildiğini kanıtlamaz; ekranı da açın.

**Kanıt ve başarısızlık değerlendirmesi:** PID/port, commit, build ve URL; gerçek .env değerleri/anahtarları rapora eklenmez.


### MAN-X02 · P1 — Ortam değişkeni önceliği ve web portu

**Veri/önkoşul:** Ayrı test süreci.

**Uygulama:** AS/VLM/OBJECT/PLATE ayarlarını shell export ve .env üzerinden ayrı deneyin; WEB_PORT'u da karşılaştırın. Her değişimde gerekli sunucuyu yeniden başlatın.

**Beklenen / geçiş koşulu:** Etkin ayar beklenene uyar; .env varsayımıyla farklı model/port kullanılmaz. Eksik anahtarın hatası anlaşılır; anahtar stdout'a basılmaz.

**Kanıt ve başarısızlık değerlendirmesi:** Sadece anahtar adı ve gizli olmayan etkin değerler. run_web .env yükleme sırası riski açık kalabilir.


### MAN-X03 · P1 — Opsiyonel paket/model eksikliği

**Veri/önkoşul:** Ayrı venv/container.

**Uygulama:** Temel paketlerle metadata/sohbet/rapor açılışını; YOLO, ONNX, FFmpeg/PyAV eksikliğinde ilgili aracı deneyin. Üretim venv'inden paket kaldırmayın.

**Beklenen / geçiş koşulu:** İlgili özellik açık hata verir; runtime gizli pip/model indirmesi yok. Diğer bağımsız özellikler kullanılabilir. İlk kurulumda model backbone indirmesi gerekiyorsa önceden açıkça hazırlanır.

**Kanıt ve başarısızlık değerlendirmesi:** Bağımlılık listesi, ağ kaydı, tool hatası. İzole ortam yoksa BLOCKED.


### MAN-X04 · P1 — Uzun kullanım, çıktı birikimi ve yeniden başlatma

**Veri/önkoşul:** V04/V05 tekrarları.

**Uygulama:** 20 kısa işten sonra süreç RAM, session/job sayısı ve disk kullanımını başlangıçla karşılaştırın. Sonuçları saklayıp uygulamayı yeniden başlatın.

**Beklenen / geçiş koşulu:** Dosya/oturumların otomatik temizlendiği varsayılmaz; sınırsız büyüme ve eski job erişimi belgelenir. Kullanıcıya sunulan dosyalar yanlışlıkla silinmez; temizleme politikası yoksa işletim açığıdır.

**Kanıt ve başarısızlık değerlendirmesi:** İş başına süre, RAM/disk trendi, restart sonrası erişim. Performans sınırı ölçümden önce ekipçe belirlenmeli.


## 13 — Eğitim ve yardımcı girişler (rapor testinden ayrı)


### MAN-T01 · P2 — Eğitim/doğrulama veri ayrımı

**Veri/önkoşul:** Özellik dosyalarının anonim manifesti.

**Uygulama:** Eğitim başlatmadan normal/anormal train/validation listelerini denetleyin; aynı kaynak videonun kliplerini ve k-fold kalan örneklerini karşılaştırın.

**Beklenen / geçiş koşulu:** Aynı kaynak/hash iki kümeye düşmemeli; hiçbir örnek sessizce kaybolmamalı. Mevcut normal_train_files=normal_files ve normal_test aynı listeden seçim riski kapatılmadan bağımsız doğruluk iddiası verilmez.

**Kanıt ve başarısızlık değerlendirmesi:** Küme kesişimi/coverage raporu. Bu paket 75 epoch veya CUDA eğitimi başlatmaz.


### MAN-T02 · P2 — FC eval, checkpoint ve cihaz uyumu

**Veri/önkoşul:** Sentetik özellik + mevcut checkpoint kopyası.

**Uygulama:** Aynı özelliği eval modunda tekrar skorlayın; train modundaki dropout/gürültüyü ayırın. Checkpoint input_dim/state_dict uyumunu kontrol edin.

**Beklenen / geçiş koşulu:** Eval aynı ortamda deterministik olmalı; shape uyuşmazlığı açık hata. Sigmoid skoru kalibre olasılık değildir. Eğitim varsayılan CUDA bekler; CPU/Mac'te otomatik eğitim garantisi yok.

**Kanıt ve başarısızlık değerlendirmesi:** Checkpoint hash, feature_dim, seed, model modu ve skorlar. Gerçek eğitim ayrı izin/kapasite planı gerektirir.


### MAN-T03 · P2 — CLI ile web davranışı ayrışması

**Veri/önkoşul:** test_main.py + kısa sohbet.

**Uygulama:** CLI girişini ayrı açın; metadata, takip sorusu ve başarısız görev deneyin; aynılarını React Sohbet'te karşılaştırın.

**Beklenen / geçiş koşulu:** test_main.py bir birim test dosyası değildir. CLI'nin messages/conversation state farkı nedeniyle yanlış hafıza veya yalnız feedback gösterimi varsa kaydedilir; web PASS'i CLI'yi kapsamaz.

**Kanıt ve başarısızlık değerlendirmesi:** İki girişin nihai cevap/geçmiş/trace örnekleri; otomatik discovery proje kökünden başlatılmaz.


## 14 — Model kalitesi ve nihai kabul


### MAN-Q01 · P1 — Olay doğruluğu ve yön/mesafe hataları

**Veri/önkoşul:** V01/V04/V05/V06/V08.

**Uygulama:** Model çıktısını görmeden iki inceleyici olay aralıklarını ve belirsizliklerini etiketlesin; sonra raporla karşılaştırın. Çelişkiyi üçüncü inceleme veya INCONCLUSIVE ile çözün.

**Beklenen / geçiş koşulu:** Yanlış “normal”, yanlış suç, kaçan olay ve yanlış zaman ayrı sayılır. Görsel iddialar kaynak kareyle desteklenir; araç varlığı suç kanıtı değildir.

**Kanıt ve başarısızlık değerlendirmesi:** İnsan etiketleri, ham rapor, olay bazında TP/FP/FN. İki saniyelik örnek kare incelemesi tam video etiketi yerine geçmez.


### MAN-Q02 · P1 — Tekrarlar ve dağılım değişimi

**Veri/önkoşul:** Gündüz/gece, dikey/yatay, CCTV/dashcam.

**Uygulama:** Aynı videoyu üç yeni oturumda aynı ayarla; ayrıca farklı kamera/gece/açı koşullarında çalıştırın. Kalibrasyon için kullanılan videoyu nihai testten ayırın.

**Beklenen / geçiş koşulu:** Bir başarılı örnek genelleme kanıtı değildir. Tekrarlarda karar, kategori, OCR ve eylem seçimi farkları ölçülür; source/model/config hash sabit tutulur.

**Kanıt ve başarısızlık değerlendirmesi:** Kategori doğruluğu, zaman örtüşmesi, yanlış-normal oranı, OCR exact match, araç-plaka yanlış eşleşmesi. Minimum örnek sayısı ve eşikler ekipçe önceden belirlenir.


### MAN-Q03 · P1 — Son kabul: teknik başarı ve anlam doğruluğu

**Veri/önkoşul:** Tüm P0 sonuçları + örnek P1.

**Uygulama:** Sonuç tablosunu kapatın; FAIL/BLOCKED/INCONCLUSIVE satırlarını ayrı listeleyin. Artifact dosyalarını açıp trace→JSON→video eşleşmesini son kez kontrol edin.

**Beklenen / geçiş koşulu:** Tüm P0'lar kanıtlı PASS olmadan zincir güvenilir/eksiksiz denmez. Skor “olasılık”, OCR güveni “kesin doğru” diye sunulmaz. Henüz olmayan ROI/track ilişkisinin testleri BLOCKED kalır.

**Kanıt ve başarısızlık değerlendirmesi:** İmzalı kabul özeti, açıklar, commit/model/config/video sürümleri. Otomatik mock testlerinin geçmesi gerçek model doğruluğu değildir.


## NOT: eşik değerleri ve başarı toleransları

Kaynak-zaman sınırları kod sözleşmesine göre test edilir; model doğruluğuna rastgele yüzde hedef atanmaz. Kare/PTS toleransları M05'te, arşiv süre toleransı `max(0.15 sn, 2/fps)` kodundadır. Görsel kalite için önce temsil edici etiketli veri ve iş gereksinimi belirlenmelidir. Yanlış araca plaka atamak veya analiz yapmadan düşük risk vermek, küçük bir istatistik farkı değil P0 hatadır.
