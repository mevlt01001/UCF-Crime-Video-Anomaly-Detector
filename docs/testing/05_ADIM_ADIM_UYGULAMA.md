# Manuel test uygulama rehberi

Bu belge **nasıl uygulanacağını**, [senaryo planı](02_MANUEL_TEST_PLANI.md) ise her senaryonun **kabul koşulunu** tanımlar. Henüz hiçbir senaryo bu paket kapsamında çalıştırılmadı. Her denemede tek değişkeni değiştirin. Hata alınca ayarları rastgele değiştirerek tekrar başlatmayın; önce kanıtı saklayın.

## 1. Terminal ve uygulamayı hazırlama

Komut bloklarına terminalin kullanıcı adı/`%` kısmını veya önceki çıktıyı eklemeyin. İki terminal kullanın: birincisi sunucu, ikincisi kayıt/test komutları için.

Her iki terminalde proje köküne geçin:

```bash
cd /Users/ahmetcan/Projects/UCF-Crime-Video-Anomaly-Detector
```

Sunucu zaten doğru sürümle çalışıyorsa yeniden başlatmayın. Çalışmıyorsa birinci terminalde önce derleyin:

```bash
npm --prefix frontend run build
```

Yalnız derleme başarılıysa:

```bash
.venv/bin/python run_web.py
```

Terminali açık bırakın. Varsayılan adres `http://127.0.0.1:8000/`; terminal farklı adres bildiriyorsa onu kaydedin. `7860` Gradio ile React testlerini karıştırmayın. `address already in use` çıkarsa yeni sunucu başlamamıştır; kendi eski sunucunuzu çalıştığı terminalde Ctrl+C ile durdurun veya mevcut olanı kullanın. Bilmediğiniz süreci öldürmeyin. Portu kimin tuttuğunu görmek için:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

Model anahtarı/checkpoint eksikse bunu önkoşul hatası olarak kaydedin. Bu aşamada bağımlılık silmeyin, ağırlık indirmeyin, `.env` içeriğini paylaşmayın. Gerçek rapor denemesi yapılandırılmış uzak LLM/VLM servisine video kareleri gönderebilir; yalnız paylaşmaya yetkili olduğunuz test videolarını kullanın.

## 2. İlk deneme: MAN-G01, mevcut trafik videosu

### 2.1 Kanıt klasörünü açma

İkinci terminalde:

```bash
.venv/bin/python scripts/testing/prepare_manual_run.py --scenario MAN-G01 --video "/Users/ahmetcan/Downloads/WhatsApp Video 2026-08-27 at 22.47.15.mp4" --out "_stuff/test_runs/manual-G01-01"
```

Bu komut yalnız dosya hash'i/metadata ve mevcut commit bilgisini kaydeder; modeli çalıştırmaz, video yüklemez. `run.json` ve doldurulacak `notes.md` üretir. Dosya taşındıysa gerçek yolunu kullanın. Klasör zaten varsa `manual-G01-02` gibi yeni isim seçin; eski kanıtı ezmeyin. Metadata okunamadıysa analize geçmeden hatayı paylaşın. Bu yardımcı OpenCV metadata kontrolü yapar; ses/rotation/VFR doğrulaması yapmaz.

### 2.2 İnsan referansını önce yazma

1. Kaynak videoyu yerel oynatıcıda açın. Model sonucunu okumadan olayın başladığı/bittiği yaklaşık saniyeleri not edin.
2. Görünen araçların hareket yönünü, yakınlaşmayı ve gerçekten görünür bir temas olup olmadığını ayrı yazın. Görünmeyen teması veya sürücünün niyetini varsaymayın.
3. `notes.md` içine gözleminizi ve emin olmadığınız noktaları yazın. Gerekirse ilgili saniyede ekran görüntüsü alın. Dikey görüntüde araç/en-boy oranını referans olarak saklayın.

### 2.3 Arayüzde uygulama

1. React sayfasında **Yeni sohbet** seçin. Başka iş çalışıyorsa bitmesini bekleyin.
2. Soldaki **Hedef video → Dosya Seç** ile yukarıdaki dosyanın aynısını yükleyin.
3. Dosya adı ve video önizlemesi görünene kadar bekleyin. Önizlemeyi oynatıp doğru dosya olduğunu doğrulayın. Yükleme hatasında raporu başlatmayın.
4. **Video Raporu → Rapor oluştur** düğmesine **bir kez** basın. Bu sekmede ayrıca sohbet sorusu yazılmaz.
5. **Canlı süreç** altında sırayla planner, executor, gerçek tool çağrıları ve reviewer kayıtlarını izleyin. Planner'ın “aracı kullanacağım” demesi tool çalıştığı anlamına gelmez; çağrı argümanı ve tool sonucu bulunmalı.
6. İş bitene kadar sayfayı yenilemeyin/yeni sohbet açmayın. Başarıda rapor ve **JSON indir**, hatada hata metni beklenir. Beklenmedik uzun beklemede geçen süreyi not edin; sürekli tekrar tıklamayın.

### 2.4 Çıktıları saklama

1. Canlı süreçte **Tümünü kopyala** seçin. Hemen ardından ikinci terminalde aşağıdakini çalıştırın. Bu komut panodaki metni kaydeder; önce başka bir şey kopyalamayın:

```bash
pbpaste > "_stuff/test_runs/manual-G01-01/trace.txt"
```

2. JSON oluştuysa **JSON indir** seçin; indirilen dosyayı Finder ile aynı kanıt klasörüne taşıyın. Oluşmadıysa `notes.md` içine “JSON oluşmadı” yazın; boş JSON oluşturmayın.
3. Son ekranı ve hata varsa açık **Detay** bölümünü ekran görüntüsü olarak aynı klasöre koyun. Sunucu terminalindeki ilgili traceback'i `server-error.txt` adıyla kaydedin; anahtarları/özel verileri ayıklayın.
4. `notes.md` içine adresi, başlangıç/bitiş saatini, görünen hatayı ve insan gözlemiyle farkı yazın. Session/job ID görünüyorsa kaydedin; görünmüyorsa tahmin etmeyin. Gerekirse Chrome geliştirici araçları → Network altında başlatma isteğinin yanıtına bakın; tüm HAR dosyasını paylaşmayın.

### 2.5 Ne kontrol edeceğiz?

| Kontrol | Kanıtta aranacak şey | Aykırılık |
|---|---|---|
| Hedef | Tool argümanları yüklenen aynı videoyu gösteriyor | Başka video/yol: FAIL |
| Tarama | Başarılı `run_abnormal_event_segmenter`, süre, segment listesi | Sadece plan var: analiz yapılmadı |
| Görsel inceleme | `analyze_video_with_vlm` gerçek sonucu ve kaynak zamanları | “İncelendi” sözü tek başına yetmez |
| Geometri | VLM'ye giden örnek ile kaynak oranı | Önizlemenin doğru olması VLM girişinin doğru olduğunu kanıtlamaz; örnek yoksa bu alt kontrol INCONCLUSIVE |
| Anlam | Yön/yakınlaşma insan referansıyla uyumlu | Model yanlış yön söylüyorsa teknik başarıya rağmen FAIL |
| Sonuç | Doğrulanmış rapor veya dürüst hata | Analiz yokken `dusuk` risk yayımlamak FAIL |

İlk paylaşım: `run.json`, `notes.md`, `trace.txt`, varsa indirilen JSON ve son ekran. Kaynak videoyu tekrar göndermeniz gerekmiyor. Yerel yollar/plakalar/kişisel görüntüler için paylaşım öncesi kontrol yapın. **G01'i bitirip değerlendirmeden tüm P0 listesini topluca çalıştırmayın.**

## 3. G02–G08: rapor sorunlarını ayrı yakalama

Her senaryo için bölüm 2'nin kayıt/yükleme/başlatma/kanıt adımlarını yeni klasör ve senaryo ID ile tekrarlayın. Kullanılacak veri ve tam kabul koşulu ana plandadır.

| Senaryo | Yapılacak özel işlem | Sonuç nasıl yorumlanır? |
|---|---|---|
| G02 | Normal videoda rapor üretin; tool sonucunda `segments` boş mu bakın. Boşsa bundan sonra VLM çağrısı var mı inceleyin. | Segment çıktıysa sıfır-segment koşulu denenmedi: INCONCLUSIVE. Eşiği sırf sıfır elde etmek için değiştirmeyin. Boş segment + görsel kanıt yokken normal rapor FAIL. |
| G03 | Önizlemesi çalışan yüklenmiş video ile rapor üretin. Reviewer “video_path eksik” derse yükleme adı, tool çağrıları ve tüm feedback'i saklayın. | Reviewer sözü dosyanın gerçekten eksik olduğunun kanıtı değildir. Hiç tool çalışmadıysa decoder arızası diye sınıflandırmayın. |
| G04 | G03/başka doğal başarısızlıkta executor'un tool çağırmadan metadata hatası veya düşük risk üretip üretmediğini kontrol edin. | Taslak ile yayımlanan raporu ayırın. Validator reddetmişse son kullanıcıya hatalı rapor verilmemesi ayrı bir olumlu kontroldür. |
| G05 | Başarısız işin son reviewer etiketi, final metni ve sayfa durumunu birlikte kaydedin. | “final answer approved” ile “doğrulanamadı” birlikteyse etiket FAIL; koruyucu son hata mesajı bunu ortadan kaldırmaz. |
| G06 | Birden fazla aralık üreten videoda her segmentin başlangıç/bitişini bir satıra yazın, yanına VLM aralıklarını koyun. | Birleşik görsel kapsamı karşılaştırın; yalnız çağrı sayısını saymayın. Eksik aralığı saniyesiyle belirtin. |
| G07 | Kategorisi belirsiz görünür olayı kullanın; insan gözlemi ve rapor kategorisini karşılaştırın. | Belirsizliği zorla kesin suç/kategoriye çevirmek başarısızlık. |
| G08 | Normal videoda eylemler listesini gerçek tool kayıtlarıyla karşılaştırın. | Öneri ile gerçekleştirilmiş dosya işlemini ayırın; gereksiz arşiv/plaka işlemini kaydedin. |

G03–G05 doğal olarak oluşmazsa “geçti” demeyin. Aynı isteği sınırsız tekrar ederek hata aramayın; kontrollü açık testlerini bölüm 8'deki `gaps` grubuyla ayrıca çalıştıracağız.

## 4. U01–U08: oturum ve iptal uygulaması

**U01:** A videosuyla rapor/sohbet üretin ve kanıtları alın. İş bitince Yeni sohbet → görsel olarak farklı B videosu yükleyin → rapor üretin. B tool yollarında, olaylarında, indirilen JSON'da A'ya ait veri olmamalı. Aynı oturumda video değiştirme düğmesinin kapalı olması beklenen davranıştır; bu engeli aşmayın.

**U02:** İki ayrı tarayıcı sekmesinde aynı uygulamayı açın, her birinde Yeni sohbet kullanıp A ve B'yi yükleyin. Önce A'da sonra B'de rapor başlatın. Her sekmenin trace/JSON'unu ayrı klasöre kaydedin. İki işi aynı videoyla denemek karışmayı ayırt etmeyi zorlaştırır. Fiziksel model kaynak kilidi nedeniyle bekleme olabilir; beklemeyi veri karışmasıyla eşitlemeyin.

**U03:** Rapor oluştur'a hızlı çift tıklayın; tek sekmede aynı iş için birden fazla başlatma kabulü olup olmadığını Network ve trace'ten kontrol edin. Sonra bağımsız sekmelerde birer kez başlatın. Ayrı oturumların ayrı job oluşturması normal; aynı oturumda aynı anda iki iş kabulü sorun. Butonun kapanması tek başına backend kilidi kanıtı değildir.

**U04/U05:** Uzunca bir video ile Analyzer veya Rapor başlatın. Tool/model işlemi başladıktan sonra İptal'e bir kez basın; saatini not edin. Yeni işe geçişin ne zaman açıldığını, eski sonucun sonradan ekrana gelip gelmediğini kaydedin. İptal native/model çağrısını anında kesmeyebilir; fakat iptal edilmiş rapor başarı gibi yayımlanmamalı. Üretilmiş dosyaların silinmesini varsaymayın, yan etkileri listeleyin.

**U06:** Chrome geliştirici araçları → Network → Offline ile yalnız tarayıcı bağlantısını kısa süre kesin; sunucuyu kapatmayın. Sonra No throttling'e dönün. Job'ın son durumu geliyor mu, trace atlanıyor mu, yineleniyor mu kontrol edin. Offline süresini yazın. Bunu ilk başarılı temel denemeden sonra yapın.

**U07:** Ayrı denemede iş çalışırken sayfayı yenileyin. Yenileme sonrası oturum/video durumunu kaydedin; önceki job'ın yeni oturuma karışmaması gerekir. **U08:** Gradio denemesini ayrı kaydedin; React ve Gradio'nun aynı oturum/memory sözleşmesini paylaştığını varsaymayın.

## 5. R/E/Z: JSON, eylem ve hedef zincir

1. R01/R02 için indirilen JSON'u metin düzenleyicide açın: yalnız JSON mu, olay zamanları süre içinde mi? Raporu trace'teki görsel kanıtla eşleyin. Ekrandan kopyalanmış metin yerine indirilen asıl dosyayı saklayın.
2. R06/R07 için her iddiayı kaynağa bağlayın: anomali skoru suç olasılığı değildir; görünmeyen plaka/kimlik veya analiz edilmemiş ses hakkında kesin iddia bulunmamalı.
3. R08 için Sohbet'te ayırt edilebilir bir istek gönderin, sonra Video Raporu üretin; raporun sohbet cevabını veya önceki videoyu devralmadığını kontrol edin.
4. E01–E06 için `eylemler` satırlarının karşısına gerçek tool adı, çağrı/sonuç, başarı/hata ve çıktı dosyasını yazın. “Kaydedildi” ifadesini yalnız dosya açılabiliyor ve aynı videoya aitse doğrulayın.
5. Z01–Z08'de **önce yalnız Video Raporu** çalıştırın; Sohbet'ten elle araç çağrısı eklemeyin. Hangi aracın neden çalıştığı/atlanmış olduğu ve olay→araç→plaka→OCR bağlantıları var mı kontrol edin.
6. Mevcut kodda takip kutusunun plaka tespitine aktarılması ve track ID'nin OCR'ye bağlanması tam zincir olarak sağlanmıyor. Bu bağlar yoksa Z03/Z04 **BLOCKED**; araçları ayrı çalıştırmak zinciri PASS yapmaz. Olmayan otomasyonu test etmek için kod eklemeyin.

R03–R05 ve E03/E04 gibi sahte tool sonucu/bozuk JSON/çağrı eşleşmesi testleri arayüzden deterministik üretilemez. Gerçek servis cevabını bozmayın; bölüm 8'de sözleşme/açık testleri kullanılır. Manuel testin karşılanmayan kısmını BLOCKED ve gerekçesiyle kaydedin.

## 6. S/M/O/P/C/A: araçları tek tek deneme

Bu tur zincir kabulü değil, bileşen kontrolüdür. Her veri setinde önce doğru videoyu yükleyin. Aşağıdaki metinler **Sohbet kutusuna** yazılır; terminal komutu değildir. Örnek 2–4 saniye aralığı yalnız video 4 saniyeden uzunsa ve seçilen nesne o aralıkta görünüyorsa kullanılmalı; aksi halde gözlemlediğiniz gerçek aralığı yazın.

| Bileşen | Sohbet isteği örneği | Mutlaka incelenecek kanıt |
|---|---|---|
| Metadata | Yüklü hedef video için yalnız get_video_info çalıştır, süre/fps/boyutları göster. | Kaynak dosya bilgileriyle karşılaştırma |
| Segmenter | Yüklü videonun tamamında yalnız run_abnormal_event_segmenter çalıştır; skor ve aralıkları göster. | Süre, threshold, segment başlangıç/bitişleri; Analyzer sekmesiyle aynı ayar karşılaştırması |
| VLM | Yüklü videonun 2–4 saniyesini analyze_video_with_vlm ile incele; yalnız görünür hareketi anlat, belirsizliği belirt. | Requested/effective/sampled range, örnek sayısı ve insan referansı |
| Tracking | Yüklü videonun 2–4 saniyesinde detect_and_track_objects ile araçları takip et ve kutulu çıktı üret. | Gerçek çağrıdaki sınıflar, ID/kutu/zamanlar; çıktı videosunu oynatma |
| Plaka tespiti | Yüklü videonun 2–4 saniyesinde detect_license_plate_regions çalıştır; manifest ve kırpım dosyalarını göster, henüz OCR yapma. | Tam manifest, gerçek PNG'ler, kaynak kare üzerindeki kutular |
| OCR | Bir önceki başarılı tespitin döndürdüğü crops manifestini read_license_plate_crops ile oku; yeni tespit yapma. | Gerçek manifest yolu, hash bağı, text/candidate/belirsizlik |
| Genel klip | Yüklü videonun 2–4 saniyesini save_video_segment ile yeni bir çıktı dosyasına kaydet. | Aynı kaynak değil yeni dosya, oynatma ve süre |
| Arşiv | Görsel olarak doğrulanmış olayın aralığını archive_anomaly_clip ile uygun kategori ve gözleme dayalı gerekçeyle arşivle. | Gerçek çağrıdaki kategori, metadata, hash, oynatılabilir klip |

LLM istenen aracı çağırmazsa bunu “araç hatası” diye yazmayın: yönlendirme gerçekleşmedi. Trace'teki gerçek parametreleri kaydedin. Arşiv için normal bir kesiti yapay biçimde suç diye etiketlemeyin; olay aralığını ve desteklenen kategoriyi ana plandan seçin.

**O/P/C görsel kontrol yöntemi:** Önce kaynak kareyi ilgili saniyede durdurun. Kutunun doğru araç/plakayı çevrelediğini, koordinatların görüntü dışına taşmadığını karşılaştırın. Manifestteki PNG'yi açın; insanın okuyabildiği metni OCR sonucunu görmeden yazın. Aynı plakanın tekrar kırpımlarını farklı araç sanmayın. Preview listesi kesilmişse tam manifesti inceleyin; preview sayısını toplam sonuç sanmayın. Çıktı dosyası yoksa yalnız yol gösterilmesi başarı değildir.

**S/M zaman kontrolü:** Başlangıç, orta ve son kareyi karşılaştırın. Kaynak zamanını geçici klibin 0 tabanlı zamanı ile karıştırmayın. VFR/rotation/ses testinde metadata tahmini tek başına yeterli değildir; uygun referans video yoksa BLOCKED. Sesli çıktı testi ses analiz edildiği anlamına gelmez.

**Negatif testler:** Checkpoint silme, manifest bozma, kaynak üstüne yazma, disk doldurma, servis kesme gibi işlemleri günlük çalışma ortamında yapmayın. Bunlar disposable veri/izole ortam veya mock gerektirir. O/P/C/A senaryosunun normal örneğini geçmek, hata kolunu da geçtiğiniz anlamına gelmez. Ortam hazırlanmamışsa ilgili hata kolunu BLOCKED bırakın.

## 7. X/T/Q ve sonuç kaydı

X için uygulama sürümü/adres/başlatma komutu, bağımlılık hatası ve uzun denemelerde süre/bellek/çıktı birikimini kaydedin. Ölçmediğiniz performansı “normal” diye onaylamayın. T senaryoları veri ayrımı/kod incelemesiyle başlar; eğitim scriptini import ederek test etmeyin, eğitim başlatmayın. Q için aynı videoyu yeni oturumlarda tekrarlayıp insan referansına göre hataları sayın; tek doğru cevap genelleme başarısı değildir.

Her denemede `notes.md` içine şu satırları doldurun:

```text
Senaryo: MAN-G01
Deneme: manual-G01-01
Durum: FAIL / PASS / BLOCKED / INCONCLUSIVE
Yaptığım işlem:
Beklenen:
Gözlenen:
Kanıt dosyaları:
Session/job ID (varsa):
Sonraki adım:
```

Sonra [sonuc_kaydi.csv](sonuc_kaydi.csv) ilgili satırına durum/run_id/kanıt yolunu girin. Tekrarların ayrıntıları kendi klasörlerinde kalır; CSV son değerlendirmeyi özetler. `run.json` başlangıçta CALISTIRILMADI kalır; değerlendirme sonunda onu da aynı durumla güncelleyin. Hazırlık komutunun başarılı olması senaryonun PASS olması değildir.

## 8. Otomatik testleri çıktılarıyla çalıştırma

Manuel ilk turu değerlendirdikten sonra aşağıdaki grupları **tek tek** çalıştırın, ilk grubun çıktısını paylaşın. Script gerçek çıkış kodunu korur; hata olursa otomatik düzeltme/paket kurulumu yapmaz. Her grup `_stuff/test_runs/` altında benzersiz klasöre `output.log` ve `summary.json` yazar; paylaşılacak tam yolu terminal sonunda gösterir.

Mevcut regresyonlar:

```bash
.venv/bin/python scripts/testing/run_checks.py existing
```

Yeni izole sözleşme testleri:

```bash
.venv/bin/python scripts/testing/run_checks.py contracts
```

Bilinen açıkların hedef davranış testleri:

```bash
.venv/bin/python scripts/testing/run_checks.py gaps
```

Frontend lint ve ardından build (lint başarısızsa build çalışmaz):

```bash
.venv/bin/python scripts/testing/run_checks.py frontend
```

`gaps` grubunda mevcut açıklar nedeniyle assertion başarısızlığı beklenir; bu sonuçları gizlemeyin. `ModuleNotFoundError` ise açık doğrulanmış değildir, kurulum engelidir. `contracts` model doğruluğunu, `frontend` ise tarayıcı davranışını kanıtlamaz. Frontend komutu derleme/cache dosyaları oluşturabilir. Hiçbiri git add/commit/push yapmaz. Ayrıntılı kapsam ve tek dosya komutları [otomatik rehberde](03_OTOMATIK_TESTLER.md).

**Paylaşılacak:** grubun tam `output.log` dosyası ve `summary.json`; yalnız son “FAILED” satırı yeterli değil. Dosyalar yerelde kalır, otomatik olarak gönderilmez. Bu rehber hazırlanırken testler çalıştırılmadı.
