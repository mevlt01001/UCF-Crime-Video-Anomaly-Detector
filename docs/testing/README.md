# Proje kabul ve regresyon test paketi

**Hazırlanma:** 28 Ağustos 2026. **Durum:** senaryolar ve yeni test kodları hazırlandı; bu paket için test yürütülmedi. Sözdizimi/doküman tutarlılık kontrolü test geçişi değildir.

## Önce hangi dosyayı açmalıyım?

**Uygulamaya buradan başlayın: [Adım adım uygulama rehberi](05_ADIM_ADIM_UYGULAMA.md).** Sunucu komutları, ilk gerçek video denemesi, ekran adımları, beklenen sonuçlar, kanıt kaydı ve test komutları burada. [Manuel kayıt hazırlayıcı](../../scripts/testing/prepare_manual_run.py) ve [test/çıktı çalıştırıcı](../../scripts/testing/run_checks.py) dosyaları repoya eklendi; çalıştırılmadı.

1. [Mimari, mevcut yetenekler ve açıklar](01_MIMARI_VE_ACIKLAR.md): hangi parça ne yapıyor, neyi henüz garanti etmiyor?
2. **[Manuel kabul senaryoları](02_MANUEL_TEST_PLANI.md): ana çalışma belgesi.** Önce P0/genel akış, sonra zincir ve araç ayrıntıları.
3. [Otomatik testlerin kapsamı ve çalıştırılması](03_OTOMATIK_TESTLER.md): daha sonra, ayrı bir aşamada çalıştırılacak.
4. [Sonuç kayıt tablosu](sonuc_kaydi.csv): her senaryo başlangıçta `CALISTIRILMADI`.
5. [Kapsam matrisi](04_KAPSAM_MATRISI.md): hangi test neyi kanıtlar, hangi alan henüz manuel veya eksik?
6. [Veri/kanıt kayıt şablonu](veri_kaydi.json): video hash'i ve insan referansı olmadan model doğruluğu ölçülmez.

## Test sırası ve durma koşulu

| Kapı | Kontrol | Sonraki aşamaya geçme koşulu |
|---|---|---|
| 0 | Ortam, doğru arayüz/sunucu, kaynak kopyaları | Kurulum ve test verileri hazır; gerçek anahtarlar logda yok |
| 1 | G01–G08: görüntü, hedef yol, araç çalıştırma, gerçek başarısızlık | P0 bulgu kapalı; eksik analiz düşük risk diye yayımlanmıyor |
| 2 | U01–U08: oturum, iptal, arayüz ve SSE | Yanlış video/oturum sonucu, sahte başarı, takılı iş yok |
| 3 | R01–R08 ve E01–E06: rapor/eylem kanıtı | Şema, kapsam, çağrı-sonuç ve eylem kayıtları doğru |
| 4 | Z01–Z08: hedeflenen zincir | Eksik özellik açıkça BLOCKED; manuel ayrı tool başarısı otomatik zincir başarısı sayılmaz |
| 5 | S/M/O/P/C/A: araç ve model ayrıntıları | Dosya, koordinat, zaman, belirsizlik ve hata sınırları doğrulanmış |
| 6 | X/T/Q: işletim, eğitim, kalite | Uygulama kapsamına uygun sonuçlar kaydedilmiş |

**İlk oturum:** G01, G02, G03, G04, G05, U01, U02, R01, E01, Z01, Z02, Z03. G01–G05'teki açıklar kapanmadan “rapor güvenilir” kabulü vermeyin. Zincir eksikse G/R/O/P/C/A testleri ayrı yürütülebilir; zincir testinin sonucu yine BLOCKED kalır.

## Kapsam sınırları

- Bu çalışma uygulama kodunu düzeltmez, ağırlık indirmez, model çağırmaz, eğitim başlatmaz ve Git add/commit/push yapmaz.
- Önceden geçen 22 test, bu yeni paketin veya yeni uçtan uca zincirin geçtiği anlamına gelmez.
- Kullanıcının kapsam dışında bıraktığı dış erişim/yükleme yolu güvenliği bu yerel kabulün geçiş koşulu değildir. Açık kapanmış sayılmaz; ağ erişimi açılmadan önce ayrıca test edilmelidir.
- Çalışma başında `SAVE_VIDEO_SEGMENT_TEST_SCENARIOS.md` yerelde silinmişti. Silme işlemi korunmuştur; bu paket dosyayı geri getirmez. Kayıt testleri burada A05/A06 altında yer alır.
- Eski senaryolardaki `+` işaretleri geçmiş sürümün sonuçlarıdır; otomatik olarak bu tabloya taşınmaz.
- Başka bir modelin “normal” demesi referans etiket değildir. İnsan gözlemi, kaynak kare, zaman ve dosya kanıtı gerekir.

## Sonuç durumları

`CALISTIRILMADI`: deneme yok. `PASS`: tüm beklenenler kanıtla sağlandı. `FAIL`: gözlenen aykırılık var. `BLOCKED`: eksik özellik/önkoşul yüzünden test tamamlanamadı. `INCONCLUSIVE`: model/insan kanıtı karar için yetersiz. `N/A`: yalnız kapsam gerekçesiyle uygulanamaz; bir hatayı gizlemek için kullanılamaz.

“HTTP 200”, “ok=true”, “dosya yolu var”, “reviewer onayladı” ve “test exception vermedi” tek başına PASS değildir. Yanlış araca plaka bağlamak, analiz yapmadan düşük risk vermek veya başka oturumun sonucunu göstermek P0 başarısızlıktır.
