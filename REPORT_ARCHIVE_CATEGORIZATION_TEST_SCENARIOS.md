# Report modu — zorunlu kategorili arşiv manuel testleri

Kapsam: **Video Raporu** sekmesi, report-only `archive_anomaly_clip` davranışı,
`utils/reporting.py` (`REPORT_RULES`, `validate_report`) ve `utils/agents.py`
(mod bazlı tool seti).

**Önkoşullar (tüm senaryolar):**

- Uygulama yeniden başlatılmış olsun (`run_web.py` veya eşdeğeri).
- FFmpeg kurulu olsun.
- React arayüz: `http://127.0.0.1:8000/` → **Video Raporu** sekmesi.
- Test videoları için `docs/testing/02_MANUEL_TEST_PLANI.md` içindeki V01–V13
  tanımları veya eşdeğer kısa olay videoları kullanın.
- Kanıt: session/job ID, **tam node trace** (tool adı + parsed JSON), nihai rapor
  JSON, `output_path` altındaki dosyaların varlığı. Yalnız sohbet/rapor metni
  “kaydedildi” demesi yeterli değildir.

**Kategoriler:** `hirsizlik`, `soygun`, `kavga_saldiri`, `trafik_kazasi`,
`is_kazasi`, `diger`, `belirsiz`.

**Önemli ürün kuralı (bu belgenin odağı):**

| Mod | `archive_anomaly_clip` |
| --- | --- |
| **Video Raporu** | Segmenter anomali döndürdüyse **her segment için zorunlu** (VLM sonrası) |
| **Sohbet** | Araç **yok**; kategorili arşiv yapılmaz |

Teknik arşiv sözleşmesi (FFmpeg, cache, çakışma kodları) için:
[ARCHIVE_ANOMALY_CLIP_TEST_SCENARIOS.md](ARCHIVE_ANOMALY_CLIP_TEST_SCENARIOS.md).

Rapor `eylemler` kaydı için:
[REPORT_ACTIONS_TEST_SCENARIOS.md](REPORT_ACTIONS_TEST_SCENARIOS.md)
— segment varken `eylemler=[]` artık geçerli değildir.

---

## En temel senaryo — MAN-RA00 · P0

**Amaç:** Report akışının “anomali → görsel inceleme → kategorili arşiv → rapor”
zincirini uçtan uca doğrulamak.

**Veri:** En az bir anomali segmenti üreten kısa video (V05 veya benzeri; süre
tercihen &lt; 60 sn).

**Uygulama:**

1. Yeni oturum açın; videoyu yükleyin.
2. **Video Raporu** sekmesinde yalnız **Rapor oluştur**'a basın; ek prompt yazmayın.
3. Trace'i izleyin: `run_abnormal_event_segmenter` → her segment için
   `analyze_video_with_vlm` → **her segment için** `archive_anomaly_clip`.
4. Nihai JSON'u ve disk çıktısını kontrol edin.

**Beklenen / geçiş koşulu:**

| Kontrol | Beklenen |
| --- | --- |
| Segmenter | `ok=true`, `segments` en az 1 kayıt |
| VLM | Her segment aralığı görsel olarak kapsanmış |
| Arşiv | Her segment için bir `archive_anomaly_clip` çağrısı (başarılı veya `[BASARISIZ]` kayıtlı) |
| Kategori | Görsel kanıta uygun enum; emin değilse `belirsiz`, listede yoksa `diger` |
| Dosya | `_stuff/lab_runs/actions/archive/<kategori>/<hash>/clip.mp4` + `metadata.json` (başarılı arşivde) |
| Rapor `eylemler` | Gerçek tool kayıtlarıyla birebir; uydurma `[BASARILI]` yok |
| Rapor onayı | Validator geçer; JSON indirilebilir |

**FAIL örnekleri:** Segment var ama arşiv çağrısı yok; arşiv yalnız Sohbet'te
yapılmış ama raporda yok; kategori enum dışı; trace'te olmayan işlem raporda
`[BASARILI]` yazılı.

---

## 1 — Report zorunlu arşiv (P0)

### MAN-RA01 · P0 — Çok segmentli video: segment başına bir arşiv

**Veri:** V05 (en az 3 anomali aralığı).

**Uygulama:** MAN-RA00 ile aynı; segment sayısını trace'ten sayın.

**Beklenen:** N segment → N arşiv denemesi. Her biri segment `start_time`–
`end_time` ile uyumlu (bitiş video sonunda kırpılmış olabilir, `END_TIME_CLAMPED`
uyarısı kabul).

**FAIL:** Yalnız bir segment arşivlenmiş; diğerleri atlanmış → rapor reddedilmeli
veya validator hatası trace'te görünmeli.

---

### MAN-RA02 · P0 — Belirsiz olay: `belirsiz` kategorisi yeterli

**Veri:** V06 belirsiz/örtülü sürüm veya olay türü net seçilemeyen video.

**Uygulama:** Rapor oluşturun; VLM metninde tür net değilse arşiv kategorisine bakın.

**Beklenen:** Arşiv **yine çalışır**; kategori `belirsiz` (veya kanıtla desteklenen
başka enum). “Arşivlemedim çünkü emin değildim” geçerli değil.

**FAIL:** Segment var, arşiv yok; veya kategori boş/uydurma.

---

### MAN-RA03 · P0 — Listede olmayan olay: `diger`

**Veri:** İnsan etiketi “olay var ama 7 kategoriden hiçbiri tam uymuyor” olan kesit.

**Uygulama:** Rapor oluşturun; kategori ve `explanation` gerekçesini okuyun.

**Beklenen:** `diger` + kanıta dayalı kısa gerekçe; klip yine kaydedilir.

---

### MAN-RA04 · P0 — Sıfır segment: arşiv zorunlu değil

**Veri:** V04 veya segmenter'ın boş döndüğü doğrulanmış video (trace:
`segments=[]`, uyarı `NO_SEGMENTS_ABOVE_THRESHOLD`).

**Uygulama:** Rapor oluşturun.

**Beklenen:** `archive_anomaly_clip` **çağrılmaz**. Rapor yine üretilebilir;
`eylemler=[]` bu durumda geçerlidir. “Tüm video güvenli” iddiası yapılmamalı;
kapsam `ozet`'te açıklanmalı.

**FAIL:** Segment yokken arşiv çağrısı (gereksiz); veya boş segmentten doğrudan
düşük risk / güvenli raporu.

---

### MAN-RA05 · P0 — Arşiv başarısız, görsel kapsam tam: rapor yine geçebilir

**Veri:** Geçerli anomali videosu; FFmpeg'i geçici olarak devre dışı bırakan
**izole test ortamı** (üretim `.env` veya sistem FFmpeg'ini bozmayın).

**Uygulama:** Arşiv adımı `FFMPEG_NOT_FOUND` veya benzeri ile `[BASARISIZ]` dönsün;
segmenter + VLM tamamlansın.

**Beklenen:** Rapor `eylemler`'de `[BASARISIZ] archive_anomaly_clip (...)` satırı;
görsel kapsam tamamsa nihai JSON yine onaylanabilir. Hata `ozet` veya eylem
özetinde görünür.

**Not:** Kontrollü ortam kurulamazsa **BLOCKED**; PASS sayılmaz.

---

### MAN-RA06 · P0 — Eksik arşiv: rapor reddedilmeli

**Veri:** V05; mümkünse agent'ın arşiv atladığı bir koşul (ör. tool tur sınırına
yakın çok segmentli video — V12).

**Uygulama:** Trace'te en az bir segment için arşiv yokken rapor taslağı üretilmeye
çalışılsın.

**Beklenen:** `validate_report` hatası: `... aralığı için kategorili arşiv
tamamlanmadı.` Rapor indirilemez / onaylanmaz; reviewer geri bildiriminde validator
mesajı.

---

## 2 — Sohbet izolasyonu (P0)

### MAN-RA10 · P0 — Sohbette arşiv aracı yok

**Veri:** MAN-RA00 ile aynı video.

**Uygulama:** **Sohbet** sekmesinde örnek mesajlar deneyin:

- “2–5 saniyeyi kavga olarak arşivle”
- “Anomali bulunan kesiti kategorili arşive kaydet”
- “Rapordaki gibi arşivle”

**Beklenen:** Trace'te **`archive_anomaly_clip` hiç görünmez**. Planner/executor
kataloğunda da bu araç olmamalı (izinli debug ile). Model başka araçlar
(segmenter, VLM, save_video_segment vb.) kullanabilir.

**FAIL:** Sohbette `archive_anomaly_clip` çağrısı veya `_stuff/.../archive/` altında
sohbet kaynaklı yeni klip.

---

### MAN-RA11 · P0 — Sohbette yapılan işlem rapora yansımaz

**Veri:** Aynı video, aynı oturum.

**Uygulama:**

1. Sohbet'te segmenter + VLM çalıştırın (arşiv olmadan).
2. Ardından **Rapor oluştur**'a basın.

**Beklenen:** Rapor kendi trace'ini üretir; Sohbet'teki tool sonuçları rapor
`eylemler`'ine otomatik taşınmaz. Rapor segment bulursa kendi arşiv adımlarını
çalıştırır.

---

## 3 — Kategori ve gerekçe kalitesi (P1)

### MAN-RA20 · P1 — Kategori görsel kanıtla uyumlu

**Veri:** İnsan etiketli V05/V06.

**Uygulama:** Rapor sonrası VLM metni, seçilen kategori ve `explanation`'ı karşılaştırın.

**Beklenen:** Bariz çelişki yok (ör. VLM “normal trafik” derken `kavga_saldiri`).
Kod kategori doğruluğunu otomatik denetlemez; **insan FAIL** verilebilir.

---

### MAN-RA21 · P1 — Anomali skoru ≠ suç kategorisi

**Veri:** Yüksek anomali skoru ama görsel olarak sıradan faaliyet (V01 benzeri).

**Uygulama:** Rapor oluşturun; kategori yalnız skora dayanıyor mu kontrol edin.

**Beklenen:** Kategori VLM bulgusuna dayanır; skor tek başına `yuksek` risk veya
ağır kategori gerekçesi olmamalı.

---

### MAN-RA22 · P1 — `metadata.json` ile rapor tutarlılığı

**Veri:** MAN-RA00 başarılı koşul.

**Uygulama:** `metadata.json` içindeki `category`, `explanation`, aralık ve
`output_path`'i rapor `eylemler` satırıyla karşılaştırın.

**Beklenen:** Birebir uyum; klip oynatıldığında süre segment aralığıyla
`max(0.15 sn, 2/fps)` toleransında uyumlu.

---

## 4 — Teknik arşiv regresyonları (report bağlamında, P1)

Bu maddeler [ARCHIVE_ANOMALY_CLIP_TEST_SCENARIOS.md](ARCHIVE_ANOMALY_CLIP_TEST_SCENARIOS.md)
ile örtüşür; report akışında **Sohbet yerine Rapor** üzerinden tetiklenir.

### MAN-RA30 · P1 — Cache: aynı rapor/job tekrarı

**Uygulama:** Aynı video için raporu ikinci kez üretin (aynı segmentler).

**Beklenen:** İkinci arşiv çağrısında `cache_hit=true`, aynı `output_path`, ilk
`explanation` korunur.

---

### MAN-RA31 · P1 — Kategori çakışması

**Uygulama:** Aynı kesit daha önce `belirsiz` ile arşivlendiyse, kontrollü ortamda
aynı kesiti farklı kategoriyle tekrar arşivlemeye zorlayın (Sohbet değil — yalnız
report tool seti ile izole test).

**Beklenen:** `ARCHIVE_CATEGORY_CONFLICT`; ikinci kategori klasörü oluşmaz.

---

### MAN-RA32 · P1 — Geçersiz girdi reddi

**Senaryolar:** enum dışı kategori, boş/`>2000` karakter gerekçe, geçersiz zaman
aralığı, kaynak dosya yok.

**Beklenen:** `ok=false` ve ilgili hata kodu; sahte `output_path` yok.

---

## 5 — Rapor kaydı ve öneriler (P0–P1)

### MAN-RA40 · P0 — `eylemler` gerçek arşiv kayıtlarıyla eşleşir

**Beklenen format:**

`[BASARILI] archive_anomaly_clip (<call_id>): Klip kaydedildi; kategori=...;
{start}–{end} sn; dosya=<output_path>`

**FAIL:** Uydurma call_id; sıra değişikliği; eksik/fazla satır → validator reddi.

---

### MAN-RA41 · P1 — `[ONERI]` arşivin yerine geçmez

**Uygulama:** Rapor `eylemler`'e `[ONERI] ... arşivlenmelidir` eklenmiş taslak
(gerçek arşiv kaydı varken).

**Beklenen:** Öneri ek satır olabilir; gerçek `[BASARILI]` kayıtları aynen durur.
Öneri tek başına arşiv yerine geçmez.

---

### MAN-RA42 · P0 — Başarılı arşiv varken “yeniden arşivlenmeli” önerisi

**Uygulama:** MAN-RA00 sonrası rapor metninde aynı aralık için tekrar arşiv
`[ONERI]` var mı bakın.

**Beklenen:** Aynı hedef/aralıkta `[BASARILI]` varken tekrar arşiv önerilmemeli
(prompt kuralı).

---

## 6 — Kapasite ve sınır senaryoları (P1–P2)

### MAN-RA50 · P1 — Çok segment + tool tur bütçesi

**Veri:** V12 (uzun / çok segment).

**Uygulama:** Trace'te tool-node tur sayısını (`MAX_TOOL_ROUNDS=8`) ve pending
çağrı kapanmalarını izleyin.

**Beklenen:** Tüm segmentler VLM + arşiv ile tamamlanır **veya** MAN-RA06 gibi
validator reddi net görülür. Sessiz eksik arşiv FAIL.

---

### MAN-RA51 · P2 — İptal davranışı

**Uygulama:** Arşiv FFmpeg çalışırken raporu iptal edin; yeni oturum açın.

**Beklenen:** İptal rollback değildir; tamamlanmış arşiv dosyaları silinmiş
gibi gösterilmez. Yarım kalan iş kanıtlanabilir durumda kalabilir.

---

## 7 — Hızlı kontrol matrisi

| ID | Özet | Mod | PASS anahtarı |
| --- | --- | --- | --- |
| MAN-RA00 | Altın yol: anomali → VLM → arşiv → rapor | Rapor | N segment = N arşiv; dosya + JSON |
| MAN-RA01 | Çok segment | Rapor | Her segment arşivli |
| MAN-RA02 | Belirsiz olay | Rapor | `belirsiz` ile arşiv |
| MAN-RA03 | `diger` kategori | Rapor | Klip kaydedildi |
| MAN-RA04 | Sıfır segment | Rapor | Arşiv yok; rapor olabilir |
| MAN-RA05 | Arşiv FAIL, VLM OK | Rapor (izole) | `[BASARISIZ]` + rapor geçebilir |
| MAN-RA06 | Eksik arşiv | Rapor | Validator reddi |
| MAN-RA10 | Sohbet izolasyonu | Sohbet | Arşiv tool yok |
| MAN-RA11 | Sohbet ≠ rapor trace | Her ikisi | Ayrı izler |
| MAN-RA20–22 | Kategori/kanıt kalitesi | Rapor | İnsan denetimi |
| MAN-RA30–32 | Teknik arşiv | Rapor | Hata kodları / cache |
| MAN-RA40–42 | `eylemler` sözleşmesi | Rapor | Birebir kayıt |
| MAN-RA50–51 | Bütçe / iptal | Rapor | Sessiz eksik yok |

---

## Sonuç kaydı

Her koşu için `docs/testing/sonuc_kaydi.csv` satırı açın:

- `senaryo_id`: MAN-RAxx
- `video`: V01–V13 kodu + hash
- `mod`: report / chat
- `segment_sayisi`, `arsiv_cagri_sayisi`, `eylemler_arsiv_sayisi`
- `sonuc`: PASS / FAIL / BLOCKED / INCONCLUSIVE
- `kanit`: trace dosyası, JSON, `output_path` listesi

**BLOCKED:** FFmpeg bozma, çok segmentli V12 yok, izole ortam kurulamadı.

Bu testler modelin olay türünü **doğru bildiğini** kanıtlamaz; report-only
arşiv kuralının ve kayıt sözleşmesinin çalıştığını kanıtlar.

## Video bazında arşiv ayrımı

- İki farklı videoda aynı kategori ve `olay_1` ile arşiv oluşturun. Beklenen:
  ayrı video klasörleri ve ayrı `incident.json`; her kayıt yalnız kendi kaynağını içerir.
- Aynı videonun aynı olayına ikinci kesit ekleyin: tek olay kaydında iki kesit olmalı.
- Aynı kesiti tekrar arşivleyin: yeni kopya yerine `cache_hit=true` dönmeli.
- Yeni düzen `kategori/video_kimliği/olay_id/klip_kimliği` şeklindedir.
  Eski klasörler silinmez veya taşınmaz; eski karışmış kayıtlar otomatik onarılmaz.
  Eski düzende kaydedilen kesitin yeniden istenmesi yeni düzende kopya oluşturabilir.
