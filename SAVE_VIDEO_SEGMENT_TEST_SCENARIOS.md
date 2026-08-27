# Klip Kaydetme Tool — Test Senaryoları

Test edilen tool: `save_video_segment(video_path, start_sec, end_sec, output_filename)`

## Ön koşul

```bash
ffmpeg -version
```

FFmpeg kurulu değilse beklenen hata:

```json
{"ok": false, "data": {}, "warnings": [], "error": {"code": "FFMPEG_NOT_FOUND", "message": "FFmpeg bulunamadı veya çalıştırılamadı."}}
```

## Senaryolar

| # | Test | Girdi | Beklenen |
|---|---|---|---|
| 1 | Normal kesme | `10–20 sn`, `clip.mp4` | Dosya oluşur, açılır ve yaklaşık 10 saniyedir. |
| 2 | Uzantısız ad | `olay_kesiti` | Dosya `olay_kesiti.mp4` olarak oluşturulur. |
| 3 | Kaynak yok | Olmayan video yolu | `ok=false`, `error.code=FILE_NOT_FOUND`. |
| 4 | Negatif başlangıç | `start=-5`, `end=10` | `ok=false`, `error.code=INVALID_TIME_RANGE`. |
| 5 | Eşit zaman | `start=10`, `end=10` | `ok=false`, `error.code=INVALID_TIME_RANGE`. |
| 6 | Ters zaman | `start=20`, `end=10` | `ok=false`, `error.code=INVALID_TIME_RANGE`. |
| 7 | Boş çıktı adı | `output_filename=""` | `ok=false`, `error.code=INVALID_OUTPUT_PATH`. |
| 8 | Klasör yok | `/olmayan/klasor/clip.mp4` | `ok=false`, `error.code=INVALID_OUTPUT_PATH`. |
| 9 | Bozuk kaynak | Bozuk `.mp4` dosyası | `ok=false`; başarı sonucu dönmez. |
| 10 | Yazma izni yok | Korunan klasöre çıktı | Hata döner; başarı denmez. |
| 11 | Geçersiz çıktı | Boş/okunamayan çıktı | `dosya boş` veya `geçerli video olarak okunamadı` hatası döner. |
| 12 | Aynı dosya adı | Var olan `clip.mp4` | `-y` nedeniyle dosyanın üzerine yazılır ve yeni dosya açılabilir. |
| 13 | Agent seçimi | “Anomaliyi `anomali.mp4` olarak kaydet” | Agent `save_video_segment` tool'unu çağırır. |
| 14 | Başlangıç video dışında | 17 sn video, `start=40`, `end=45` | `ok=false`, `error.code=TIME_OUT_OF_RANGE`; FFmpeg çalışmaz. |
| 15 | Yalnız bitiş taşıyor | 17 sn video, `start=12`, `end=25` | `ok=true`, `saved_range.end_sec=17`; `END_TIME_CLAMPED` uyarısı döner. |

## Başarı kriterleri

- FFmpeg hata vermemeli.
- Çıktı dosyası oluşmalı ve boş olmamalı.
- Video açılmalı ve en az bir kare okunabilmeli.
- Başarıda `ok=true`, `error=null` ve `data.output_path` dolu olmalı.
- Hatalarda `ok=false` olmalı; Agent sonucu başarılı gibi sunmamalı.
