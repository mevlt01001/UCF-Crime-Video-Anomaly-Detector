# Klip Kaydetme Tool — Test Senaryoları

Test edilen tool: `save_video_segment(video_path, start_sec, end_sec, output_filename)`

## Ön koşul

```bash
ffmpeg -version
```

FFmpeg kurulu değilse beklenen hata:

```text
Video kesme işlemi başarısız: FFmpeg bulunamadı veya çalıştırılamadı.
```

## Senaryolar

| # | Test | Girdi | Beklenen |
|---|---|---|---|
| 1 | Normal kesme | `10–20 sn`, `clip.mp4` | Dosya oluşur, açılır ve yaklaşık 10 saniyedir. |
| 2 | Uzantısız ad | `olay_kesiti` | Dosya `olay_kesiti.mp4` olarak oluşturulur. |
| 3 | Kaynak yok | Olmayan video yolu | `kaynak video bulunamadı` hatası döner. |
| 4 | Negatif başlangıç | `start=-5`, `end=10` | `start_sec negatif olamaz` hatası döner. |
| 5 | Eşit zaman | `start=10`, `end=10` | `end_sec, start_sec değerinden büyük olmalıdır` hatası döner. |
| 6 | Ters zaman | `start=20`, `end=10` | Aynı zaman doğrulama hatası döner. |
| 7 | Boş çıktı adı | `output_filename=""` | `output_filename boş olamaz` hatası döner. |
| 8 | Klasör yok | `/olmayan/klasor/clip.mp4` | `çıktı klasörü bulunamadı` hatası döner. |
| 9 | Bozuk kaynak | Bozuk `.mp4` dosyası | `FFmpeg hata verdi` mesajı döner; başarı denmez. |
| 10 | Yazma izni yok | Korunan klasöre çıktı | Hata döner; başarı denmez. |
| 11 | Geçersiz çıktı | Boş/okunamayan çıktı | `dosya boş` veya `geçerli video olarak okunamadı` hatası döner. |
| 12 | Aynı dosya adı | Var olan `clip.mp4` | `-y` nedeniyle dosyanın üzerine yazılır ve yeni dosya açılabilir. |
| 13 | Agent seçimi | “Anomaliyi `anomali.mp4` olarak kaydet” | Agent `save_video_segment` tool'unu çağırır. |

## Başarı kriterleri

- FFmpeg hata vermemeli.
- Çıktı dosyası oluşmalı ve boş olmamalı.
- Video açılmalı ve en az bir kare okunabilmeli.
- Sistem yalnızca bu kontrollerden sonra `Başarılı!` demeli.
