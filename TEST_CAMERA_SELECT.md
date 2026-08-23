# Kamera seçimi testleri

2026-08-23 · Gemma 4 E2B Q4 · n_ctx 8192 · M5 24 GB

Kural ~0 ms. Model ~6.5–7 s (30 kamera).

## Ne tuttu, ne kaçtı

| Soru | Sonuç | Yol |
|---|---|---|
| `cam_03` / `garaj` / `b blok` / `rampa` | doğru veya sordu | kural |
| `arabaların girdiği yer` | cam_03 | model |
| `yemek kasanın olduğu yer` | cam_09 | model |
| `çorum nerdedir` / `mavi montlu adam` | yok | model |
| `bilgisayar kabinleri` (cam_14 olmalı) | yok | model kaçırdı |
| `açık alanda park eden araçlar` (cam_22/23) | yok | model kaçırdı |
| `dışarda açık` | bahçe (yanlış) | model |

Eski hatalar (düzeldi): `b blok` tahmin, başkent → lobi, 30 kamerada context 2048 taşması.

## RAG

Hepsini çözmez. Aday sayısını düşürür (5–15), 100+ kamerada süre ve isabet biraz artabilir. Embedder Türkçe zayıfsa yine kaçırır. Kural durur.

Önce 100 kamerada aynı soruları tekrarla. Kaçırma çoksa RAG ekle.

## 100 kamera senaryoları

Açılışta `100 kamera` olsun. Her satırda status + süre yaz.

**Kural (~0 ms)**

| Yaz | Beklenen |
|---|---|
| `cam_03` | cam_03 |
| `kamera 18` | cam_18 |
| `garaj` | sor: 03 / 04 |
| `b blok` | sor (birkaç aday) |
| `a blok zemin kat` | sor (01, 02, 24…) |
| `bodrum` | sor (08, 14) |
| `fitness` | cam_26 |

**Model (süre not et)**

| Yaz | Beklenen |
|---|---|
| `arabaların girdiği yer` | cam_03 |
| `yemek kasanın olduğu yer` | cam_09 |
| `kitap okunan salon` | cam_29 |
| `personelin kart bastığı kapı` | cam_21 |
| `bilgisayar kabinlerinin bulunduğu oda` | cam_14 |
| `dışarda açık alanda park edilen araçlar` | cam_22 veya 23 |

**Olmamalı**

| Yaz | Beklenen |
|---|---|
| `çorum nerdedir` | yok |
| `mavi montlu adam` | yok |
| `berberde biri var mıydı` | yok |

