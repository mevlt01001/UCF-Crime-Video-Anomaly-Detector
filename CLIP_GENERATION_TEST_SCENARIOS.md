# Klip Üretme Manuel Test Senaryoları

Bu senaryolar, videonun sonundaki tam klibin işlendiğini, eksik kalan framelerin
atıldığını ve boş skor nedeniyle programın crash olmadığını doğrular.

## Test ayarları

Test sırasında `.env` değerleri:

```dotenv
AS_CLIP_SIZE=32
AS_OVERLAP=0
AS_FPS=20
```

Uygulamayı bu değerleri kaydettikten sonra yeniden başlatın. Üretilen videoları
Analyzer veya Agent üzerinden anomali segmenter aracına gönderin.

> Anomali bulunmaması test başarısızlığı değildir. Burada klip sayısı ve işlemin
> kontrollü tamamlanması test edilmektedir.

## Test videolarını hazırlama

```bash
ffmpeg -f lavfi -i testsrc=size=224x224:rate=20 -frames:v 31 -pix_fmt yuv420p /tmp/clip-31.mp4
ffmpeg -f lavfi -i testsrc=size=224x224:rate=20 -frames:v 32 -pix_fmt yuv420p /tmp/clip-32.mp4
ffmpeg -f lavfi -i testsrc=size=224x224:rate=20 -frames:v 64 -pix_fmt yuv420p /tmp/clip-64.mp4
ffmpeg -f lavfi -i testsrc=size=224x224:rate=20 -frames:v 68 -pix_fmt yuv420p /tmp/clip-68.mp4
```

## Senaryolar

### 1. Bir klipten kısa video

- Girdi: `/tmp/clip-31.mp4`
- Beklenen: Video çok kısa olduğuna dair kontrollü hata dönmeli.
- Başarı ölçütü: Uygulama kapanmamalı ve `torch.concat`/`torch.cat` hatası görülmemeli.

### 2. Tam olarak bir klip

- Girdi: `/tmp/clip-32.mp4`
- Beklenen: 1 klip işlenmeli ve analiz tamamlanmalı.
- Başarı ölçütü: Sıfır klip veya boş tensor listesi hatası oluşmamalı.

### 3. Tam olarak iki klip

- Girdi: `/tmp/clip-64.mp4`
- Beklenen: 2 klip işlenmeli.
- Başarı ölçütü: İkinci, yani son tam klip atlanmamalı.

### 4. İki tam klip ve eksik kalan frameler

- Girdi: `/tmp/clip-68.mp4`
- Beklenen: 2 tam klip işlenmeli, sondaki 4 frame atılmalı.
- Başarı ölçütü: Eksik parça klip yapılmamalı ve analiz crash olmadan tamamlanmalı.

### 5. Overlap kullanılan üretim

`.env` değerini geçici olarak değiştirin ve uygulamayı yeniden başlatın:

```dotenv
AS_OVERLAP=16
```

- Girdi: `/tmp/clip-64.mp4`
- Hesap: `stride = 32 - 16 = 16`
- Beklenen: Başlangıçları 0, 16 ve 32 olan toplam 3 klip işlenmeli.
- Başarı ölçütü: Son klip 32–63 aralığını içermeli ve atlanmamalı.

### 6. Art arda farklı videolar

- İşlem: Önce 31, sonra 32, ardından 68 framelik videoyu aynı uygulama oturumunda çalıştırın.
- Beklenen: İlk video kontrollü hata vermeli; diğer iki analiz çalışmaya devam etmeli.
- Başarı ölçütü: Kısa video hatası sonraki istekleri veya sohbet oturumunu bozmamalı.

## Başarısızlık belirtileri

Aşağıdakilerden biri görülürse test başarısızdır:

- `torch.cat(): expected a non-empty list of Tensors`
- Tam 32 framelik videoda sıfır klip üretilmesi
- Tam 64 framelik videoda son klibin atlanması
- Eksik kalan 4 framenin tam klip gibi işlenmesi
- Kontrollü hatadan sonra uygulamanın yeni istek kabul etmemesi
