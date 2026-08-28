# Görev başlangıcı video metadata — manuel testler

Uygulamayı yeniden başlatın. Süresi bilinen iki video (örn. 17 ve 30 saniye)
ve boş/bozuk bir dosya hazırlayın. Kod otomatik metadata hazırlığını **planner'ın
LLM çağrısından önce** yapar; yeni node veya `get_video_info` tool çağrısı değildir.

| Senaryo | Beklenen sonuç |
| --- | --- |
| Video yükle, mesaj gönderme | Yükleme mevcut şekilde çalışır; agent/analiz başlamaz. |
| Web sohbetinde videosuz “Merhaba” | “Önce video yükleyin.” uyarısı; mesaj geçmişe eklenmez, iş/LLM çağrısı başlamaz. |
| 17 sn videoda “Süresi ve FPS'i nedir?” | Teknik bilgiyle uyumlu cevap; ayrı tool çağrısı şart değildir. |
| 17 sn videoda “40. saniyede ne oluyor?” | Süre dışı olduğu belirtilir; görülmemiş olay uydurulmaz. Yanlış tool çağrısı yapılırsa tool reddeder. |
| “10–25 saniyeyi incele” | Planner süreyi bilir; çağrı 25 ile yapılırsa tool 17'ye kırpar ve uyarır. |
| Aynı dosyada ikinci soru | Güncel metadata önbellekten gelir; video okuyucusu gereksiz yeniden oluşturulmaz. |
| Yeni sohbet + 30 sn video | Yeni videonun süresi kullanılır; 17 sn eski metadata taşınmaz. |
| Bozuk/silinmiş video ile “Merhaba” | Metadata `unavailable`, süre bilinmiyor; ilgisiz sohbet devam eder. |
| Bozuk video ile analiz/rapor | Başarılı analiz veya düşük risk uydurulmaz; mevcut tool/rapor doğrulamaları hatayı yönetir. |
| Aynı yoldaki dosyayı değiştir, yeni görev başlat | Dosya imzası değiştiğinden metadata yeniden okunur. |
| İki farklı oturum/video | Her görevin kendi metadata'sı vardır; bağlamlar karışmaz. |
| Rapor oluştur | Aynı node sırası, araç ayrımı ve rapor şeması; metadata olay/VLM/arşiv kanıtı yerine geçmez. |
| Reviewer yeniden planner'a gönderir | Metadata yeniden hazırlanır; değişmeyen dosyada önbellek kullanılır. |
| Web UI, lab.py, doğrudan graph | Üçü de aynı planner hazırlığını kullanır; UI'a özel zorunlu adım yok. |

## Geliştirici kontrolü

- `planner_node` dönüşündeki `video_context`: `ready`, `no_video` veya `unavailable`.
  Başarılıysa süre/FPS/kare sayısı/boyutlar; hatada `metadata=null` (sıfır değil).
- Planner, executor ve reviewer model girişinde aynı teknik snapshot bulunmalı.
- `conversation_messages` ve `messages` içine hazırlık mesajı eklenmemeli;
  ek tool turu, eylem kaydı veya model çağrısı oluşmamalı.
- Arayüz trace'inde ayrı metadata node'u görünmemesi normaldir; otomatik hazırlık
  planner içindedir. Agent isterse mevcut `get_video_info` tool'unu yine çağırabilir.
- Önbellek testi: `_get_video_metadata` içindeki `VideoReader` oluşturma sayısını
  gözleyin; aynı dosyada tekrar görevde artmamalı, dosya değişince artmalı.

Metadata görev başlangıcı snapshot'ıdır; işlem sırasında dosya değişirse güvenlik
otoritesi tool'ların güncel dosya/zaman kontrolleridir. Süre hesabı mevcut
`frame_count / avg_fps` yöntemini korur; VFR süre hassasiyeti bu değişikliğin konusu değildir.

Web UI video zorunluluğunu girişte uygular; doğrudan graph'ın videosuz teknik
bağlamı desteklemesi bu kısıtı kaldırmaz. Video yükledikten sonra aynı mesajı
gönderin: normal işlem başlamalıdır. “Yeni sohbet” sonrası videosuz gönderimde
uyarı yeniden görünmelidir. Enter ve gönder butonu aynı kontrolü kullanır.
