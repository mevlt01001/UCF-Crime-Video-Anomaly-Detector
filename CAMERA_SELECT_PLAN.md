# Kamera Seçim Planı

## Amaç

Kullanıcı sorusundan doğru kamerayı seçmek. Model ID uydurmaz. Video analizi bu tool’un işi değil.

## Veri kaynağı

v1 kaynağı [`config/cameras.json`](config/cameras.json). Postgres / Coolify sonra bağlanır; `CameraRepository` imzası aynı kalır (`get_all`, `get_by_id`, `filter_location`).

Her kamera:

- `id` — değişmez
- `name`
- `aliases`
- `location` — bina / kat
- `view`
- `description`
- `overlaps_with`
- `video_path`

Serbest dil `description` ve `aliases` içinde durur.

## Çalışma sırası

1. Kullanıcı sorar
2. **Kural (önce):** cümlede `id` / bina / kat / resmi ad varsa havuzu daralt
3. **RAG:** kalanlarda `description` / `view` alanına bak, 5–15 aday çıkar
4. **Model:** adaylardan seç veya kullanıcıya sor
5. **Kural (sonra):** dönen `id` listede yoksa reddet

Kural karar vermez; eledir ve kilitler. RAG yakalar. Son söz modeldedir.

Örnek: “B blok, arabaların girdiği yer” → kural B blok’u keser çünkü kullanıcı söylemiştir. RAG sonra sadece o havuzda description arar. Kullanıcı “B blok” demeseydi kural kesmez, RAG tüm listede aday arardı.

## Senaryolar

| Durum | Ne yapılır |
|---|---|
| 10–50 kamera | Listeyi modele ver, RAG yok |
| Net id / yer | Kural yeter, RAG atlanır |
| Serbest dil / çok lokasyon | Kural + RAG + model |
| Tek net aday | Devam |
| Birden fazla / zayıf skor | Kullanıcıya sor, tahmin etme |
| Aday yok | “Bu alanı gören kamera yok” |

## İlk sürüm

Az kamerayla başla: JSON + kural + picker (Gemma varsa, yoksa stub).

Çalıştırma: `python app.py`

Kaç soruda kural kaçırıyor ölç. Oran yüksekse RAG ekle.

Büyüme (polis merkezi / 600 kamera / çok lokasyon) için hybrid zaten planlı; ilk sürümde kurma. DB geçişi: aynı repository arayüzü, JSON yerine Postgres.

## Tool sözleşmesi

LangGraph / LangChain `@tool`. Ev yapımı registry yok.

Yeni tool = fonksiyon + **docstring** + `@tool`. Model seçimi docstring’deki “ne zaman kullan / kullanma”ya göre olur. `TOOLS` listesi grafa `bind_tools` ile verilir.

Router’ın gördüğü argüman sadece `query`. CLI işi `run_select_camera` (repo / picker burada).

## Yapılmayacaklar

- Listeyi 600’lük prompt’a gömme
- Gemma’dan kelime avlatma
- Description varken sadece sabit alias’a güvenme
- Belirsizlikte tahmin etme
- RAG’i kuralın yerine koyma — RAG sadece aday üretir
