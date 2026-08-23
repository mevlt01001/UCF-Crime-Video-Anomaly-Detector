# UCF-Crime Video Anomaly Detector

Gözetim videosunda anomali tespiti için araştırma prototipi. Üst katman: kullanıcı sorusundan **doğru kamerayı seçmek**. Video analizi ve VLM sonra bağlanır.

Şu an çalışan parça **kamera seçimi v1** (JSON katalog + kural + picker).

## Ne yapar (v1)

```
soru → kural (id / bina / kat / alias) → adaylar → picker → id kilidi
```

- `matched` — tek kamera
- `ambiguous` — kullanıcıya sor, tahmin yok
- `not_found` — bu alanı gören kamera yok

Katalog: [`config/cameras.json`](config/cameras.json). Postgres sonra; `CameraRepository` imzası aynı kalır.

Tool’lar LangChain `@tool` + docstring. Router (LangGraph) gelince `TOOLS` bağlanır.

## Gereksinimler

- Python 3.9+ (`python3`)
- Mac’te `python` komutu olmayabilir

**Gerçek anlama testi** için ayrıca:

| Ne | Boyut | Ne işe yarar |
|---|---|---|
| Gemma 4 E2B Instruct GGUF (Q4 / Q4_K_M) | ~2.5–3.1 GB disk, ~4–7 GB RAM | Adaylar arasından seçer |
| `llama-cpp-python` | küçük | Modeli yükler |

Qwen-VL ve S3D **şimdi gerekmez**.

M5 / 16 GB+ RAM’de Gemma E2B Q4 sorunsuz çalışır.

## Kurulum

```bash
python3 -m pip install --user -r requirements.txt
python3 -m pip install --user llama-cpp-python huggingface_hub
```

Gemma GGUF (~3.1 GB), bir kez:

```bash
python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id='unsloth/gemma-4-E2B-it-GGUF',
    filename='gemma-4-E2B-it-Q4_K_M.gguf',
    local_dir='.',
))
"
```

Windows’ta `python3` yoksa `python` kullan. GGUF repo’da yok; herkes kendi indirmeli.

## Çalıştırma

Kural + stub (model yok):

```bash
python3 app.py
```

Gemma ile (indirdiğin dosyanın tam yolu):

```bash
GEMMA_MODEL_PATH=/tam/yol/gemma-4-E2B-it-Q4_K_M.gguf python3 app.py
```

Apple Silicon (isteğe bağlı):

```bash
GEMMA_MODEL_PATH=/tam/yol/gemma-4-E2B-it-Q4_K_M.gguf GEMMA_N_GPU_LAYERS=99 python3 app.py
```

Açılışta `Picker: GemmaPicker` görmelisin. `StubPicker` ise yol yanlış veya `llama-cpp-python` yok.

Çıkış: `q`

## Denenecek sorular

| Soru | Beklenen |
|---|---|
| `cam_03` | matched, Garaj girişi |
| `garaj` | ambiguous, hangisi diye sorar |
| `garaj içi` | matched, cam_04 |
| `B blok, arabaların girdiği yer` | Gemma: cam_03; stub: sorar |
| `Z blok` | not_found |

## Klasörler

```
app.py                 CLI
config/cameras.json    katalog
catalog/               repository + kurallar
tools/select_camera.py LangChain tool
agent/picker.py        GemmaPicker / StubPicker
utils/                 video / VLM (v1 bağlı değil)
```

Tasarım notu: [`CAMERA_SELECT_PLAN.md`](CAMERA_SELECT_PLAN.md).

## Sonra

- Coolify Postgres (`cameras` DB, `pgvector/pgvector:pg17`)
- LangGraph router
- Video analyzer + VLM tool’ları
