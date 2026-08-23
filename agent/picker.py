from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from catalog.models import Camera


@dataclass
class PickResult:
    camera_id: str | None = None
    ask_user: str | None = None
    not_found: bool = False


class Picker(Protocol):
    def pick(self, query: str, candidates: list[Camera]) -> PickResult:
        ...


class StubPicker:
    """Tek aday varsa seçer; birden fazla varsa kullanıcıya sorar. Tahmin etmez."""

    def pick(self, query: str, candidates: list[Camera]) -> PickResult:
        if len(candidates) == 1:
            return PickResult(camera_id=candidates[0].id)
        if not candidates:
            return PickResult()
        options = ", ".join(f"{c.id} ({c.name})" for c in candidates)
        return PickResult(ask_user=f"Birden fazla kamera uyuyor. Hangisi: {options}?")


class GemmaPicker:
    """Aday listesinden id seçer veya belirsizse sorar. Listede olmayan id üretilemez."""

    def __init__(self, model_path: str, n_gpu_layers: int = 0, n_ctx: int = 8192):
        from llama_cpp import Llama

        self._allowed: set[str] = set()
        self.n_ctx = n_ctx
        print(f"Gemma yükleniyor (n_ctx={n_ctx})...")
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=8,
            n_batch=256,
            verbose=False,
        )

    def pick(self, query: str, candidates: list[Camera]) -> PickResult:
        if len(candidates) == 1:
            return PickResult(camera_id=candidates[0].id)
        if not candidates:
            return PickResult()

        self._allowed = {c.id for c in candidates}
        prompt = _build_prompt(query, candidates)
        try:
            response = self.model.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Görevin yalnızca gözetim kamerası seçmek. "
                            "Soru bir yer/kamera ile ilgili değilse veya listede o yer yoksa "
                            "kamera seçme, not_found de. "
                            "Listede olmayan id uydurma. Sadece JSON döndür."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=128,
                temperature=0.1,
            )
        except ValueError as exc:
            print(f"Gemma context taştı ({exc}) → not_found")
            return PickResult(not_found=True)
        text = response["choices"][0]["message"]["content"].strip()
        return _parse_pick(text, self._allowed, candidates)


def make_picker() -> Picker:
    model_path = os.environ.get("GEMMA_MODEL_PATH", "").strip()
    if not model_path:
        print("GEMMA_MODEL_PATH yok, StubPicker kullanılıyor.")
        return StubPicker()
    path = Path(model_path)
    if not path.is_file():
        print(f"Gemma dosyası yok: {model_path} → StubPicker")
        return StubPicker()
    if path.stat().st_size < 1024:
        print(f"Gemma dosyası boş/eksik ({path.stat().st_size} byte): {model_path} → StubPicker")
        return StubPicker()
    try:
        layers = int(os.environ.get("GEMMA_N_GPU_LAYERS", "0"))
        n_ctx = int(os.environ.get("GEMMA_N_CTX", "8192"))
        return GemmaPicker(model_path, n_gpu_layers=layers, n_ctx=n_ctx)
    except Exception as exc:
        print(f"Gemma yüklenemedi ({exc}) → StubPicker")
        return StubPicker()


def _build_prompt(query: str, candidates: list[Camera]) -> str:
    lines = []
    for camera in candidates:
        aliases = ", ".join(camera.aliases)
        lines.append(
            f"- {camera.id} | {camera.name} | {camera.location} | "
            f"{aliases} | {camera.description}"
        )
    catalog = "\n".join(lines)
    return (
        f"Kullanıcı sorusu: {query}\n\n"
        f"Aday kameralar:\n{catalog}\n\n"
        "Soru kamera/yer seçimi değilse veya listede yoksa: {\"not_found\": true}\n"
        "Tek kamera netse: {\"id\": \"cam_03\"}\n"
        "Listede birkaç yer uyuyorsa: {\"ask_user\": \"Hangi kamera: garaj girişi mi çıkış mı?\"}\n"
    )


def _parse_pick(text: str, allowed: set[str], candidates: list[Camera]) -> PickResult:
    payload = _extract_json(text)
    if payload is None:
        return StubPicker().pick("", candidates)

    if payload.get("not_found") in (True, "true", "True", 1):
        return PickResult(not_found=True)

    ask = payload.get("ask_user")
    if isinstance(ask, str) and ask.strip():
        return PickResult(ask_user=ask.strip())

    camera_id = payload.get("id")
    if isinstance(camera_id, str) and camera_id in allowed:
        return PickResult(camera_id=camera_id)
    return PickResult(not_found=True)


def _extract_json(text: str) -> dict | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
