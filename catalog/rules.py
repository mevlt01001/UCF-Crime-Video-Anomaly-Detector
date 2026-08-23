from __future__ import annotations

import re

from .models import Camera
from .repository import CameraRepository

_ID_RE = re.compile(r"\bcam[_-]?0*(\d+)\b", re.IGNORECASE)
_KAMERA_RE = re.compile(r"\bkamera\s*0*(\d+)\b", re.IGNORECASE)
_BLOK_RE = re.compile(r"\b([a-zçğıöşü])\s*blok\b", re.IGNORECASE)
_KAT_NUM_RE = re.compile(r"\b(\d+)\s*\.\s*kat\b", re.IGNORECASE)
_ZEMIN_RE = re.compile(r"\bzemin\s*kat\b", re.IGNORECASE)
_BODRUM_RE = re.compile(r"\bbodrum\b", re.IGNORECASE)


def narrow(query: str, cameras: list[Camera]) -> tuple[list[Camera], bool]:
    """Soruya göre havuzu kes.

    Dönüş: (adaylar, kesin_id_mi).
    Filtre yoksa havuz olduğu gibi kalır (boş ≠ filtresiz).
    Kullanıcının söylediği yer/id hiç uymuyorsa boş liste döner.
    """
    exact = _find_explicit_id(query, cameras)
    if exact is not None:
        return [exact], True

    pool = list(cameras)
    applied_hard_filter = False

    blok = _find_blok(query)
    if blok:
        applied_hard_filter = True
        pool = [c for c in pool if blok in _fold(c.location)]

    kat = _find_kat(query)
    if kat:
        applied_hard_filter = True
        pool = [c for c in pool if kat in _fold(c.location)]

    named = _filter_name_alias(query, pool)
    if named:
        pool = named
    elif applied_hard_filter and not pool:
        return [], False

    return pool, False


def leftover_intent(query: str, pool: list[Camera]) -> str:
    """id / blok / kat / ortak alias çıkınca geriye kalan metin.

    Boşsa soru sadece yer veya ortak isimdir; model tahmin etmemeli.
    """
    text = _fold(query)
    text = _ID_RE.sub(" ", text)
    text = _KAMERA_RE.sub(" ", text)
    blok = _find_blok(query)
    if blok:
        text = text.replace(blok, " ")
    kat = _find_kat(query)
    if kat:
        text = text.replace(kat, " ")
    shared: set[str] | None = None
    for camera in pool:
        phrases = {_fold(camera.name), _fold(camera.id), *(_fold(a) for a in camera.aliases)}
        shared = phrases if shared is None else shared & phrases
    for phrase in sorted(shared or [], key=len, reverse=True):
        if len(phrase) >= 3:
            text = text.replace(phrase, " ")
    return " ".join(text.split())


def validate_id(camera_id: str | None, repo: CameraRepository) -> Camera | None:
    if not camera_id:
        return None
    return repo.get_by_id(camera_id)


def _find_explicit_id(query: str, cameras: list[Camera]) -> Camera | None:
    match = _ID_RE.search(query) or _KAMERA_RE.search(query)
    if not match:
        return None
    number = match.group(1)
    for camera in cameras:
        cam_match = _ID_RE.search(camera.id)
        if cam_match and cam_match.group(1) == number:
            return camera
        if camera.id.casefold() == f"cam_{number}".casefold():
            return camera
    return None


def _find_blok(query: str) -> str | None:
    match = _BLOK_RE.search(_fold(query))
    if not match:
        return None
    letter = match.group(1)
    return f"{letter} blok"


def _find_kat(query: str) -> str | None:
    folded = _fold(query)
    if _ZEMIN_RE.search(folded):
        return "zemin kat"
    if _BODRUM_RE.search(folded):
        return "bodrum"
    match = _KAT_NUM_RE.search(folded)
    if match:
        return f"{match.group(1)}. kat"
    return None


def _filter_name_alias(query: str, cameras: list[Camera]) -> list[Camera]:
    folded_query = _fold(query)
    scored: list[tuple[int, Camera]] = []
    for camera in cameras:
        phrases = [camera.name, camera.id, *camera.aliases]
        best = max((_phrase_len(folded_query, p) for p in phrases), default=0)
        if best:
            scored.append((best, camera))
    if not scored:
        return []
    best_len = max(score for score, _ in scored)
    return [camera for score, camera in scored if score == best_len]


def _phrase_len(folded_query: str, phrase: str) -> int:
    token = _fold(phrase)
    if len(token) < 3 or token not in folded_query:
        return 0
    return len(token)


def _fold(text: str) -> str:
    return (text or "").casefold().strip()
