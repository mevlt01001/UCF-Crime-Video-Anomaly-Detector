from __future__ import annotations

from langchain_core.tools import tool

from agent.picker import Picker, make_picker
from catalog.repository import CameraRepository
from catalog.rules import leftover_intent, narrow, validate_id

from .base import ToolResult


def run_select_camera(
    query: str,
    repo: CameraRepository | None = None,
    picker: Picker | None = None,
) -> ToolResult:
    repo = repo or CameraRepository()
    picker = picker or make_picker()

    cameras = repo.get_all()
    if not cameras:
        return ToolResult(status="not_found", data=[], ask_user=None)

    pool, exact_id = narrow(query, cameras)

    if not pool:
        return ToolResult(status="not_found", data=[], ask_user=None)

    if exact_id and len(pool) == 1:
        camera = validate_id(pool[0].id, repo)
        if camera is None:
            return ToolResult(status="not_found", data=[])
        return ToolResult(
            status="matched",
            data=[camera.to_result(reason="kural: açık id")],
        )

    if len(pool) > 1 and not leftover_intent(query, pool):
        options = ", ".join(f"{c.id} ({c.name})" for c in pool)
        return ToolResult(
            status="ambiguous",
            data=[c.to_result(reason="aday") for c in pool],
            ask_user=f"Birden fazla kamera uyuyor. Hangisi: {options}?",
        )

    pick = picker.pick(query, pool)
    whole_catalog = len(pool) == len(cameras)

    if pick.not_found or (whole_catalog and pick.ask_user):
        return ToolResult(status="not_found", data=[], ask_user=None)

    if pick.ask_user:
        return ToolResult(
            status="ambiguous",
            data=[c.to_result(reason="aday") for c in pool],
            ask_user=pick.ask_user,
        )

    allowed = {c.id for c in pool}
    if not pick.camera_id or pick.camera_id not in allowed:
        return ToolResult(status="not_found", data=[])

    camera = validate_id(pick.camera_id, repo)
    if camera is None:
        return ToolResult(status="not_found", data=[])

    reason = "kural: tek aday" if len(pool) == 1 else "model seçimi"
    return ToolResult(status="matched", data=[camera.to_result(reason=reason)])


@tool
def select_camera(query: str) -> dict:
    """Kullanıcı sorusuna göre katalogdan doğru gözetim kamerasını seçer.

    Ne zaman kullan:
    - Kullanıcı bir yer, olay, giriş-çıkış veya kamera soruyorsa
    - Video / VLM tool'undan önce hangi kameranın kaydı gerektiği belli değilse

    Ne zaman kullanma:
    - Kamera id'si zaten seçilmişse
    - Video içinde ne olduğunu anlatmak veya anomali aramak için
      (bunlar ayrı toollardır)

    Args:
        query: Kullanıcının doğal dil sorusu. Kamera id, bina, kat veya
            serbest yer tarifi içerebilir.

    Returns:
        status, data, ask_user:
        - matched: data içinde tek kamera (id, name, video_path)
        - ambiguous: ask_user dolu; kullanıcıya sor, tahmin etme
        - not_found: bu alanı gören kamera yok
    """
    result = run_select_camera(query)
    return {
        "status": result.status,
        "data": result.data,
        "ask_user": result.ask_user,
    }


TOOLS = [select_camera]
