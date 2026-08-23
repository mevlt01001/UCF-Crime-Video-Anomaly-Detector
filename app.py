from __future__ import annotations

from time import perf_counter

from agent.picker import make_picker
from catalog.repository import CameraRepository
from tools import TOOLS, run_select_camera


def main() -> None:
    repo = CameraRepository()
    picker = make_picker()

    print("Kamera seçimi v1 (JSON). Çıkmak için q.")
    print(f"Katalog: {repo.path} ({len(repo.get_all())} kamera)")
    print(f"Picker: {type(picker).__name__}")
    print("Toollar:")
    for item in TOOLS:
        print(f"  - {item.name}")

    while True:
        try:
            query = input("\nSoru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"q", "quit", "exit"}:
            break
        if not query:
            continue

        result, elapsed = _run_timed(query, repo, picker)
        _print_result(result, elapsed)

        while result.status == "ambiguous" and result.ask_user:
            try:
                clarification = input("Yanıt: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not clarification:
                break
            query = f"{query} {clarification}"
            result, elapsed = _run_timed(query, repo, picker)
            _print_result(result, elapsed)


def _run_timed(query, repo, picker):
    started = perf_counter()
    result = run_select_camera(query, repo=repo, picker=picker)
    return result, perf_counter() - started


def _print_result(result, elapsed: float) -> None:
    print(f"status: {result.status}")
    print(f"süre: {elapsed:.3f}s ({elapsed * 1000:.0f} ms)")
    if result.ask_user:
        print(f"soru: {result.ask_user}")
    if not result.data:
        if result.status == "not_found":
            print("Bu alanı gören kamera yok.")
        return
    for item in result.data:
        print(
            f"  {item['id']}  {item['name']}"
            f"  [{item.get('reason', '')}]  {item.get('video_path', '')}"
        )


if __name__ == "__main__":
    main()
