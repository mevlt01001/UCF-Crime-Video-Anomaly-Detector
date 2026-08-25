import os
from pathlib import Path
from typing import Optional

_LOADED = False


def load_env() -> None:
    global _LOADED
    if _LOADED:
        return

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    _LOADED = True


def env_get(name: str, default: Optional[str] = None) -> str:
    load_env()
    value = os.getenv(name, default if default is not None else "")
    return (value or "").strip()


def env_require(name: str) -> str:
    value = env_get(name)
    if not value:
        raise RuntimeError(f"{name} eksik. .env dosyasını kontrol et.")
    return value
