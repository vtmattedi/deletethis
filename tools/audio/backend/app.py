"""HTTP API, live WebSocket, and the static browser UI.

    python tools/audio/backend/app.py --target 192.168.1.50:3333

The browser displays and configures. It never classifies, and it
never talks to the ESP32.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow `python backend/app.py` as well as `-m backend.app`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from backend import config as config_module
    from backend.models import ConfigPatch
    from backend.stream import StreamService
else:
    from . import config as config_module
    from .models import ConfigPatch
    from .stream import StreamService

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

AppConfig = config_module.AppConfig


def create_app(settings: AppConfig) -> FastAPI:
    service = StreamService(settings)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        service.start()

        print(f"Web: http://{settings.http_host}:{settings.http_port}")

        yield

        service.stop()
        service.join(timeout=3.0)

    app = FastAPI(title="INMP441 audio backend", lifespan=lifespan)

    app.state.settings = settings
    app.state.service = service

    # ------------------------------------------------------ status

    @app.get("/api/status")
    async def get_status() -> dict:
        payload = service.status()
        payload["config"] = settings.to_api()

        return payload

    # ------------------------------------------------------ config

    @app.get("/api/config")
    async def get_config() -> dict:
        return settings.to_api()

    @app.patch("/api/config")
    async def patch_config(patch: ConfigPatch) -> dict:
        changes = patch.changes()

        if not changes:
            return settings.to_api()

        settings.classifier = settings.classifier.patched(changes)
        settings.events = settings.events.patched(changes)

        # Applied at once, without touching the TCP connection: the
        # rules changed, not the audio.
        if service.classifier is not None:
            service.classifier.apply_config(settings.classifier)

        if service.events is not None:
            service.events.apply_config(settings.events)

        print(f"config: {changes}")

        return settings.to_api()

    # ------------------------------------------------------ events

    @app.get("/api/events")
    async def get_events() -> dict:
        if service.events is None:
            return {"events": []}

        return {"events": service.events.recent()}

    @app.get("/api/events/{identifier}/audio")
    async def get_event_audio(identifier: str) -> FileResponse:
        # The identifier becomes a path, so it must not be able to
        # leave the events directory.
        path = (settings.events_dir / f"{identifier}.wav").resolve()

        if (
            settings.events_dir.resolve() not in path.parents
            or not path.is_file()
        ):
            raise HTTPException(status_code=404, detail="no such event")

        return FileResponse(path, media_type="audio/wav")

    # -------------------------------------------------------- live

    @app.websocket("/ws/live")
    async def live(socket: WebSocket) -> None:
        await socket.accept()

        interval = 1.0 / max(settings.live_hz, 0.5)

        try:
            while True:
                payload = service.status()
                payload["config"] = settings.to_api()

                await socket.send_json(payload)
                await asyncio.sleep(interval)
        except (WebSocketDisconnect, ConnectionError):
            return
        except RuntimeError:
            # Socket closed under us while sending.
            return

    # ---------------------------------------------------------- UI

    if config_module.WEB.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=config_module.WEB, html=True),
            name="web",
        )
    else:

        @app.get("/")
        async def missing_ui() -> JSONResponse:
            return JSONResponse(
                {"detail": f"no UI at {config_module.WEB}"},
                status_code=404,
            )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local backend for the INMP441 detector.",
    )

    parser.add_argument(
        "--target",
        default=config_module.DEFAULT_TARGET,
        help=(
            "ESP32 audio stream, host[:port] or a serial port "
            f"(default: {config_module.DEFAULT_TARGET})"
        ),
    )

    parser.add_argument(
        "--host",
        default=config_module.DEFAULT_HTTP_HOST,
        help=(
            "Address to serve on. The default is loopback only; "
            "use 0.0.0.0 to reach it from another machine"
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=config_module.DEFAULT_HTTP_PORT,
        help=f"(default: {config_module.DEFAULT_HTTP_PORT})",
    )

    parser.add_argument(
        "--live-hz",
        type=float,
        default=config_module.DEFAULT_LIVE_HZ,
        help=(
            "Snapshots per second to the browser; the classifier "
            f"always runs at full rate (default: "
            f"{config_module.DEFAULT_LIVE_HZ:g})"
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = AppConfig(
        target=args.target,
        http_host=args.host,
        http_port=args.port,
        live_hz=args.live_hz,
    )

    print("Backend starting")
    print(f"ESP32: {settings.target}")

    uvicorn.run(
        create_app(settings),
        host=settings.http_host,
        port=settings.http_port,
        log_level="warning",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
