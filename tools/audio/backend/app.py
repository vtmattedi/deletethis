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
import time
from pathlib import Path

if __package__ in (None, ""):
    # Allow `python backend/app.py` as well as `-m backend.app`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from backend import config as config_module
    from backend.config import ClassifierConfig
    from backend.events import event_timeline
    from backend.history import to_epoch
    from backend.models import (
        ConfigPatch,
        EventBulkReviewRequest,
        EventDeleteRequest,
        ReviewPatch,
    )
    from backend.stream import StreamService
else:
    from . import config as config_module
    from .config import ClassifierConfig
    from .events import event_timeline
    from .history import to_epoch
    from .models import (
        ConfigPatch,
        EventBulkReviewRequest,
        EventDeleteRequest,
        ReviewPatch,
    )
    from .stream import StreamService

import uvicorn
from classifier import VERSIONS
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

AppConfig = config_module.AppConfig


def _is_benign_proactor_reset(context: dict) -> bool:
    """Recognize Windows' noisy reset while closing a client socket."""
    error = context.get("exception")
    callback = " ".join(
        str(context.get(key, "")) for key in ("message", "handle")
    )
    return (
        sys.platform == "win32"
        and isinstance(error, ConnectionResetError)
        and getattr(error, "winerror", None) == 10054
        and "_ProactorBasePipeTransport._call_connection_lost" in callback
    )


def load_saved_config(settings: AppConfig) -> dict:
    """Validate and apply persisted classifier/event settings.

    A saved file belongs to the classifier version that wrote it. One
    with no version key predates versions and is v1. Applying it to a
    different version would mean guessing what its numbers meant under
    rules they were never tuned for, so it is refused instead.
    """
    payload = dict(
        config_module.load_runtime_config(settings.config_path)
    )
    saved_version = payload.pop("classifierVersion", "v1")
    running = settings.classifier.version

    if saved_version != running:
        raise ValueError(
            f"saved config is for classifier {saved_version}, but "
            f"this run is {running}; not reinterpreting it"
        )

    changes = ConfigPatch.model_validate(payload).changes()
    _require_version_fields(running, changes)
    settings.classifier = settings.classifier.patched(changes)
    settings.events = settings.events.patched(changes)
    return changes


def _require_version_fields(version: str, changes: dict) -> None:
    """The v2-only settings mean nothing to a v1 classifier."""
    foreign = [
        name for name in config_module.V2_API_FIELDS
        if name in changes
    ]
    if version == "v1" and foreign:
        raise ValueError(
            f"{', '.join(foreign)} only apply to classifier v2, "
            f"but this run is {version}"
        )


def create_app(settings: AppConfig) -> FastAPI:
    service = StreamService(settings)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop = asyncio.get_running_loop()
        previous_exception_handler = loop.get_exception_handler()

        def handle_asyncio_exception(loop, context: dict) -> None:
            if _is_benign_proactor_reset(context):
                return
            if previous_exception_handler is not None:
                previous_exception_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(handle_asyncio_exception)
        service.start()

        print(f"Web: http://{settings.http_host}:{settings.http_port}")

        try:
            yield
        finally:
            loop.set_exception_handler(previous_exception_handler)
            service.stop()
            service.join(timeout=3.0)
            service.close()

    app = FastAPI(title="INMP441 audio backend", lifespan=lifespan)

    app.state.settings = settings
    app.state.service = service

    # ------------------------------------------------------ status

    @app.get("/api/status")
    async def get_status() -> dict:
        payload = service.status()
        payload["classifierVersion"] = settings.classifier.version
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

        try:
            _require_version_fields(
                settings.classifier.version, changes
            )
        except ValueError as error:
            raise HTTPException(
                status_code=422, detail=str(error)
            ) from error

        classifier = settings.classifier.patched(changes)
        events = settings.events.patched(changes)
        saved = classifier.to_api()
        saved.update(events.to_api())
        try:
            config_module.write_runtime_config(settings.config_path, saved)
        except OSError as error:
            raise HTTPException(
                status_code=500,
                detail=f"could not save configuration: {error}",
            ) from error

        settings.classifier = classifier
        settings.events = events

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
    async def get_events(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, alias="pageSize", ge=1, le=100),
        search: str = Query(default="", max_length=200),
    ) -> dict:
        if service.events is None:
            return {
                "events": [], "page": page, "pageSize": page_size,
                "total": 0, "pages": 0,
            }

        return service.events.list_events(page, page_size, search)

    @app.post("/api/events/manual", status_code=202)
    async def record_manual_event() -> dict:
        identifier = service.record_manual()
        if identifier is None:
            raise HTTPException(
                status_code=409,
                detail="audio stream has not produced a state yet",
            )
        return {"id": identifier, "status": "recording"}

    @app.delete("/api/events")
    async def delete_events(request: EventDeleteRequest) -> dict:
        deleted = []
        not_found = []
        # Preserve request order but never process one event twice.
        for identifier in dict.fromkeys(request.ids):
            if service.events.delete(identifier):
                deleted.append(identifier)
            else:
                not_found.append(identifier)
        return {"deleted": deleted, "notFound": not_found}

    @app.patch("/api/events/reviews")
    async def patch_event_reviews(
        request: EventBulkReviewRequest,
    ) -> dict:
        if service.events is None:
            raise HTTPException(status_code=404, detail="no event store")

        identifiers = list(dict.fromkeys(request.ids))
        reviews = {}
        for identifier in identifiers:
            metadata = service.events.get(identifier)
            if metadata is None:
                raise HTTPException(
                    status_code=404,
                    detail={"message": "event selection is stale", "ids": [identifier]},
                )

            if request.classificationCorrect:
                actual_from = metadata.get("from") or "UNKNOWN"
                actual_to = metadata.get("to") or "UNKNOWN"
            else:
                actual_from = request.actualFrom
                actual_to = request.actualTo

            reviews[identifier] = {
                "classificationCorrect": request.classificationCorrect,
                "actualFrom": actual_from,
                "actualTo": actual_to,
                "interference": request.interference,
                "notes": request.notes,
            }

        updated, missing = service.events.review_many(reviews)
        if missing:
            raise HTTPException(
                status_code=404,
                detail={"message": "event selection is stale", "ids": missing},
            )
        return {
            "updated": [metadata["id"] for metadata in updated],
            "count": len(updated),
        }

    @app.get("/api/events/{identifier}")
    async def get_event(identifier: str) -> dict:
        metadata = (
            service.events.get(identifier)
            if service.events is not None
            else None
        )

        if metadata is None:
            raise HTTPException(status_code=404, detail="no such event")

        return metadata

    @app.get("/api/events/{identifier}/audio")
    async def get_event_audio(identifier: str) -> FileResponse:
        path = (
            service.events.wav_path(identifier)
            if service.events is not None
            else None
        )

        if path is None:
            raise HTTPException(status_code=404, detail="no such event")

        return FileResponse(path, media_type="audio/wav")

    @app.patch("/api/events/{identifier}/review")
    async def patch_event_review(
        identifier: str, patch: ReviewPatch
    ) -> dict:
        metadata = (
            service.events.review(identifier, patch.model_dump())
            if service.events is not None
            else None
        )
        if metadata is None:
            raise HTTPException(status_code=404, detail="no such event")
        return metadata

    @app.delete(
        "/api/events/{identifier}",
        status_code=204,
        response_class=Response,
    )
    async def delete_event(identifier: str) -> Response:
        deleted = (
            service.events.delete(identifier)
            if service.events is not None
            else False
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="no such event")
        return Response(status_code=204)

    @app.get("/api/events/{identifier}/timeline")
    def get_event_timeline(identifier: str) -> dict:
        # Sync, so Starlette runs it off the event loop: this reads a
        # WAV and runs an FFT over every window in it.
        if service.events is None:
            raise HTTPException(status_code=404, detail="no such event")

        metadata = service.events.get(identifier)
        path = service.events.wav_path(identifier)

        if metadata is None or path is None:
            raise HTTPException(status_code=404, detail="no such event")

        # Rebuilt with the settings that were live when it was
        # recorded, so the picture matches the decision that was made.
        # Events recorded before versions existed carry no
        # classifierVersion and were classified by v1.
        recorded = ClassifierConfig.from_api(
            metadata.get("classifierConfig", {}),
            metadata.get("classifierVersion"),
        )

        timeline = event_timeline(
            path,
            recorded,
            pre_seconds=metadata.get("audio", {}).get(
                "preSeconds", 0.0
            ),
        )

        timeline["id"] = identifier
        timeline["classifierConfig"] = recorded.to_api()

        return timeline

    # ----------------------------------------------------- history

    @app.get("/api/history")
    def get_history(
        request: Request,
        seconds: float = 900.0,
        points: int = 2000,
    ) -> dict:
        """Compact state history. `from`/`to` accept epoch or ISO.

        Declared as query params by hand because `from` is a Python
        keyword and cannot be a parameter name.
        """
        now = time.time()

        params = request.query_params

        end = to_epoch(params.get("to"), now)
        start = to_epoch(
            params.get("from"), end - max(seconds, 1.0)
        )

        if start > end:
            start, end = end, start

        return service.history.query(start, end, points)

    # -------------------------------------------------------- live

    @app.websocket("/ws/live")
    async def live(socket: WebSocket) -> None:
        await socket.accept()

        interval = 1.0 / max(settings.live_hz, 0.5)

        try:
            while True:
                payload = service.status()
                payload["classifierVersion"] = (
                    settings.classifier.version
                )
                payload["config"] = settings.to_api()

                await socket.send_json(payload)
                await asyncio.sleep(interval)
        except (WebSocketDisconnect, ConnectionError):
            return
        except RuntimeError:
            # Socket closed under us while sending.
            return

    @app.websocket("/ws/audio")
    async def live_audio(socket: WebSocket) -> None:
        """Raw live PCM for on-demand browser monitoring.

        A bounded per-client queue drops the oldest packet when the
        browser falls behind. Live listening should become current
        again, never accumulate seconds of delayed audio.
        """
        await socket.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)

        def enqueue(payload: bytes) -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(payload)

        def receive_audio(payload: bytes) -> None:
            try:
                loop.call_soon_threadsafe(enqueue, payload)
            except RuntimeError:
                pass

        service.add_audio_listener(receive_audio)
        last_rate = 0

        try:
            while True:
                connected, sample_rate = service.audio_status()
                if sample_rate and sample_rate != last_rate:
                    await socket.send_json({
                        "type": "format",
                        "sampleRate": sample_rate,
                        "channels": 1,
                        "encoding": "s32le-24bit-right-aligned",
                    })
                    last_rate = sample_rate

                try:
                    payload = await asyncio.wait_for(
                        queue.get(), timeout=1.0
                    )
                except TimeoutError:
                    await socket.send_json({
                        "type": "status",
                        "connected": connected,
                    })
                    continue

                if last_rate:
                    await socket.send_bytes(payload)
        except (WebSocketDisconnect, ConnectionError, RuntimeError):
            return
        finally:
            service.remove_audio_listener(receive_audio)

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
        "--classifier",
        choices=VERSIONS,
        default="v2",
        help=(
            "Classifier version. Each version keeps its events, "
            "history and saved settings in its own results folder "
            "(v1: tools/audio/results, v2: tools/audio/results/v2), "
            "so switching is a restart and never mixes evidence "
            "(default: v2)"
        ),
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

    settings = AppConfig.for_version(
        args.classifier,
        target=args.target,
        http_host=args.host,
        http_port=args.port,
        live_hz=args.live_hz,
    )

    try:
        changes = load_saved_config(settings)
    except (OSError, ValueError) as error:
        print(f"config: ignoring invalid {settings.config_path}: {error}")
    else:
        if changes:
            print(f"config: loaded {settings.config_path}")

    print("Backend starting")
    print(f"ESP32: {settings.target}")
    print(
        f"Classifier: {settings.classifier.version}   "
        f"results: {settings.events_dir.parent}"
    )

    uvicorn.run(
        create_app(settings),
        host=settings.http_host,
        port=settings.http_port,
        log_level="warning",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
