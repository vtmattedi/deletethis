import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "audio"
sys.path.insert(0, str(TOOLS))

from backend.app import create_app, load_saved_config  # noqa: E402
from backend.config import (  # noqa: E402
    AppConfig,
    load_runtime_config,
    write_runtime_config,
)
from backend.models import ConfigPatch  # noqa: E402


class PersistedConfigTests(unittest.TestCase):
    def test_patch_endpoint_persists_frontend_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = AppConfig(
                config_path=root / "config.json",
                history_path=root / "history.db",
                events_dir=root / "events",
            )
            app = create_app(settings)
            endpoint = next(
                route.endpoint for route in app.routes
                if getattr(route, "path", None) == "/api/config"
                and "PATCH" in getattr(route, "methods", set())
            )
            result = asyncio.run(endpoint(ConfigPatch(holdSeconds=4.5)))
            saved = load_runtime_config(settings.config_path)
            app.state.service.close()

        self.assertEqual(result["holdSeconds"], 4.5)
        self.assertEqual(saved["holdSeconds"], 4.5)
        self.assertIn("compressorThreshold", saved)
        self.assertIn("eventPostSeconds", saved)

    def test_atomic_round_trip_and_startup_application(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            payload = {
                "compressorThreshold": -47.5,
                "fanMidThreshold": -61.0,
                "fanHighThreshold": -64.0,
                "fanRequire": "both",
                "medianSeconds": 0.7,
                "holdSeconds": 3.0,
                "eventPreSeconds": 10.0,
                "eventPostSeconds": 12.0,
            }
            write_runtime_config(path, payload)
            self.assertEqual(load_runtime_config(path), payload)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

            settings = AppConfig(config_path=path)
            changes = load_saved_config(settings)

        self.assertEqual(changes, payload)
        self.assertEqual(settings.classifier.compressor_threshold, -47.5)
        self.assertEqual(settings.classifier.median_seconds, 0.7)
        self.assertEqual(settings.events.pre_seconds, 10.0)
        self.assertEqual(settings.events.post_seconds, 12.0)

    def test_missing_file_uses_defaults_and_invalid_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            settings = AppConfig(config_path=path)
            self.assertEqual(load_saved_config(settings), {})

            path.write_text(
                json.dumps({"fanRequire": "sometimes"}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_saved_config(settings)


if __name__ == "__main__":
    unittest.main()
