"""Publish Watson's v2 observations to an MQTT broker (HiveMQ Cloud).

Topics, under a prefix (default ``tester``):

    tester/fan          "1" / "0"   when fan_detected changes (retained)
    tester/compressor   "1" / "0"   when compressor_detected changes (retained)
    tester/beep         "1"         every time a beep is found (not retained)

Fan and compressor are state, so the latest value is retained and a
subscriber that connects later still learns it. A beep is an event with
no "off": it is always "1", and is not retained, so a late subscriber is
not told about a beep that already happened.

It also listens on ``<prefix>/sent``. A message there, whose payload is
the code of a command the controller just sent, is recorded as a command
marker (see ``POST /api/commands``): events within 10 s carry it as
``commandContext``. Retained messages are ignored, so a stale code is not
replayed as a new command after a reconnect. Watson only records it; it
never changes what is detected.

This publishes what Watson observed and nothing more. It does not say the
air conditioner is on or that a command worked.

Credentials come from the environment (a ``.env`` file is read if there
is one), never from the code:

    MQTT_HOST        broker host, e.g. xxxx.s1.eu.hivemq.cloud (unset = off)
    MQTT_PORT        8883 (TLS)
    MQTT_USERNAME
    MQTT_PASSWORD
    MQTT_TOPIC_PREFIX  tester
    MQTT_CLIENT_ID     optional
    MQTT_SENT_LATENCY_SECONDS  how late a ``sent`` code arrives (2.0)

HiveMQ Cloud presents a publicly trusted certificate, so TLS uses the
system trust store and the connection is verified.
"""

from __future__ import annotations

import os
import ssl
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

DEFAULT_PORT = 8883
DEFAULT_PREFIX = "tester"
TOPICS = ("fan", "compressor", "beep")


def load_env_file(path: Path, environ: dict | None = None) -> bool:
    """Read KEY=VALUE lines into the environment.

    Variables that are already set win, so a real environment (Docker,
    the shell) overrides the file. Returns whether the file existed.
    """
    environ = os.environ if environ is None else environ

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False

    for line in text.splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        environ.setdefault(key.strip(), value)

    return True


@dataclass(frozen=True)
class MqttSettings:
    host: str
    port: int = DEFAULT_PORT
    username: str = ""
    password: str = ""
    prefix: str = DEFAULT_PREFIX
    client_id: str = ""
    # Seconds between a command really being sent and its code reaching
    # us on ``<prefix>/sent`` (controller -> broker -> here).
    sent_latency: float = 2.0

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "MqttSettings | None":
        """Settings from the environment, or None when MQTT is not set up."""
        environ = os.environ if environ is None else environ
        host = environ.get("MQTT_HOST", "").strip()

        if not host:
            return None

        return cls(
            host=host,
            port=int(environ.get("MQTT_PORT") or DEFAULT_PORT),
            username=environ.get("MQTT_USERNAME", ""),
            password=environ.get("MQTT_PASSWORD", ""),
            prefix=(environ.get("MQTT_TOPIC_PREFIX") or DEFAULT_PREFIX)
            .strip("/"),
            client_id=environ.get("MQTT_CLIENT_ID", ""),
            sent_latency=float(
                environ.get("MQTT_SENT_LATENCY_SECONDS") or 2.0
            ),
        )


class MqttPublisher:
    """A background MQTT connection that publishes observations.

    paho runs its own network thread and reconnects by itself; messages
    published while disconnected are queued (QoS 1) and sent on
    reconnect. ``publish_*`` never blocks the audio thread and never
    raises: a broker problem must not stop the classifier.
    """

    def __init__(
        self,
        settings: MqttSettings,
        client_factory: Callable[[], object] | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.settings = settings
        self.log = log
        self.lock = threading.Lock()
        self.connected = False
        self.last_error = ""
        self.published = 0

        self.client = (client_factory or self._default_client)()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # Called with the code from ``<prefix>/sent``; set by the owner.
        self.on_sent: Callable[[str], None] | None = None
        self.sent_received = 0

    # ------------------------------------------------------ connection

    def _default_client(self):
        import paho.mqtt.client as mqtt

        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.settings.client_id,
        )
        client.tls_set_context(ssl.create_default_context())

        if self.settings.username:
            client.username_pw_set(
                self.settings.username, self.settings.password
            )

        client.reconnect_delay_set(min_delay=1, max_delay=30)

        return client

    def start(self) -> None:
        """Connect in the background; returns at once."""
        try:
            self.client.connect_async(
                self.settings.host, self.settings.port, keepalive=30
            )
            self.client.loop_start()
        except Exception as error:               # noqa: BLE001
            self._fail(f"cannot start: {error}")

    def stop(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:                        # noqa: BLE001
            pass

    def _on_connect(self, client, userdata, flags, reason, properties=None):
        failed = getattr(reason, "is_failure", False)

        with self.lock:
            self.connected = not failed

        if failed:
            self._fail(f"refused: {reason}")
        else:
            self.last_error = ""
            try:
                client.subscribe(self.topic("sent"), qos=1)
            except Exception as error:           # noqa: BLE001
                self._fail(f"cannot subscribe to sent: {error}")
            self.log(
                f"mqtt: connected to {self.settings.host}:"
                f"{self.settings.port}, publishing under "
                f"{self.settings.prefix}/"
            )

    def _on_disconnect(
        self, client, userdata, flags=None, reason=None, properties=None
    ):
        with self.lock:
            was = self.connected
            self.connected = False

        if was:
            self.log(f"mqtt: disconnected ({reason}); retrying")

    def _on_message(self, client, userdata, message) -> None:
        if message.topic != self.topic("sent") or message.retain:
            return

        try:
            code = bytes(message.payload).decode("utf-8", "replace").strip()
        except Exception:                        # noqa: BLE001
            return

        if not code or self.on_sent is None:
            return

        self.sent_received += 1

        try:
            self.on_sent(code[:60])
        except Exception as error:               # noqa: BLE001
            self._fail(f"handling sent {code!r} failed: {error}")

    def _fail(self, message: str) -> None:
        self.last_error = message
        self.log(f"mqtt: {message}")

    # ------------------------------------------------------ publishing

    def topic(self, name: str) -> str:
        return f"{self.settings.prefix}/{name}"

    def _publish(self, name: str, payload: str, retain: bool) -> None:
        try:
            self.client.publish(
                self.topic(name), payload, qos=1, retain=retain
            )
            self.published += 1
        except Exception as error:               # noqa: BLE001
            self._fail(f"publish {name} failed: {error}")

    def publish_observation(self, name: str, value: bool) -> None:
        """``fan`` or ``compressor``: "1" or "0", retained."""
        if name not in ("fan", "compressor"):
            raise ValueError(f"not an observation: {name!r}")

        self._publish(name, "1" if value else "0", retain=True)

    def publish_beep(self) -> None:
        """A beep happened. Always "1", never retained."""
        self._publish("beep", "1", retain=False)

    def status(self) -> dict:
        return {
            "host": self.settings.host,
            "connected": self.connected,
            "published": self.published,
            "sentReceived": self.sent_received,
            "error": self.last_error,
        }
