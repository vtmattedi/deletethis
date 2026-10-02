"""MQTT publishing: topics, payloads, retain, and that it can't break Watson."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "audio"))

from backend.mqtt_publisher import (  # noqa: E402
    MqttPublisher,
    MqttSettings,
    load_env_file,
)


class FakeClient:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    def publish(self, topic, payload, qos=0, retain=False):
        if self.fail:
            raise OSError("broker down")
        self.sent.append((topic, payload, qos, retain))

    def subscribe(self, topic, qos=0):
        self.subscribed = getattr(self, "subscribed", []) + [topic]

    def connect_async(self, *a, **k):
        self.connected_to = a

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass


def publisher(fail=False, prefix="tester"):
    client = FakeClient(fail)
    pub = MqttPublisher(
        MqttSettings(host="h.example", prefix=prefix),
        client_factory=lambda: client, log=lambda m: None,
    )
    return pub, client


class PublishTests(unittest.TestCase):
    def test_fan_and_compressor_are_one_or_zero_and_retained(self):
        pub, client = publisher()
        pub.publish_observation("fan", True)
        pub.publish_observation("compressor", False)

        self.assertEqual(client.sent, [
            ("tester/fan", "1", 1, True),
            ("tester/compressor", "0", 1, True),
        ])

    def test_a_beep_is_always_one_and_not_retained(self):
        pub, client = publisher()
        pub.publish_beep()
        pub.publish_beep()

        self.assertEqual(client.sent, [("tester/beep", "1", 1, False)] * 2)

    def test_only_fan_and_compressor_are_observations(self):
        pub, _ = publisher()
        with self.assertRaises(ValueError):
            pub.publish_observation("beep", True)

    def test_the_prefix_is_configurable(self):
        pub, client = publisher(prefix="lab")
        pub.publish_beep()
        self.assertEqual(client.sent[0][0], "lab/beep")

    def test_a_broker_failure_never_raises(self):
        pub, _ = publisher(fail=True)
        pub.publish_observation("fan", True)
        pub.publish_beep()
        self.assertIn("failed", pub.status()["error"])


class Message:
    def __init__(self, topic, payload, retain=False):
        self.topic, self.payload, self.retain = topic, payload, retain


class SentTests(unittest.TestCase):
    def setUp(self):
        self.pub, self.client = publisher()
        self.codes = []
        self.pub.on_sent = self.codes.append

    def test_it_subscribes_to_sent_on_connect(self):
        self.pub._on_connect(self.client, None, None, 0)
        self.assertEqual(self.client.subscribed, ["tester/sent"])

    def test_a_code_is_handed_over(self):
        self.pub._on_message(self.client, None, Message("tester/sent", b" 0x20DF10EF "))
        self.assertEqual(self.codes, ["0x20DF10EF"])
        self.assertEqual(self.pub.status()["sentReceived"], 1)

    def test_retained_empty_and_other_topics_are_ignored(self):
        for m in (Message("tester/sent", b"A1", retain=True),
                  Message("tester/sent", b"  "),
                  Message("tester/fan", b"1")):
            self.pub._on_message(self.client, None, m)
        self.assertEqual(self.codes, [])

    def test_a_failing_handler_does_not_raise(self):
        def boom(code):
            raise RuntimeError("x")
        self.pub.on_sent = boom
        self.pub._on_message(self.client, None, Message("tester/sent", b"A1"))
        self.assertIn("failed", self.pub.status()["error"])


class SettingsTests(unittest.TestCase):
    def test_unset_host_means_off(self):
        self.assertIsNone(MqttSettings.from_env({}))
        self.assertIsNone(MqttSettings.from_env({"MQTT_HOST": "  "}))

    def test_read_from_environment(self):
        s = MqttSettings.from_env({
            "MQTT_HOST": "x.hivemq.cloud", "MQTT_USERNAME": "u",
            "MQTT_PASSWORD": "p", "MQTT_TOPIC_PREFIX": "/tester/",
        })
        self.assertEqual((s.host, s.port, s.username, s.password, s.prefix),
                         ("x.hivemq.cloud", 8883, "u", "p", "tester"))

    def test_env_file_does_not_override_the_environment(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".env"
            path.write_text(
                "# comment\nMQTT_HOST=from-file\nMQTT_PASSWORD='s3cret'\n"
                "MQTT_USERNAME=file-user\n", encoding="utf-8")
            env = {"MQTT_USERNAME": "shell-user"}

            self.assertTrue(load_env_file(path, env))
            self.assertEqual(env["MQTT_HOST"], "from-file")
            self.assertEqual(env["MQTT_PASSWORD"], "s3cret")
            self.assertEqual(env["MQTT_USERNAME"], "shell-user")
            self.assertFalse(load_env_file(Path(d) / "none", env))


if __name__ == "__main__":
    unittest.main()
