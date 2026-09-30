import sys
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "audio"
sys.path.insert(0, str(TOOLS))

from backend.app import _is_benign_proactor_reset  # noqa: E402


class WindowsResetFilterTests(unittest.TestCase):
    @staticmethod
    def reset_error(code=10054):
        error = ConnectionResetError("connection reset")
        error.winerror = code
        return error

    def test_only_ignores_reset_from_proactor_close_callback(self):
        context = {
            "message": (
                "Exception in callback "
                "_ProactorBasePipeTransport._call_connection_lost()"
            ),
            "exception": self.reset_error(),
        }
        with patch("backend.app.sys.platform", "win32"):
            self.assertTrue(_is_benign_proactor_reset(context))

            context["message"] = "Fatal read error on pipe transport"
            self.assertFalse(_is_benign_proactor_reset(context))

    def test_does_not_hide_other_errors_or_platforms(self):
        context = {
            "message": (
                "Exception in callback "
                "_ProactorBasePipeTransport._call_connection_lost()"
            ),
            "exception": self.reset_error(10053),
        }
        with patch("backend.app.sys.platform", "win32"):
            self.assertFalse(_is_benign_proactor_reset(context))
        context["exception"] = self.reset_error()
        with patch("backend.app.sys.platform", "linux"):
            self.assertFalse(_is_benign_proactor_reset(context))


if __name__ == "__main__":
    unittest.main()
