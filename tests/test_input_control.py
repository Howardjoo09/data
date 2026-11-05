"""Exercise key routing without installing a global hook or sending real input."""
import ctypes
import importlib.util
from pathlib import Path
import threading
import unittest
from unittest.mock import MagicMock, patch

from engine import Edit, Player


def load_simulated_backend():
    fake_user32 = MagicMock()
    fake_user32.GetForegroundWindow.return_value = 101
    fake_user32.GetAsyncKeyState.return_value = 0
    fake_user32.CallNextHookEx.return_value = 0
    fake_kernel32 = MagicMock()
    path = Path(__file__).resolve().parents[1] / "windows_input.py"
    spec = importlib.util.spec_from_file_location("simulated_windows_input", path)
    module = importlib.util.module_from_spec(spec)
    with patch("os.name", "nt"), patch.object(ctypes, "WinDLL", create=True,
            side_effect=lambda name, **kwargs: fake_user32 if name == "user32" else fake_kernel32), \
            patch.object(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, create=True):
        spec.loader.exec_module(module)
    return module


class InputControlTests(unittest.TestCase):
    def setUp(self):
        self.backend = load_simulated_backend()
        self.control = self.backend.Controller.__new__(self.backend.Controller)
        self.control.player = Player("각")
        self.control.lock = threading.RLock()
        self.control.own_window = 1
        self.control.target = None
        self.control.exit_requested = False
        self.control.message = ""
        self.control.down = set()
        self.control.swallowed = set()
        self.sender = patch.object(self.backend, "send_edit").start()
        self.addCleanup(patch.stopall)

    def event(self, vk, down=True, injected=False):
        data = self.backend.KeyboardData(vkCode=vk, flags=0x10 if injected else 0)
        return self.control._keyboard(0, 0x100 if down else 0x101, ctypes.addressof(data))

    def tap(self, vk):
        return self.event(vk), self.event(vk, False)

    def test_start_repeat_pause_resume_and_completion(self):
        self.assertEqual(self.tap(0x75), (1, 1))
        self.assertEqual(self.event(0x41), 1)
        self.event(0x41)  # held key repeat
        self.assertEqual(self.sender.call_count, 1)
        self.event(0x41, False)
        self.tap(0x76)
        self.assertFalse(self.control.player.active)
        self.tap(0x75)
        self.tap(0x20)
        self.sender.assert_called_with(Edit(1, "가"))
        self.tap(0x41)
        self.sender.assert_called_with(Edit(1, "각"))
        self.assertTrue(self.control.player.done)
        self.assertEqual(self.tap(0x41), (0, 0))

    def test_injected_events_do_not_advance(self):
        self.tap(0x75)
        self.event(0x41, injected=True)
        self.sender.assert_not_called()

    def test_manual_edit_and_mouse_click_detach_replacement(self):
        self.tap(0x75)
        self.tap(0x41)
        self.tap(0x76)
        self.assertEqual(self.tap(0x42), (0, 0))
        self.tap(0x75)
        self.tap(0x41)
        self.sender.assert_called_with(Edit(0, "가"))
        self.control._mouse(0, 0x201, 0)
        self.tap(0x41)
        self.sender.assert_called_with(Edit(0, "각"))

    def test_window_change_pauses_before_sending(self):
        self.tap(0x75)
        self.backend.user32.GetForegroundWindow.return_value = 202
        self.assertEqual(self.tap(0x41), (0, 0))
        self.sender.assert_not_called()
        self.assertFalse(self.control.player.active)

    def test_shortcuts_pass_through_and_escape_exits(self):
        self.tap(0x75)
        self.backend.user32.GetAsyncKeyState.return_value = -32768
        self.assertEqual(self.tap(0x43), (0, 0))
        self.sender.assert_not_called()
        self.assertEqual(self.tap(0x1B), (1, 1))
        self.assertTrue(self.control.exit_requested)

    def test_send_failure_does_not_advance_and_pauses(self):
        self.tap(0x75)
        self.sender.side_effect = RuntimeError("failed")
        self.tap(0x41)
        self.assertEqual((self.control.player.index, self.control.player.stage), (0, 0))
        self.assertFalse(self.control.player.active)
        self.assertEqual(self.control.message, "failed")


if __name__ == "__main__":
    unittest.main()
