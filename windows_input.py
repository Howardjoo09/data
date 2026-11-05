"""Windows global keyboard hook and Unicode input. No third-party dependencies."""
import ctypes as C
from ctypes import wintypes as W
import os
import threading

from engine import Player

if os.name != "nt":
    raise OSError("이 프로그램은 Windows 10/11에서 실행하세요.")

user32 = C.WinDLL("user32", use_last_error=True)
kernel32 = C.WinDLL("kernel32", use_last_error=True)
ULONG_PTR = W.WPARAM
LRESULT = W.LPARAM
HOOKPROC = C.WINFUNCTYPE(LRESULT, C.c_int, W.WPARAM, W.LPARAM)


class KeyboardData(C.Structure):
    _fields_ = [("vkCode", W.DWORD), ("scanCode", W.DWORD), ("flags", W.DWORD),
                ("time", W.DWORD), ("dwExtraInfo", ULONG_PTR)]


class MouseInput(C.Structure):
    _fields_ = [("dx", W.LONG), ("dy", W.LONG), ("mouseData", W.DWORD),
                ("dwFlags", W.DWORD), ("time", W.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KeyboardInput(C.Structure):
    _fields_ = [("wVk", W.WORD), ("wScan", W.WORD), ("dwFlags", W.DWORD),
                ("time", W.DWORD), ("dwExtraInfo", ULONG_PTR)]


class HardwareInput(C.Structure):
    _fields_ = [("uMsg", W.DWORD), ("wParamL", W.WORD), ("wParamH", W.WORD)]


class InputUnion(C.Union):
    _fields_ = [("mi", MouseInput), ("ki", KeyboardInput), ("hi", HardwareInput)]


class Input(C.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("type", W.DWORD), ("value", InputUnion)]


user32.SetWindowsHookExW.argtypes = [C.c_int, HOOKPROC, W.HINSTANCE, W.DWORD]
user32.SetWindowsHookExW.restype = W.HANDLE
user32.CallNextHookEx.argtypes = [W.HANDLE, C.c_int, W.WPARAM, W.LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [W.HANDLE]
user32.GetForegroundWindow.restype = W.HWND
user32.GetAncestor.argtypes = [W.HWND, W.UINT]
user32.GetAncestor.restype = W.HWND
user32.GetAsyncKeyState.argtypes = [C.c_int]
user32.GetAsyncKeyState.restype = C.c_short
user32.SendInput.argtypes = [W.UINT, C.POINTER(Input), C.c_int]
user32.SendInput.restype = W.UINT
user32.GetMessageW.argtypes = [C.POINTER(W.MSG), W.HWND, W.UINT, W.UINT]
user32.GetMessageW.restype = W.BOOL
user32.PostThreadMessageW.argtypes = [W.DWORD, W.UINT, W.WPARAM, W.LPARAM]
kernel32.GetModuleHandleW.argtypes = [W.LPCWSTR]
kernel32.GetModuleHandleW.restype = W.HMODULE
kernel32.GetCurrentThreadId.restype = W.DWORD


def send_edit(edit):
    events = []

    def pair(vk, scan=0, flags=0):
        events.append(Input(type=1, ki=KeyboardInput(vk, scan, flags, 0, 0)))
        events.append(Input(type=1, ki=KeyboardInput(vk, scan, flags | 2, 0, 0)))

    for _ in range(edit.erase):
        pair(8)
    # CR and Tab must be virtual keys; all other text uses UTF-16 Unicode input.
    if edit.text == "\n":
        pair(13)
    elif edit.text == "\t":
        pair(9)
    else:
        encoded = edit.text.encode("utf-16-le")
        for i in range(0, len(encoded), 2):
            pair(0, int.from_bytes(encoded[i:i + 2], "little"), 4)
    batch = (Input * len(events))(*events)
    if user32.SendInput(len(events), batch, C.sizeof(Input)) != len(events):
        raise RuntimeError("입력 전송 실패: 브라우저와 프로그램을 같은 권한으로 실행하세요. 문서를 확인한 뒤 다시 시작하세요.")


class Controller:
    def __init__(self, own_window):
        self.player = Player()
        self.lock = threading.RLock()
        self.own_window = user32.GetAncestor(own_window, 2)
        self.target = None
        self.exit_requested = False
        self.message = "글을 준비한 뒤 Google Docs에서 F6을 누르세요."
        self.ready = False
        self.thread_id = None
        self.down = set()
        self.swallowed = set()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def load(self, text):
        with self.lock:
            self.player.reset(text)
            self.target = None
            self.message = "준비 완료 · Google Docs의 입력 위치를 클릭하고 F6을 누르세요."

    def snapshot(self):
        with self.lock:
            if self.player.active and user32.GetForegroundWindow() != self.target:
                self.player.pause()
                self.player.detach()
                self.message = "창 전환으로 일시 중단 · 문서에서 F6으로 재개하세요."
            return (self.player.active, self.player.index, self.player.stage,
                    len(self.player.text), self.player.done, self.message, self.exit_requested)

    def pause(self):
        with self.lock:
            self.player.pause()
            self.message = "일시 중단 · 일반 타자 사용 가능 · F6으로 재개"

    def close(self):
        if self.thread_id:
            user32.PostThreadMessageW(self.thread_id, 0x12, 0, 0)
        self.thread.join(timeout=2)

    @staticmethod
    def _is_trigger(vk):
        return (vk == 0x20 or 0x30 <= vk <= 0x5A or 0x60 <= vk <= 0x6F
                or 0xBA <= vk <= 0xC0 or 0xDB <= vk <= 0xDF or vk == 0xE2)

    def _keyboard(self, code, message, address):
        if code < 0:
            return user32.CallNextHookEx(None, code, message, address)
        data = C.cast(address, C.POINTER(KeyboardData)).contents
        if data.flags & 0x10:  # Never treat synthetic Unicode/backspace as a user trigger.
            return user32.CallNextHookEx(None, code, message, address)
        vk = data.vkCode
        down = message in (0x100, 0x104)
        up = message in (0x101, 0x105)
        with self.lock:
            repeated = vk in self.down
            if down:
                self.down.add(vk)
            if up:
                self.down.discard(vk)
                if vk in self.swallowed:
                    self.swallowed.discard(vk)
                    return 1
            if down and vk in (0x75, 0x76, 0x1B):  # F6 / F7 / Esc
                self.swallowed.add(vk)
                if not repeated:
                    if vk == 0x1B:
                        self.player.pause()
                        self.exit_requested = True
                    elif vk == 0x76:
                        self.pause()
                    else:
                        window = user32.GetForegroundWindow()
                        if window == self.own_window:
                            self.message = "먼저 Google Docs의 입력 위치를 클릭하세요."
                        elif not self.player.text:
                            self.message = "먼저 입력할 글을 준비하세요."
                        elif self.player.done:
                            self.message = "입력 완료 · 처음부터 하려면 글을 다시 준비하세요."
                        else:
                            if self.target != window:
                                self.player.detach()
                            self.target = window
                            self.player.start()
                            self.message = "입력 중 · F7 일시 중단 / Esc 종료"
                return 1
            if down:
                window = user32.GetForegroundWindow()
                if self.player.active and window != self.target:
                    self.player.pause()
                    self.player.detach()
                shortcut = any(user32.GetAsyncKeyState(k) & 0x8000
                               for k in (0x11, 0x12, 0x5B, 0x5C))
                if self.player.active and self._is_trigger(vk) and not shortcut:
                    self.swallowed.add(vk)
                    if not repeated:
                        try:
                            send_edit(self.player.peek())
                            self.player.commit()
                            if self.player.done:
                                self.message = "입력 완료 · 일반 타자로 돌아왔습니다."
                        except Exception as exc:
                            self.player.pause()
                            self.player.detach()
                            self.message = str(exc)
                    return 1
                if vk in self.swallowed:
                    return 1
                # Manual typing/navigation invalidates the location of partial output.
                if vk not in (0x10, 0xA0, 0xA1):
                    self.player.detach()
        return user32.CallNextHookEx(None, code, message, address)

    def _mouse(self, code, message, address):
        if code >= 0 and message in (0x201, 0x204, 0x207, 0x20B, 0x20A, 0x20E):
            with self.lock:
                self.player.detach()
        return user32.CallNextHookEx(None, code, message, address)

    def _run(self):
        keyboard_hook = mouse_hook = None
        self._keyboard_callback = HOOKPROC(self._keyboard)
        self._mouse_callback = HOOKPROC(self._mouse)
        try:
            self.thread_id = kernel32.GetCurrentThreadId()
            # Create this thread's message queue before the UI can request shutdown.
            msg = W.MSG()
            user32.PeekMessageW(C.byref(msg), None, 0, 0, 0)
            module = kernel32.GetModuleHandleW(None)
            keyboard_hook = user32.SetWindowsHookExW(13, self._keyboard_callback, module, 0)
            if not keyboard_hook:
                raise C.WinError(C.get_last_error())
            mouse_hook = user32.SetWindowsHookExW(14, self._mouse_callback, module, 0)
            if not mouse_hook:
                raise C.WinError(C.get_last_error())
            self.ready = True
            while True:
                result = user32.GetMessageW(C.byref(msg), None, 0, 0)
                if result == 0:
                    break
                if result == -1:
                    raise C.WinError(C.get_last_error())
                user32.TranslateMessage(C.byref(msg))
                user32.DispatchMessageW(C.byref(msg))
        except Exception as exc:
            with self.lock:
                self.player.pause()
                self.message = f"키보드 연결 오류: {exc}"
        finally:
            self.ready = False
            if keyboard_hook:
                user32.UnhookWindowsHookEx(keyboard_hook)
            if mouse_hook:
                user32.UnhookWindowsHookEx(mouse_hook)
