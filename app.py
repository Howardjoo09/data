"""Run on Windows with: py -3 app.py"""
import os
import sys


def main():
    if os.name != "nt":
        print("Windows 10/11용 프로그램입니다. Windows에서 실행하세요.", file=sys.stderr)
        return 1
    import tkinter as tk
    from tkinter import messagebox, ttk
    from windows_input import Controller

    root = tk.Tk()
    root.title("한 글자씩 · Google Docs 입력 도우미")
    root.geometry("740x620")
    root.minsize(600, 510)
    root.option_add("*Font", ("맑은 고딕", 10))
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="한 글자씩", font=("맑은 고딕", 22, "bold")).pack(anchor="w")
    ttk.Label(frame, text="글 준비 → Google Docs 클릭 → F6 → 문자 키 또는 스페이스",
              wraplength=680).pack(anchor="w", pady=(4, 15))
    editor = tk.Text(frame, wrap="word", height=12, undo=True, padx=12, pady=12)
    editor.pack(fill="both", expand=True)
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=12)
    status = tk.StringVar()
    progress = tk.StringVar(value="아직 준비된 글이 없습니다.")
    root.update_idletasks()
    controller = Controller(root.winfo_id())

    def prepare():
        text = editor.get("1.0", "end-1c")
        if not text:
            messagebox.showinfo("글 준비", "입력할 글을 먼저 붙여넣으세요.")
            return
        if any(ord(c) < 32 and c not in "\n\r\t" for c in text):
            messagebox.showerror("글 확인", "줄바꿈·탭 이외의 제어 문자를 제거하세요.")
            return
        controller.load(text)

    prepare_button = ttk.Button(buttons, text="이 글로 준비 / 처음부터", command=prepare)
    prepare_button.pack(side="left")
    ttk.Button(buttons, text="일시 중단 (F7)", command=controller.pause).pack(side="right")
    ttk.Label(frame, textvariable=progress).pack(anchor="w")
    ttk.Label(frame, textvariable=status, wraplength=680).pack(anchor="w", pady=8)
    ttk.Separator(frame).pack(fill="x", pady=10)
    ttk.Label(frame, text="F6 시작·재개     F7 일반 타자로 전환     Esc 프로그램 종료\n"
              "예: 각 = ㄱ → 가 → 각 · 영어/공백/줄바꿈은 한 번에 한 글자\n"
              "중단 중 수정·커서 이동 후 재개하면 다음 단계 글자를 현재 위치에 새로 씁니다.\n"
              "먼저 빈 문서에서 확인하세요. 단축키는 프로그램 실행 중 다른 창에서도 작동합니다.",
              wraplength=680, foreground="#555555").pack(anchor="w")

    def close():
        controller.close()
        root.destroy()

    def refresh():
        active, index, stage, total, done, message, exit_requested = controller.snapshot()
        if exit_requested:
            close()
            return
        progress.set(f"{'입력 중' if active else '완료' if done and total else '대기'} · "
                     f"완료한 글자 {index} / {total} · 현재 글자의 완료 단계 {stage}")
        status.set(message)
        prepare_button.configure(state="disabled" if active else "normal")
        root.after(100, refresh)

    root.protocol("WM_DELETE_WINDOW", close)
    refresh()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
