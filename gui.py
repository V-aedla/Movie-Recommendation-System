"""Simple Tkinter window for the movie recommender.  Run: python gui.py"""
import tkinter as tk
from tkinter import scrolledtext

from assistant import ChatAssistant


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Movie Recommender")
        self.geometry("620x480")
        self.configure(bg="#f2f2f2")
        self.bot = ChatAssistant()

        self.log = scrolledtext.ScrolledText(self, wrap="word", state="disabled",
                                             font=("Arial", 11), bg="white", fg="#222222")
        self.log.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        bar = tk.Frame(self, bg="#f2f2f2")
        bar.pack(fill="x", padx=10, pady=(0, 10))
        self.entry = tk.Entry(bar, font=("Arial", 11))
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _e: self.send())
        tk.Button(bar, text="Send", command=self.send).pack(side="left", padx=(6, 0))
        tk.Button(bar, text="Tonight", command=lambda: self.say("tonight")).pack(side="left", padx=(6, 0))

        self.write("Assistant", self.bot.greeting())
        self.write("Assistant", "Type 'help' to see what I can do.")
        self.entry.focus()

    def write(self, who: str, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"{who}: {text}\n\n")
        self.log.configure(state="disabled")
        self.log.see("end")

    def say(self, text: str) -> None:
        self.write("You", text)
        self.write("Assistant", self.bot.handle(text))

    def send(self) -> None:
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, "end")
            self.say(text)


if __name__ == "__main__":
    App().mainloop()
