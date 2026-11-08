"""Command-line chat for the movie recommender.  Run: python app.py"""
import sys

from assistant import ChatAssistant


def main() -> None:
    bot = ChatAssistant()
    print("Movie Recommender (hybrid AI).  Type 'help' for commands.\n")
    if "--quiet" not in sys.argv:
        print(bot.greeting(), "\n")
    while True:
        try:
            text = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in {"quit", "exit"}:
            break
        print(bot.handle(text), "\n")


if __name__ == "__main__":
    main()
