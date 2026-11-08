"""Chat-style front end shared by the command-line app and the GUI."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from recommender import HybridRecommender, Recommendation

PROFILE_PATH = Path(__file__).parent / "user_profile.json"

HELP = """You can talk to me in plain language or use commands:
  movies like Inception        find films similar to one you know
  something scary and tense    describe a mood or theme
  rate Inception 5             rate a film 1-5 (this trains your profile)
  recommend                    personalised picks from your ratings
  genre horror                 popular films in a genre
  tonight                      a proactive suggestion for right now
  profile                      show your ratings
  reset                        clear your ratings
  help / quit"""


def format_recs(recs: list[Recommendation]) -> str:
    if not recs:
        return "I could not find anything for that. Try different words."
    lines = []
    for k, r in enumerate(recs, 1):
        lines.append(f"{k}. {r.title} ({r.year}) - {r.genres}\n   {r.reason}")
    return "\n".join(lines)


class ChatAssistant:
    def __init__(self, engine: HybridRecommender | None = None,
                 profile_path: Path | None = PROFILE_PATH):
        self.engine = engine or HybridRecommender()
        self.profile_path = profile_path
        self.ratings: dict[str, float] = {}
        if profile_path and Path(profile_path).exists():
            self.ratings = json.loads(Path(profile_path).read_text())

    def _save(self) -> None:
        if self.profile_path:
            Path(self.profile_path).write_text(json.dumps(self.ratings, indent=2))

    def greeting(self, now: datetime | None = None) -> str:
        """Proactive opening message: the assistant speaks first."""
        period, recs = self.engine.proactive(self.ratings, now=now, n=3)
        return (f"Good {period}! Without being asked, here is what I would "
                f"suggest right now:\n{format_recs(recs)}")

    def handle(self, text: str) -> str:
        text = text.strip()
        low = text.lower()
        if not text:
            return "Type 'help' to see what I can do."
        if low in {"help", "?"}:
            return HELP
        if low == "tonight":
            return self.greeting()
        if low == "profile":
            if not self.ratings:
                return "You have not rated anything yet. Try: rate Inception 5"
            return "\n".join(f"{t}: {r:g}" for t, r in sorted(self.ratings.items()))
        if low == "reset":
            self.ratings.clear()
            self._save()
            return "Your ratings have been cleared."
        if low == "recommend":
            if not self.ratings:
                return ("I need a few ratings first, so here are community favourites:\n"
                        + format_recs(self.engine.popular(5)))
            return format_recs(self.engine.recommend_for_ratings(self.ratings, 5))

        m = re.match(r"^rate\s+(.+?)\s+([1-5])$", text, re.I)
        if m:
            i = self.engine.find_title(m.group(1))
            if i is None:
                return f"I do not know a film called '{m.group(1)}'."
            title = self.engine.titles[i]
            self.ratings[title] = float(m.group(2))
            self._save()
            return f"Saved: {title} = {m.group(2)}. Type 'recommend' for new picks."

        m = re.match(r"^genre\s+(.+)$", low)
        if m:
            seen = {self.engine.find_title(t) for t in self.ratings}
            return format_recs(self.engine.popular(5, genre=m.group(1),
                                                   exclude={s for s in seen if s is not None}))

        m = re.match(r"^(?:movies?|films?)?\s*(?:like|similar to)\s+(.+)$", low)
        if m:
            recs = self.engine.similar_to(m.group(1), n=5)
            if not recs:
                return f"I do not know a film called '{m.group(1)}'."
            return format_recs(recs)

        return format_recs(self.engine.recommend_from_text(text, 5))
