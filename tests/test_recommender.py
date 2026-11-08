import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from assistant import ChatAssistant
from recommender import HybridRecommender, part_of_day


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = HybridRecommender()

    def test_fuzzy_title_match(self):
        self.assertEqual(self.engine.titles[self.engine.find_title("matrx")], "The Matrix")

    def test_similar_excludes_self_and_returns_n(self):
        recs = self.engine.similar_to("Inception", n=5)
        self.assertEqual(len(recs), 5)
        self.assertNotIn("Inception", [r.title for r in recs])

    def test_similar_to_scifi_is_scifi(self):
        top = self.engine.similar_to("The Matrix", n=3)
        self.assertTrue(all("Sci-Fi" in r.genres or "Action" in r.genres for r in top))

    def test_text_query_scary_returns_horror(self):
        recs = self.engine.recommend_from_text("something scary", n=3)
        self.assertTrue(any("Horror" in r.genres for r in recs))

    def test_rated_films_are_not_recommended(self):
        profile = {"Inception": 5, "The Matrix": 5}
        titles = [r.title for r in self.engine.recommend_for_ratings(profile, n=10)]
        self.assertFalse(set(titles) & set(profile))

    def test_dislikes_push_genre_down(self):
        liked = self.engine.recommend_for_ratings({"Toy Story": 5}, n=3)
        disliked = self.engine.recommend_for_ratings({"Toy Story": 1}, n=3)
        self.assertNotEqual([r.title for r in liked], [r.title for r in disliked])

    def test_genre_filter(self):
        recs = self.engine.popular(5, genre="horror")
        self.assertTrue(all("Horror" in r.genres for r in recs))

    def test_part_of_day(self):
        self.assertEqual(part_of_day(8), "morning")
        self.assertEqual(part_of_day(23), "late night")

    def test_proactive_changes_with_time(self):
        _, morning = self.engine.proactive({}, now=datetime(2026, 11, 8, 9))
        _, night = self.engine.proactive({}, now=datetime(2026, 11, 8, 23))
        self.assertNotEqual([r.title for r in morning], [r.title for r in night])


class AssistantTests(unittest.TestCase):
    def test_rate_and_recommend_flow(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.json"
            bot = ChatAssistant(profile_path=path)
            self.assertIn("Saved", bot.handle("rate Inception 5"))
            self.assertTrue(path.exists())
            self.assertIn("Because you rated Inception", bot.handle("recommend"))
            self.assertIn("1.", bot.handle("movies like The Matrix"))
            self.assertIn("cleared", bot.handle("reset"))

    def test_unknown_title(self):
        bot = ChatAssistant(profile_path=None)
        self.assertIn("do not know", bot.handle("rate zzzzzzzz 3"))


if __name__ == "__main__":
    unittest.main()
