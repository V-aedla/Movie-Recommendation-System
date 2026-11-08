"""Hybrid movie recommendation engine.

Foundation: the classic content-based approach taught in most introductory
recommender tutorials (TF-IDF vectors of movie text + cosine similarity, as
documented in the scikit-learn user guide).

Extensions added in this project:
  1. Item-based collaborative filtering from a ratings matrix.
  2. A tunable hybrid of the two similarity matrices (alpha).
  3. A personal profile built from the user's own ratings (likes AND dislikes).
  4. Free-text / mood queries mapped onto the same TF-IDF space.
  5. A proactive "tonight" suggestion that adds time-of-day context.
  6. Human-readable explanations for every recommendation.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"

# Words a person might type about how they feel, mapped to catalogue vocabulary.
MOOD_TERMS = {
    "sad": "drama loss emotional",
    "happy": "comedy family friendship",
    "funny": "comedy friends",
    "scared": "horror thriller dark",
    "scary": "horror thriller dark",
    "romantic": "romance love couple",
    "excited": "action adventure chase",
    "relaxed": "family animation comedy",
    "smart": "mystery detective mind codebreakers",
    "tense": "thriller suspense",
}

# Genres that tend to suit each part of the day (used by the proactive mode).
TIME_BOOSTS = {
    "morning": {"Family", "Comedy", "Animation"},
    "afternoon": {"Adventure", "Comedy", "Documentary", "Fantasy"},
    "evening": {"Drama", "Romance", "Action", "Crime"},
    "late night": {"Thriller", "Horror", "Mystery", "Sci-Fi"},
}


def part_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "late night"


@dataclass
class Recommendation:
    title: str
    year: int
    genres: str
    score: float
    reason: str


class HybridRecommender:
    def __init__(self, movies: pd.DataFrame | None = None,
                 ratings: pd.DataFrame | None = None, alpha: float = 0.6):
        self.alpha = alpha
        self.movies = (movies if movies is not None
                       else pd.read_csv(DATA_DIR / "movies.csv")).reset_index(drop=True)
        self.ratings = (ratings if ratings is not None
                        else pd.read_csv(DATA_DIR / "ratings.csv"))
        self.titles = self.movies["title"].tolist()
        self._index = {t.lower(): i for i, t in enumerate(self.titles)}
        self._build_content_model()
        self._build_collaborative_model()
        self._build_popularity()

    # ------------------------------------------------------------------ models
    def _build_content_model(self) -> None:
        genre_text = (self.movies["genres"].str.replace("-", "", regex=False)
                      .str.replace("|", " ", regex=False).str.lower())
        # Genres are repeated so they weigh more than any single description word.
        text = (genre_text + " ") * 3 + self.movies["description"].str.lower()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                          sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform(text)
        self._terms = np.array(self.vectorizer.get_feature_names_out())
        self.content_sim = cosine_similarity(self.tfidf)

    def _build_collaborative_model(self) -> None:
        matrix = self.ratings.pivot_table(index="user_id", columns="movie_id",
                                          values="rating")
        matrix = matrix.reindex(columns=self.movies["movie_id"])
        centered = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0.0)
        sim = cosine_similarity(centered.T.values)
        self.collab_sim = np.clip(np.nan_to_num(sim), 0.0, 1.0)

    def _build_popularity(self) -> None:
        stats = self.ratings.groupby("movie_id")["rating"].agg(["mean", "count"])
        stats = stats.reindex(self.movies["movie_id"])
        mean_all = self.ratings["rating"].mean()
        m = 5  # prior weight: how many votes before we trust a film's own average
        votes = stats["count"].fillna(0).values
        avg = stats["mean"].fillna(mean_all).values
        self.popularity = (votes / (votes + m)) * avg + (m / (votes + m)) * mean_all

    def hybrid_matrix(self, alpha: float | None = None) -> np.ndarray:
        alpha = self.alpha if alpha is None else alpha
        content, collab = self.content_sim.copy(), self.collab_sim.copy()
        for mat in (content, collab):
            np.fill_diagonal(mat, 0.0)
        content = content / (content.max() or 1.0)
        collab = collab / (collab.max() or 1.0)
        return alpha * content + (1 - alpha) * collab

    # ----------------------------------------------------------------- helpers
    def find_title(self, query: str) -> int | None:
        q = query.strip().lower()
        if q in self._index:
            return self._index[q]
        contains = [t for t in self._index if q and q in t]
        if len(contains) == 1:
            return self._index[contains[0]]
        close = difflib.get_close_matches(q, list(self._index), n=1, cutoff=0.6)
        return self._index[close[0]] if close else None

    def _make(self, j: int, score: float, reason: str) -> Recommendation:
        row = self.movies.iloc[j]
        return Recommendation(row["title"], int(row["year"]),
                              row["genres"].replace("|", ", "), float(score), reason)

    def _shared_genres(self, i: int, j: int) -> list[str]:
        a = set(self.movies.iloc[i]["genres"].split("|"))
        b = set(self.movies.iloc[j]["genres"].split("|"))
        return sorted(a & b)

    def _shared_terms(self, i: int, j: int, k: int = 2) -> list[str]:
        both = np.minimum(self.tfidf[i].toarray()[0], self.tfidf[j].toarray()[0])
        genre_words = {w for g in self.movies["genres"] for w in
                       g.replace("-", "").lower().split("|")}
        out = []
        for idx in both.argsort()[::-1]:
            if both[idx] <= 0 or len(out) == k:
                break
            term = self._terms[idx]
            if term not in genre_words and " " not in term:
                out.append(term)
        return out

    def _explain(self, i: int, j: int, prefix: str) -> str:
        genres = self._shared_genres(i, j)
        terms = self._shared_terms(i, j)
        bits = []
        if genres:
            bits.append("shared genres: " + ", ".join(genres))
        if terms:
            bits.append("similar themes: " + ", ".join(terms))
        return f"{prefix} ({'; '.join(bits)})" if bits else prefix

    def _filter_genre(self, scores: np.ndarray, genre: str | None) -> np.ndarray:
        if not genre:
            return scores
        mask = self.movies["genres"].str.lower().str.contains(genre.lower(), regex=False)
        return np.where(mask.values, scores, -np.inf)

    # --------------------------------------------------------- recommendations
    def similar_to(self, title: str, n: int = 5, alpha: float | None = None,
                   genre: str | None = None) -> list[Recommendation]:
        i = self.find_title(title)
        if i is None:
            return []
        scores = self._filter_genre(self.hybrid_matrix(alpha)[i].copy(), genre)
        scores[i] = -np.inf
        return [self._make(j, scores[j], self._explain(i, j, f"Similar to {self.titles[i]}"))
                for j in np.argsort(scores)[::-1][:n] if np.isfinite(scores[j])]

    def recommend_for_ratings(self, user_ratings: dict[str, float], n: int = 5,
                              alpha: float | None = None,
                              genre: str | None = None) -> list[Recommendation]:
        rated = {self._index[t.lower()]: r for t, r in user_ratings.items()
                 if t.lower() in self._index}
        if not rated:
            return self.popular(n, genre)
        idx = list(rated)
        weights = np.array([rated[i] - 3.0 for i in idx])  # 3 = neutral
        if not weights.any():
            weights = np.full(len(idx), 0.5)
        sim = self.hybrid_matrix(alpha)
        scores = (weights @ sim[idx]) / np.abs(weights).sum()
        scores = self._filter_genre(scores, genre)
        scores[idx] = -np.inf
        results = []
        for j in np.argsort(scores)[::-1][:n]:
            if not np.isfinite(scores[j]):
                continue
            contrib = weights * sim[idx, j]
            best = idx[int(contrib.argmax())]
            results.append(self._make(j, scores[j], self._explain(
                best, j, f"Because you rated {self.titles[best]} highly")))
        return results

    def recommend_from_text(self, query: str, n: int = 5) -> list[Recommendation]:
        words = query.lower().replace(",", " ").split()
        expanded = " ".join(words + [MOOD_TERMS[w] for w in words if w in MOOD_TERMS])
        sims = cosine_similarity(self.vectorizer.transform([expanded]), self.tfidf)[0]
        if sims.max() <= 0:
            return []
        out = []
        for j in np.argsort(sims)[::-1][:n]:
            if sims[j] > 0:
                out.append(self._make(j, sims[j], "Matches the words in your request"))
        return out

    def popular(self, n: int = 5, genre: str | None = None,
                exclude: set[int] | None = None) -> list[Recommendation]:
        scores = self._filter_genre(self.popularity.copy(), genre)
        for i in exclude or ():
            scores[i] = -np.inf
        return [self._make(j, scores[j], "Highly rated by the community")
                for j in np.argsort(scores)[::-1][:n] if np.isfinite(scores[j])]

    def proactive(self, user_ratings: dict[str, float], now: datetime | None = None,
                  n: int = 3) -> tuple[str, list[Recommendation]]:
        """Suggest films without being asked, using taste plus time of day."""
        now = now or datetime.now()
        period = part_of_day(now.hour)
        boost = TIME_BOOSTS[period]
        rated = {self._index[t.lower()] for t in user_ratings if t.lower() in self._index}

        taste = np.zeros(len(self.titles))
        if rated:
            for rec in self.recommend_for_ratings(user_ratings, n=len(self.titles)):
                taste[self._index[rec.title.lower()]] = rec.score
        else:
            taste = self.popularity.copy()
        if taste.max() > 0:
            taste = taste / taste.max()

        context = np.array([len(set(g.split("|")) & boost) / len(g.split("|"))
                            for g in self.movies["genres"]])
        scores = 0.7 * taste + 0.3 * context
        for i in rated:
            scores[i] = -np.inf
        recs = []
        for j in np.argsort(scores)[::-1][:n]:
            recs.append(self._make(j, scores[j],
                                   f"Fits your taste and a {period} watch"))
        return period, recs
