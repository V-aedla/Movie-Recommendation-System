"""Offline evaluation: hide some of each user's liked films, then check how many
the model brings back in its top-N list (recall@N). Run: python evaluate.py"""
import numpy as np
import pandas as pd

from recommender import HybridRecommender

N = 10
SEED = 7


def run() -> dict[str, float]:
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    id_to_title = dict(zip(movies["movie_id"], movies["title"]))
    rng = np.random.default_rng(SEED)
    hits = {"Popularity": [], "Content only": [], "Collaborative only": [], "Hybrid": []}

    for user, rows in ratings.groupby("user_id"):
        liked = rows[rows["rating"] >= 4]
        if len(liked) < 3:
            continue
        held = liked.sample(frac=0.3, random_state=int(rng.integers(1e6)))
        train = ratings.drop(held.index)
        engine = HybridRecommender(movies=movies, ratings=train)
        profile = {id_to_title[m]: r for m, r in
                   train[train["user_id"] == user][["movie_id", "rating"]].values}
        target = {id_to_title[m] for m in held["movie_id"]}
        seen = {engine._index[t.lower()] for t in profile}
        methods = {
            "Popularity": engine.popular(N, exclude=seen),
            "Content only": engine.recommend_for_ratings(profile, N, alpha=1.0),
            "Collaborative only": engine.recommend_for_ratings(profile, N, alpha=0.0),
            "Hybrid": engine.recommend_for_ratings(profile, N, alpha=0.6),
        }
        for name, recs in methods.items():
            hits[name].append(len({r.title for r in recs} & target) / len(target))
    return {k: float(np.mean(v)) for k, v in hits.items()}


if __name__ == "__main__":
    print(f"Mean recall@{N} across users (higher is better)")
    for name, score in run().items():
        print(f"  {name:<20} {score:.3f}")
