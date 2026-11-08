"""Generate a small, reproducible, SYNTHETIC ratings file for demonstration.

Real systems use logged user behaviour (for example the MovieLens datasets).
Here, 36 simulated users in six taste groups rate roughly 60% of the catalogue.
A rating depends on how much a film's genres overlap with the group's tastes,
plus random noise, so collaborative filtering has a real signal to learn.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

TASTE_GROUPS = {
    "sci_fi_fan": {"Sci-Fi", "Thriller", "Action"},
    "crime_buff": {"Crime", "Mystery", "Thriller", "Drama"},
    "romcom_lover": {"Romance", "Comedy"},
    "family_animation": {"Animation", "Family", "Fantasy", "Adventure"},
    "horror_fan": {"Horror", "Thriller", "Mystery"},
    "history_drama": {"War", "Drama", "Documentary", "Sports"},
}


def main(seed: int = 42, users_per_group: int = 6, coverage: float = 0.6) -> None:
    rng = np.random.default_rng(seed)
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    rows, user_id = [], 1
    for liked in TASTE_GROUPS.values():
        for _ in range(users_per_group):
            for _, movie in movies.iterrows():
                if rng.random() > coverage:
                    continue
                genres = set(movie["genres"].split("|"))
                overlap = len(genres & liked) / len(genres)
                rating = 2.0 + 2.4 * overlap + rng.normal(0, 0.7)
                rows.append((user_id, int(movie["movie_id"]),
                             int(np.clip(round(rating), 1, 5))))
            user_id += 1
    out = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])
    out.to_csv(DATA_DIR / "ratings.csv", index=False)
    print(f"Wrote {len(out)} ratings from {user_id - 1} users.")


if __name__ == "__main__":
    main()
