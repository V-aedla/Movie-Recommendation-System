# Hybrid Movie Recommender

A Python recommendation system that combines content-based filtering (TF-IDF and
cosine similarity) with item-based collaborative filtering, and adds a chat
interface, a personal profile, mood queries and a proactive "tonight" mode.

## Credits and foundation
- Foundation: the standard content-based movie recommender pattern (TF-IDF vectors
  plus cosine similarity) documented in the scikit-learn user guide and taught in
  introductory recommender-system tutorials. **Replace this line with the exact
  tutorial or repository you used as your starting point, with its URL.**
- Libraries: pandas, NumPy, scikit-learn.
- Data: 44 films with short descriptions written for this project. Ratings are
  synthetic (see `generate_ratings.py`) so the project runs without downloading
  anything. For real data, swap in the MovieLens dataset (Harper & Konstan, 2015).

## What I changed from the foundation
1. Added item-based collaborative filtering and a tunable hybrid (`alpha`).
2. Added a personal profile that learns from likes and dislikes.
3. Added free-text and mood queries mapped into the same TF-IDF space.
4. Added a proactive suggestion that uses time of day.
5. Added explanations, fuzzy title matching, tests and an offline evaluation.

## Setup
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```
python app.py        # command-line chat (the assistant speaks first)
python gui.py        # optional window (needs Tkinter)
python evaluate.py   # recall@10 comparison of the four methods
python -m unittest discover -s tests -t .   # run the tests
```

## Example session
```
you> movies like The Matrix
you> something scary and tense
you> rate Inception 5
you> recommend
you> tonight
```

## Files
| File | Purpose |
|---|---|
| `recommender.py` | Hybrid engine: models, recommendations, explanations |
| `assistant.py` | Chat logic shared by the CLI and GUI |
| `app.py`, `gui.py` | Command-line and Tkinter front ends |
| `evaluate.py` | Offline evaluation (recall@10) |
| `generate_ratings.py` | Builds the synthetic ratings file |
| `data/` | `movies.csv`, `ratings.csv` |
| `tests/` | Unit tests |