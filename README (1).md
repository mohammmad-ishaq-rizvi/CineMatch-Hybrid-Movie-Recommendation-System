# 🎬 CineMatch: Hybrid Movie Recommendation System

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR-STREAMLIT-APP-URL)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

CineMatch recommends movies from the **TMDB 5000** dataset. Pick one film, or blend several favourites into a taste profile, and get relevant, varied, well-rated suggestions, each with a short reason.

### 👉 [Try the live app](YOUR-STREAMLIT-APP-URL)
*If the app is asleep, click "Yes, get this app back up!" and wait about 30 seconds.*

![CineMatch screenshot](screenshots/app.png)

---

## What it does

| Mode | Description |
|---|---|
| **Because you watched** | Choose a movie and get similar ones, with a "why" line (shared genres, cast, director). |
| **Build my taste** | Choose up to 8 favourites; they are blended into one taste profile. |
| **Top rated** | Browse the best films by genre, release year and minimum votes. |

Sidebar controls: number of results, **variety** (diversity), **how much to favour well-rated films**, and release-year range. Posters come from the TMDB API (optional key) with a Wikipedia fallback.

## How it works

CineMatch is a hybrid of **content-based filtering**, a **quality prior** and **diversity re-ranking**.

**1. Content similarity (TF-IDF).** Each movie is turned into a vector from four separately vectorised blocks:

| Block | Source | Features | Weight |
|---|---|---|---|
| Plot | overview + tagline, unigrams and bigrams | 19,000 | 1.00 |
| Genres | prefixed tokens, e.g. `genre_science_fiction` | 20 | 1.40 |
| Keywords | prefixed tokens, e.g. `keyword_dream` | 9,804 | 1.15 |
| People | top-5 cast, directors, writers | 16,546 | 0.75 |

The blocks are stacked into a sparse **4,803 × 45,370** matrix (about 0.11% dense) and L2-normalised, so a dot product equals cosine similarity. Token prefixes keep a director's name from being mistaken for a plot word.

**2. Quality prior.** A **Bayesian average** pulls ratings of films with few votes toward the global mean, and is blended with log-scaled popularity (75% rating, 25% popularity). Final score:

```
score = (1 - quality_weight) * content_similarity + quality_weight * quality     # default quality_weight = 0.15
```

Similarity stays the main signal; quality only breaks ties in favour of reliable films.

**3. Diversity (MMR).** The top 200 candidates are re-ranked with **Maximal Marginal Relevance**, which penalises films that are too similar to ones already chosen, so you don't get ten near-identical sequels.

**4. Taste profiles.** For several favourites, their vectors are averaged into one profile, which gives simple personalisation with no user history.

## The notebook

[`hybrid_movie_recommender_explained.ipynb`](hybrid_movie_recommender_explained.ipynb) builds the whole system step by step:

| Stage | What happens |
|---|---|
| Data loading | Merge movies (4,803 × 20) with credits using a validated one-to-one join on TMDB ID. |
| Validation | Assertions for duplicate IDs, missing credits and score ranges; JSON-parse failures and missing years are counted, not hidden. |
| Feature extraction | Parse nested JSON into genres, keywords, cast, directors and writers. |
| Tokenisation | Lower-case, underscore-joined, prefixed tokens per feature type. |
| Vectorisation | Four weighted TF-IDF blocks, stacked and L2-normalised. |
| Rating | Bayesian-average quality score plus popularity. |
| Pipeline | Title lookup (handles remakes and suggests close matches), `recommend()`, MMR re-ranking and explanations. Each query scores one movie against all others, so no all-pairs similarity matrix is stored. |
| Personalisation | `recommend_from_likes()` builds a taste profile from several titles. |
| Persistence | Index, scores and config saved with `joblib`. |

**Sample output from the notebook**

- *The Avengers* → Avengers: Age of Ultron, Captain America: The Winter Soldier, Ant-Man, Captain America: Civil War
- *The Dark Knight + Inception + Interstellar* → The Dark Knight Rises, Mad Max: Fury Road, Guardians of the Galaxy, The Martian

## Run it locally

```bash
git clone https://github.com/mohammmad-ishaq-rizvi/CineMatch-Hybrid-Movie-Recommendation-System.git
cd CineMatch-Hybrid-Movie-Recommendation-System
pip install -r requirements.txt
streamlit run app.py
```

On first run the app builds its index from the two CSVs and caches it as `hybrid_movie_recommender.joblib` (git-ignored); later runs load the cache.

**Optional: posters via TMDB.** Get a free key at [themoviedb.org](https://www.themoviedb.org/settings/api), then set `TMDB_API_KEY` as an environment variable, in `.streamlit/secrets.toml`, or in Streamlit Cloud's **Secrets**:

```toml
TMDB_API_KEY = "your_key"
```

## Project structure

```
├── app.py                                    # Streamlit app + recommender logic
├── hybrid_movie_recommender_explained.ipynb  # Step-by-step build and explanation
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
├── requirements.txt
├── LICENSE
└── README.md
```

## Tech stack

Python · pandas · NumPy · SciPy · scikit-learn · Streamlit · joblib · Requests

## Limitations and next steps

- **No collaborative filtering:** the dataset has no per-user ratings, so recommendations come from movie content, not from similar users.
- **Static catalogue:** only the 4,803 films in the dataset, with no recent releases.
- **No offline evaluation yet:** next steps are precision@k against a baseline, sentence-embedding features for plots, and moving the recommender into its own tested module.

## Data and credits

Data from the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), originally from [TMDB](https://www.themoviedb.org/). *This product uses the TMDB API but is not endorsed or certified by TMDB.*

## License

Code is released under the [MIT License](LICENSE). The dataset and posters belong to their owners and are subject to TMDB's terms.

## Author

**Mohammad Ishaq Rizvi**: [GitHub](https://github.com/mohammmad-ishaq-rizvi) · [LinkedIn](https://www.linkedin.com/in/mohammad-ishaq-rizvi/)
