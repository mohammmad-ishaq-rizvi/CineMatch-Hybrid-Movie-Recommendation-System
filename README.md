# 🎬 CineMatch: Hybrid Movie Recommendation System

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR-STREAMLIT-APP-URL)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**CineMatch** recommends movies from the TMDB 5000 dataset. Pick one film, or blend several favourites into a taste profile, and it returns varied, well-rated suggestions with a short reason for each one.

### 👉 [Try the live app](YOUR-STREAMLIT-APP-URL)

> If the app is asleep, click **"Yes, get this app back up!"** and wait about 30 seconds.

![CineMatch screenshot](screenshots/app.png)
<!-- Add 1-2 screenshots to a screenshots/ folder, or record a short GIF of the app. -->

---

## Features

- **Because you watched**: pick a movie and get similar ones, each with a "why" line (shared genres, cast, director).
- **Build my taste**: choose up to 8 favourites; CineMatch averages them into one taste profile and recommends from that.
- **Top rated**: browse the best films by genre, release year and minimum votes.
- **Live controls**: adjust the number of results, variety, how much to favour well-rated films, and the release-year range.
- **Posters**: fetched from the TMDB API when a key is set, with a keyless Wikipedia fallback and a placeholder if both fail.

## How it works

CineMatch is a hybrid of three ideas: content-based filtering, a quality prior, and diversity re-ranking.

**1. Content features (TF-IDF).** Each movie becomes a vector built from four separately vectorised blocks, each with its own weight:

| Block | Source | Features | Weight |
|---|---|---|---|
| Plot | overview + tagline (1-2 word n-grams) | 19,000 | 1.00 |
| Genres | prefixed tokens, e.g. `genre_science_fiction` | 20 | 1.40 |
| Keywords | prefixed tokens, e.g. `keyword_dream` | 9,804 | 1.15 |
| People | top 5 cast, directors, writers | 16,546 | 0.75 |

The blocks are stacked into one sparse matrix of **4,803 movies × 45,370 features** (about 0.11% dense) and L2-normalised, so a dot product is cosine similarity. Prefixing tokens stops a name like a director's from being confused with a plot word.

**2. Quality prior.** Raw ratings are unreliable when a film has few votes, so a **Bayesian average** pulls low-vote films toward the dataset mean. It is combined with log-scaled popularity (75% rating, 25% popularity). The final score is:

```
score = (1 - quality_weight) * content_similarity + quality_weight * quality
```

with a default `quality_weight` of 0.15, so similarity stays the main signal.

**3. Diversity (MMR).** The top 200 candidates are re-ranked with **Maximal Marginal Relevance**, which penalises films too similar to ones already chosen. This avoids a list of near-identical sequels.

**Multi-movie taste profiles** are the mean of the selected movies' vectors, which gives simple personalisation without needing any user history.

The full walkthrough, with the reasoning behind each choice, is in the notebook: [`hybrid_movie_recommender_explained.ipynb`](hybrid_movie_recommender_explained.ipynb).

## Tech stack

Python · pandas · NumPy · SciPy (sparse matrices) · scikit-learn (TF-IDF) · Streamlit · joblib · Requests

## Run it locally

```bash
git clone https://github.com/mohammmad-ishaq-rizvi/CineMatch-Hybrid-Movie-Recommendation-System.git
cd CineMatch-Hybrid-Movie-Recommendation-System
pip install -r requirements.txt
streamlit run app.py
```

On first run the app builds its index from the two CSV files and caches it as `hybrid_movie_recommender.joblib` (this file is git-ignored). Later runs load the cache.

**Optional: TMDB posters.** Get a free API key from [themoviedb.org](https://www.themoviedb.org/settings/api) and either enter it as an environment variable:

```bash
export TMDB_API_KEY="your_key"     # Windows PowerShell: $env:TMDB_API_KEY="your_key"
```

or add it to `.streamlit/secrets.toml` (locally) or the app's **Secrets** settings (Streamlit Cloud):

```toml
TMDB_API_KEY = "your_key"
```

Without a key, posters come from Wikipedia where available.

## Project structure

```
├── app.py                                    # Streamlit app + recommender logic
├── hybrid_movie_recommender_explained.ipynb  # Step-by-step build and explanation
├── tmdb_5000_movies.csv                      # Movie metadata
├── tmdb_5000_credits.csv                     # Cast and crew
├── requirements.txt
├── LICENSE
└── README.md
```

## Limitations and future work

- **No collaborative filtering.** The dataset has no per-user ratings, so recommendations are based on movie content, not on what similar users liked.
- **Static catalogue.** It covers the 4,803 films in the dataset, so there are no recent releases.
- **No offline evaluation yet.** Next steps: measure quality with a held-out metric (for example precision@k against genre or user data) and compare against a plain TF-IDF baseline.
- **Ideas:** sentence-embedding features for the plot text, a MovieLens-based collaborative layer, moving the recommender into its own module with unit tests.

## Data and credits

- Movie data: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), originally from [The Movie Database (TMDB)](https://www.themoviedb.org/).
- *This product uses the TMDB API but is not endorsed or certified by TMDB.*

## License

The source code is released under the [MIT License](LICENSE). The dataset and poster images belong to their respective owners and are subject to TMDB's terms.

## Author

**Mohammad Ishaq Rizvi**: [GitHub](https://github.com/mohammmad-ishaq-rizvi)
