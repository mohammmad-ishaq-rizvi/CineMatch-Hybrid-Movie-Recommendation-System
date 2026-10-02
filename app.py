"""CineMatch - hybrid movie recommender (TF-IDF content + Bayesian quality + MMR diversity).

Run:  streamlit run app.py
Needs (same folder or ./data): tmdb_5000_movies.csv, tmdb_5000_credits.csv
Posters: add a free TMDB API key in the sidebar, or in .streamlit/secrets.toml as TMDB_API_KEY = "..."
"""
from __future__ import annotations

import html
import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# ----------------------------------------------------------------------------- style
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=DM+Sans:wght@400;500;700&display=swap');
:root{--bg:#15101c;--surface:#201829;--line:rgba(255,235,210,.09);--ink:#f4ead9;--mute:#a99db3;--red:#d13b52;--brass:#e0a93f;}
.stApp{background:radial-gradient(1200px 500px at 50% -10%,#3a1626 0%,var(--bg) 60%);color:var(--ink);font-family:'DM Sans',sans-serif}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden}
.block-container{padding-top:1.4rem;max-width:1250px}
section[data-testid="stSidebar"]{background:#110d17;border-right:1px solid var(--line)}
h1,h2,h3{font-family:'Fraunces',serif!important;color:var(--ink)!important}
.brand{text-align:center;padding:.6rem 0 0}
.brand h1{font-size:4.2rem;font-weight:800;letter-spacing:-.02em;margin:0;line-height:1}
.brand h1 span{color:var(--red)}
.brand p{color:var(--mute);font-size:1.05rem;margin:.5rem 0 1rem}
.strip{height:14px;margin:0 0 1.4rem;background:repeating-linear-gradient(90deg,var(--brass) 0 12px,transparent 12px 26px);
  -webkit-mask:linear-gradient(#000,#000);opacity:.55;border-radius:3px}
.hero{display:flex;gap:26px;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:22px;margin:.6rem 0 1.6rem}
.hero img{width:190px;border-radius:10px;flex:none;box-shadow:0 12px 30px rgba(0,0,0,.5)}
.hero h2{margin:0;font-size:2.1rem}
.tag{color:var(--brass);font-style:italic;margin:.2rem 0 .7rem}
.meta{color:var(--mute);font-size:.92rem;margin-bottom:.7rem}
.chip{display:inline-block;border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--ink);
  border-radius:99px;padding:2px 10px;font-size:.75rem;margin:0 5px 5px 0}
.facts{color:var(--mute);font-size:.88rem;line-height:1.7}
.facts b{color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:22px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden;position:relative;transition:border-color .2s}
.card:hover{border-color:var(--brass)}
.card img{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;background:#2b2036}
.match{position:absolute;top:8px;right:8px;background:rgba(21,16,28,.88);color:var(--brass);font-weight:700;font-size:.78rem;padding:3px 9px;border-radius:99px}
.cbody{padding:11px 12px 13px}
.ctitle{font-weight:700;font-size:.95rem;line-height:1.25;margin-bottom:3px}
.csub{color:var(--mute);font-size:.8rem;margin-bottom:7px}
.csub .star{color:var(--brass)}
.why{color:var(--mute);font-size:.76rem;margin-top:7px;border-top:1px dashed var(--line);padding-top:7px}
.stTabs [data-baseweb="tab"]{font-family:'DM Sans';font-weight:700;color:var(--mute)}
.stTabs [aria-selected="true"]{color:var(--brass)!important}
.empty{color:var(--mute);text-align:center;padding:2rem}
@media(max-width:700px){.hero{flex-direction:column}.brand h1{font-size:3rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------- config
N_CANDIDATES = 200
DEFAULT_QUALITY_WEIGHT = 0.15
DEFAULT_MMR_DIVERSITY = 0.20  # notebook: DEFAULT_MMR_LAMBDA = 0.80 -> diversity = 1 - 0.80
POSTER_BASE = "https://image.tmdb.org/t/p/w342"


# ----------------------------------------------------------------------------- data + model
def _parse(value) -> list:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _names(value, limit=None) -> list[str]:
    out = [i["name"].strip() for i in _parse(value) if isinstance(i, dict) and i.get("name", "").strip()]
    return out[:limit] if limit else out


def _crew(value) -> tuple[list[str], list[str]]:
    parsed = [i for i in _parse(value) if isinstance(i, dict) and i.get("name")]
    directors = [i["name"] for i in parsed if i.get("job") == "Director"]
    writers = [i["name"] for i in parsed if i.get("job") in {"Writer", "Screenplay", "Story"}]
    return directors, writers


def _tokenise(values, prefix) -> str:
    cleaned = [re.sub(r"[^a-z0-9]+", "_", str(v).lower()).strip("_") for v in values]
    return " ".join(f"{prefix}_{v}" for v in cleaned if v)


def _bayesian(frame: pd.DataFrame, quantile=0.60) -> pd.Series:
    votes = frame["vote_count"].fillna(0).astype(float)
    rating = frame["vote_average"].fillna(0).astype(float)
    m = votes.quantile(quantile)
    mean = rating[votes > 0].mean() if (votes > 0).any() else 5.0
    return (votes / (votes + m)) * rating + (m / (votes + m)) * mean


def find_data_dir() -> Path | None:
    for d in (Path("."), Path("data"), Path(__file__).parent, Path(__file__).parent / "data"):
        has_model = (d / "hybrid_movie_recommender.joblib").exists()
        has_csv = (d / "tmdb_5000_movies.csv").exists() and (d / "tmdb_5000_credits.csv").exists()
        if has_model or has_csv:
            return d
    return None

@st.cache_resource(show_spinner="Building the CineMatch index (first run only)...")
def load_model():
    data_dir = find_data_dir()
    if data_dir is None:
        return None
    cache = data_dir / "hybrid_movie_recommender.joblib"
    if cache.exists():
        try:
            a = joblib.load(cache)
            if {"movies", "matrix", "quality"} <= a.keys() and "cast_list" in a["movies"]:
                m = a["movies"]
                if "companies_list" not in m:
                    m["companies_list"] = m["production_companies"].map(lambda v: _names(v, 2))
                return m, a["matrix"], a["quality"]
        except Exception:
            pass

    movies_raw = pd.read_csv(data_dir / "tmdb_5000_movies.csv")
    credits = pd.read_csv(data_dir / "tmdb_5000_credits.csv").rename(columns={"movie_id": "id", "title": "credit_title"})
    movies = (movies_raw.merge(credits[["id", "cast", "crew"]], on="id", how="left")
              .drop_duplicates("id").reset_index(drop=True))

    movies["year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
    for col in ("overview", "tagline"):
        movies[col] = movies[col].fillna("")
    crew = movies["crew"].map(_crew)
    movies["director_list"] = crew.map(lambda t: t[0])
    movies["writer_list"] = crew.map(lambda t: t[1])
    movies["genres_list"] = movies["genres"].map(_names)
    movies["keywords_list"] = movies["keywords"].map(_names)
    movies["cast_list"] = movies["cast"].map(lambda v: _names(v, 5))
    movies["companies_list"] = movies["production_companies"].map(lambda v: _names(v, 2))

    text_doc = (movies["overview"] + " " + movies["tagline"]).str.strip()
    genre_doc = movies["genres_list"].map(lambda v: _tokenise(v, "genre"))
    kw_doc = movies["keywords_list"].map(lambda v: _tokenise(v, "keyword"))
    people_doc = (movies["cast_list"].map(lambda v: _tokenise(v, "cast")) + " "
                  + movies["director_list"].map(lambda v: _tokenise(v, "director")) + " "
                  + movies["writer_list"].map(lambda v: _tokenise(v, "writer"))).str.strip()

    meta = dict(token_pattern=r"(?u)\b[\w_]+\b", min_df=1, sublinear_tf=True)
    X = hstack([
        TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.90,
                        sublinear_tf=True, max_features=19_000).fit_transform(text_doc) * 1.00,
        TfidfVectorizer(**meta).fit_transform(genre_doc) * 1.40,
        TfidfVectorizer(**meta).fit_transform(kw_doc) * 1.15,
        TfidfVectorizer(**meta).fit_transform(people_doc) * 0.75,
    ], format="csr")
    norm = np.sqrt(X.multiply(X).sum(axis=1)).A1
    X = X.multiply(1 / np.maximum(norm, 1e-12)[:, None]).tocsr()

    movies["bayesian_rating"] = _bayesian(movies)
    pop = np.log1p(movies["popularity"].fillna(0).clip(lower=0)).to_numpy().reshape(-1, 1)
    rat = movies["bayesian_rating"].to_numpy().reshape(-1, 1)
    quality = 0.75 * MinMaxScaler().fit_transform(rat).ravel() + 0.25 * MinMaxScaler().fit_transform(pop).ravel()

    try:
        joblib.dump({"movies": movies, "matrix": X, "quality": quality}, cache, compress=3)
    except OSError:
        pass
    return movies, X, quality


# ----------------------------------------------------------------------------- recommender
def mmr_select(X, cands: np.ndarray, scores: np.ndarray, n: int, diversity: float) -> list[int]:
    """MMR with one candidate-vs-candidate similarity matrix computed up front (fast)."""
    Xc = X[cands]
    sim = (Xc @ Xc.T).toarray()
    remaining, selected = list(range(len(cands))), []
    while remaining and len(selected) < n:
        if not selected:
            pick = max(remaining, key=lambda i: scores[i])
        else:
            pick = max(remaining, key=lambda i: (1 - diversity) * scores[i] - diversity * sim[i, selected].max())
        selected.append(pick)
        remaining.remove(pick)
    return [int(cands[i]) for i in selected]


def explain(movies, sources: list[int], cand: int, X) -> str:
    g, c, d = set(), set(), set()
    for s in sources:
        g |= set(movies.at[s, "genres_list"])
        c |= set(movies.at[s, "cast_list"])
        d |= set(movies.at[s, "director_list"])
    reasons = []
    if len(sources) > 1:
        sims = (X[cand] @ X[sources].T).toarray().ravel()
        reasons.append("closest to " + movies.at[sources[int(sims.argmax())], "title"])
    sg = sorted(g & set(movies.at[cand, "genres_list"]))
    sc = sorted(c & set(movies.at[cand, "cast_list"]))
    sd = sorted(d & set(movies.at[cand, "director_list"]))
    if sg: reasons.append("genres: " + ", ".join(sg[:2]))
    if sc: reasons.append("cast: " + ", ".join(sc[:2]))
    if sd: reasons.append("director: " + ", ".join(sd))
    return "; ".join(reasons) if reasons else "similar plot & themes"


def recommend(model, sources: list[int], n, diversity, quality_weight, years) -> pd.DataFrame:
    movies, X, quality = model
    profile = csr_matrix(X[sources].mean(axis=0))
    content = (profile @ X.T).toarray().ravel()
    final = (1 - quality_weight) * content + quality_weight * quality

    year = movies["year"].to_numpy()
    final[~((year >= years[0]) & (year <= years[1]))] = -np.inf  # filter BEFORE picking candidates
    final[sources] = -np.inf
    k = min(N_CANDIDATES, int(np.isfinite(final).sum()))
    if k == 0:
        return movies.iloc[0:0]
    cands = np.argpartition(final, -k)[-k:]
    chosen = mmr_select(X, cands, final[cands], min(n, k), diversity)

    out = movies.loc[chosen].copy()
    out["match"] = content[chosen]
    out["why"] = [explain(movies, sources, i, X) for i in chosen]
    return out


# ----------------------------------------------------------------------------- posters
TMDB_API_KEY = ""  # set once


def get_api_key() -> str:
    try:
        key = st.secrets.get("TMDB_API_KEY", "")
    except Exception:
        key = ""
    return (key or os.environ.get("TMDB_API_KEY", "") or TMDB_API_KEY).strip()

def tmdb_poster(mid: int, api_key: str) -> str:
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key.startswith("eyJ") else {}
        params = {} if headers else {"api_key": api_key}
        r = requests.get(f"https://api.themoviedb.org/3/movie/{mid}", params=params, headers=headers, timeout=6)
        path = r.json().get("poster_path") if r.ok else None
        return f"{POSTER_BASE}{path}" if path else ""
    except Exception:
        return ""


def wiki_poster(title: str, year) -> str:
    """Keyless fallback: first Wikipedia search hit for '<title> <year> film'."""
    try:
        y = int(year) if pd.notna(year) else ""
        r = requests.get("https://en.wikipedia.org/w/api.php", timeout=6,
                         headers={"User-Agent": "CineMatch/1.0"},
                         params={"action": "query", "format": "json", "generator": "search",
                                 "gsrsearch": f"{title} {y} film", "gsrlimit": 1,
                                 "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": 342})
        for page in r.json().get("query", {}).get("pages", {}).values():
            return page.get("thumbnail", {}).get("source", "")
    except Exception:
        pass
    return ""


@st.cache_resource
def poster_store() -> dict[int, str]:
    return {}  # in-memory cache that only keeps successful lookups


def fetch_posters(ids: tuple[int, ...], api_key: str) -> dict[int, str]:
    store = poster_store()
    meta = movies.drop_duplicates("id").set_index("id")[["title", "year"]]
    missing = [i for i in ids if i not in store]

    def one(mid: int) -> tuple[int, str]:
        url = tmdb_poster(mid, api_key) if api_key else ""
        return mid, url or wiki_poster(meta.at[mid, "title"], meta.at[mid, "year"])

    if missing:
        with ThreadPoolExecutor(max_workers=12) as pool:
            for mid, url in pool.map(one, missing):
                if url:
                    store[mid] = url  # failures are never stored, so they retry next time
    return {i: store.get(i, "") for i in ids}


def placeholder(title: str) -> str:
    t = html.escape(title[:34])
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='342' height='513'><rect width='100%' height='100%' fill='#2b2036'/>"
           f"<text x='50%' y='48%' fill='#e0a93f' font-family='Georgia' font-size='22' text-anchor='middle'>🎬</text>"
           f"<text x='50%' y='56%' fill='#f4ead9' font-family='Georgia' font-size='18' text-anchor='middle'>{t}</text></svg>")
    return "data:image/svg+xml;utf8," + quote(svg)


# ----------------------------------------------------------------------------- rendering
def chips(items) -> str:
    return "".join(f"<span class='chip'>{html.escape(i)}</span>" for i in items)


def money(v) -> str:
    return f"${v / 1e6:,.0f}M" if v and v > 0 else "n/a"


def render_hero(movies, idx: int, poster: str):
    m = movies.loc[idx]
    mins = int(m["runtime"]) if pd.notna(m["runtime"]) and m["runtime"] else None
    meta = " • ".join(x for x in [str(int(m["year"])) if pd.notna(m["year"]) else "", f"{mins} min" if mins else "",
                                  str(m["original_language"]).upper(), f"★ {m['vote_average']:.1f} ({int(m['vote_count']):,} votes)"] if x)
    tagline = f"<div class='tag'>{html.escape(m['tagline'])}</div>" if m["tagline"] else ""
    st.markdown(f"""
    <div class='hero'><img src='{poster or placeholder(m['title'])}' alt='poster'>
      <div><h2>{html.escape(m['title'])}</h2>{tagline}<div class='meta'>{html.escape(meta)}</div>
        {chips(m['genres_list'])}
        <p style='margin:.6rem 0'>{html.escape(m['overview'])}</p>
        <div class='facts'><b>Director</b> {html.escape(', '.join(m['director_list'][:2]) or 'n/a')}<br>
        <b>Cast</b> {html.escape(', '.join(m['cast_list']) or 'n/a')}<br>
        <b>Budget</b> {money(m['budget'])} &nbsp; <b>Revenue</b> {money(m['revenue'])}<br>
        <b>Studio</b> {html.escape(', '.join(m['companies_list']) or 'n/a')}<br>
        <b>Themes</b> {html.escape(', '.join(m['keywords_list'][:6]) or 'n/a')}</div></div></div>""", unsafe_allow_html=True)


def render_grid(frame: pd.DataFrame, posters: dict[int, str], show_match=True):
    if frame.empty:
        st.markdown("<div class='empty'>No movies match these filters. Widen the year range.</div>", unsafe_allow_html=True)
        return
    cards = []
    for _, m in frame.iterrows():
        badge = f"<div class='match'>{m['match'] * 100:.0f}% match</div>" if show_match and "match" in m else ""
        why = f"<div class='why'>{html.escape(m['why'])}</div>" if "why" in m else ""
        yr = str(int(m["year"])) if pd.notna(m["year"]) else "n/a"
        cards.append(f"""<div class='card'>{badge}<img loading='lazy' src='{posters.get(int(m['id'])) or placeholder(m['title'])}' alt=''>
          <div class='cbody'><div class='ctitle'>{html.escape(m['title'])}</div>
          <div class='csub'>{yr} • <span class='star'>★ {m['vote_average']:.1f}</span></div>
          {chips(m['genres_list'][:3])}{why}</div></div>""")
    st.markdown(f"<div class='grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- app
st.markdown("<div class='brand'><h1>Cine<span>Match</span></h1>"
            "<p>Pick films you love. We find what to watch next.</p></div><div class='strip'></div>", unsafe_allow_html=True)

model = load_model()
if model is None:
    st.error("Dataset not found. Put `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` next to `app.py` (or in a `data/` folder), then refresh.")
    st.stop()
movies, X, quality = model

api_key = get_api_key()

with st.sidebar:
    st.header("Settings")
    n_results = st.slider("Recommendations", 4, 24, 12, 4)
    diversity = st.slider("Variety", 0.0, 0.6, DEFAULT_MMR_DIVERSITY, 0.05, help="Higher = less repetitive results (MMR).")
    q_weight = st.slider("Favour well-rated films", 0.0, 0.5, DEFAULT_QUALITY_WEIGHT, 0.05, help="Weight of the Bayesian rating + popularity score.")
    ymin, ymax = int(movies["year"].min()), int(movies["year"].max())
    years = st.slider("Release years", ymin, ymax, (ymin, ymax))
    st.caption(f"{len(movies):,} films indexed • {X.shape[1]:,} features")

order = movies.sort_values("vote_count", ascending=False).index.tolist()
labels = {i: f"{movies.at[i, 'title']} ({int(movies.at[i, 'year']) if pd.notna(movies.at[i, 'year']) else '?'})" for i in order}

tab1, tab2, tab3 = st.tabs(["Because you watched", "Build my taste", "Top rated"])

with tab1:
    default = next((i for i in order if movies.at[i, "title"] == "The Dark Knight"), order[0])
    pick = st.selectbox("Search a movie", order, index=order.index(default), format_func=labels.get, key="single")
    recs = recommend(model, [pick], n_results, diversity, q_weight, years)
    ids = tuple([int(movies.at[pick, "id"])] + recs["id"].astype(int).tolist())
    posters = fetch_posters(ids, api_key)
    render_hero(movies, pick, posters.get(int(movies.at[pick, "id"]), ""))
    st.subheader("You might also like")
    render_grid(recs, posters)

with tab2:
    favs = st.multiselect("Choose 2 to 8 films you love", order, default=[i for i in order if movies.at[i, "title"] in ("Inception", "Interstellar", "The Dark Knight")][:3],
                          format_func=labels.get, max_selections=8, key="multi")
    if len(favs) < 1:
        st.markdown("<div class='empty'>Add a few favourites and CineMatch blends them into one taste profile.</div>", unsafe_allow_html=True)
    else:
        recs = recommend(model, favs, n_results, diversity, q_weight, years)
        posters = fetch_posters(tuple(recs["id"].astype(int)), api_key)
        st.subheader("Made for your taste")
        render_grid(recs, posters)

with tab3:
    all_genres = sorted({g for gl in movies["genres_list"] for g in gl})
    c1, c2 = st.columns([2, 1])
    genre = c1.selectbox("Genre", ["All genres"] + all_genres)
    min_votes = c2.slider("Minimum votes", 0, 3000, 500, 100)
    pool = movies[(movies["vote_count"] >= min_votes) & movies["year"].between(*years)]
    if genre != "All genres":
        pool = pool[pool["genres_list"].map(lambda g: genre in g)]
    top = pool.assign(_q=quality[pool.index]).sort_values("_q", ascending=False).head(n_results)
    posters = fetch_posters(tuple(top["id"].astype(int)), api_key)
    render_grid(top, posters, show_match=False)