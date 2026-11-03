from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import time
import requests
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from auth import (
    register_user,
    login_user,
    logout_user,
    save_movie_rating,
    save_movie_review,
)
from db_config import get_connection

st.set_page_config(
    page_title="CineScope | Movie Discovery",
    page_icon="🎬",
    layout="wide",
)

DATA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DATA_DIR.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
POSTER_CACHE_FILE = DATA_DIR / "poster_cache.json"


@st.cache_data(show_spinner=False)
def load_data():
    movies_path = DATA_DIR / "movies.csv"
    ratings_path = DATA_DIR / "ratings.csv"
    if not movies_path.is_file() or not ratings_path.is_file():
        raise FileNotFoundError(
            "Keep movies.csv and ratings.csv inside the datasets folder."
        )

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    required_movies = {"movieId", "title", "genres"}
    required_ratings = {"userId", "movieId", "rating"}
    if not required_movies.issubset(movies.columns):
        raise ValueError("movies.csv must contain movieId, title, and genres.")
    if not required_ratings.issubset(ratings.columns):
        raise ValueError("ratings.csv must contain userId, movieId, and rating.")

    stats = (
        ratings.groupby("movieId")
        .agg(
            average_rating=("rating", "mean"),
            total_ratings=("rating", "count"),
        )
        .reset_index()
    )
    data = movies.merge(stats, on="movieId", how="left")
    data["average_rating"] = data["average_rating"].fillna(0).round(2)
    data["total_ratings"] = data["total_ratings"].fillna(0).astype(int)
    return movies, ratings, data


def normalise_name(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


@st.cache_data(show_spinner=False)
def local_poster_index():
    """Index local poster filenames once, rather than scanning folders per movie."""
    index = {}
    folders = (
        DATA_DIR / "pictures",
        DATA_DIR / "posters",
        DATA_DIR / "images",
        PROJECT_DIR / "pictures",
        PROJECT_DIR / "posters",
        PROJECT_DIR / "images",
    )
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.stem.lower(), str(path))
                index.setdefault(normalise_name(path.stem), str(path))
    return index


def local_poster(movie_id, title):
    index = local_poster_index()
    movie_id = str(int(movie_id))
    title = str(title).strip()
    title_no_year = title
    if title.endswith(")") and "(" in title:
        possible_title, _, year_part = title.rpartition("(")
        year = year_part[:-1].strip()
        if len(year) == 4 and year.isdigit():
            title_no_year = possible_title.strip()

    candidates = [movie_id, title, title_no_year]
    for candidate in candidates:
        path = index.get(candidate.lower()) or index.get(normalise_name(candidate))
        if path:
            return path
    return None


def read_poster_cache():
    """Read saved TMDB poster URLs so page reruns do not repeat API requests."""
    try:
        if POSTER_CACHE_FILE.is_file():
            with POSTER_CACHE_FILE.open("r", encoding="utf-8") as file:
                value = json.load(file)
                return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def write_poster_cache(cache):
    """Write poster cache atomically to avoid a partially-written JSON file."""
    temporary = POSTER_CACHE_FILE.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(cache, file, ensure_ascii=False, indent=2)
        temporary.replace(POSTER_CACHE_FILE)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def tmdb_poster_url(title, api_key):
    """Look up one TMDB poster; called only after the user requests poster loading."""
    if not api_key:
        return None

    clean_title = str(title).strip()
    query = clean_title
    year = None
    if clean_title.endswith(")") and "(" in clean_title:
        before, _, after = clean_title.rpartition("(")
        candidate_year = after[:-1].strip()
        if len(candidate_year) == 4 and candidate_year.isdigit():
            query = before.strip()
            year = candidate_year

    params = {"api_key": api_key, "query": query, "include_adult": "false"}
    if year:
        params["year"] = year
    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=(1.5, 2.5),
        )
        response.raise_for_status()
        for result in response.json().get("results", []):
            poster_path = result.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w342{poster_path}"
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return None
    return None


def get_cached_poster_sources(rows, poster_cache):
    """Return local or previously cached posters without making network calls."""
    sources = {}
    for movie_id, title in rows:
        movie_id = int(movie_id)
        local = local_poster(movie_id, title)
        if local:
            sources[movie_id] = local
            continue
        entry = poster_cache.get(str(movie_id), {})
        if isinstance(entry, dict) and entry.get("url"):
            sources[movie_id] = entry["url"]
    return sources


def fetch_and_cache_posters(rows, poster_cache):
    """Fetch only requested missing posters concurrently, then persist the results."""
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        return poster_cache, 0, len(rows)

    now = time.time()
    work = []
    skipped = 0
    for movie_id, title in rows:
        movie_id = int(movie_id)
        if local_poster(movie_id, title):
            skipped += 1
            continue
        entry = poster_cache.get(str(movie_id), {})
        checked_at = entry.get("checked_at", 0) if isinstance(entry, dict) else 0
        poster_url = entry.get("url") if isinstance(entry, dict) else None
        # Recheck negative results after one day; successful URLs after 30 days.
        ttl = 86400 if not poster_url else 30 * 86400
        if checked_at and now - float(checked_at) < ttl:
            skipped += 1
            continue
        work.append((movie_id, str(title)))

    if not work:
        return poster_cache, 0, skipped

    from concurrent.futures import ThreadPoolExecutor, as_completed
    fetched = 0
    with ThreadPoolExecutor(max_workers=min(4, len(work))) as executor:
        futures = {
            executor.submit(tmdb_poster_url, title, api_key): movie_id
            for movie_id, title in work
        }
        for future in as_completed(futures):
            movie_id = futures[future]
            try:
                url = future.result()
            except Exception:
                url = None
            poster_cache[str(movie_id)] = {"url": url, "checked_at": time.time()}
            if url:
                fetched += 1

    write_poster_cache(poster_cache)
    return poster_cache, fetched, len(work) - fetched


@st.cache_data(show_spinner=False)
def make_placeholder_poster(title, genres=""):
    """Create a branded, title-based fallback image when no real poster exists."""
    width, height = 480, 720
    image = Image.new("RGB", (width, height), (14, 20, 39))
    # Draw the gradient as horizontal bands (faster than setting every pixel).
    gradient_draw = ImageDraw.Draw(image)
    top = (15, 25, 48)
    bottom = (57, 39, 103)
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = tuple(
            int(top[i] * (1 - ratio) + bottom[i] * ratio)
            for i in range(3)
        )
        gradient_draw.line((0, y, width, y), fill=color)

    draw = ImageDraw.Draw(image, "RGBA")
    draw.ellipse((270, 90, 570, 390), fill=(112, 92, 255, 42))
    draw.ellipse((-130, 390, 250, 770), fill=(35, 205, 190, 30))
    draw.rounded_rectangle((22, 22, width - 22, height - 22), radius=26,
                           outline=(205, 195, 255, 105), width=3)
    draw.rounded_rectangle((42, 42, width - 42, height - 42), radius=20,
                           outline=(255, 255, 255, 30), width=1)

    def get_font(size, bold=False):
        candidates = (
            [r"C:\Windows\Fonts\arialbd.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"]
            if bold else
            [r"C:\Windows\Fonts\arial.ttf", "DejaVuSans.ttf", "arial.ttf"]
        )
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    font_small = get_font(19, bold=True)
    font_title = get_font(43, bold=True)
    font_genre = get_font(19, bold=False)
    font_footer = get_font(15, bold=True)

    draw.text((width // 2, 78), "CINESCOPE", font=font_small,
              fill=(197, 187, 255, 255), anchor="mm")
    draw.line((95, 112, width - 95, 112), fill=(197, 187, 255, 130), width=2)

    cleaned_title = str(title).strip()
    year = ""
    if cleaned_title.endswith(")") and "(" in cleaned_title:
        before, _, after = cleaned_title.rpartition("(")
        possible_year = after[:-1].strip()
        if len(possible_year) == 4 and possible_year.isdigit():
            cleaned_title = before.strip()
            year = possible_year

    # Wrap title to fit the poster width.
    words = cleaned_title.upper().split()
    lines = []
    current = ""
    max_text_width = width - 100
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font_title)[2] > max_text_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if not lines:
        lines = ["UNTITLED MOVIE"]

    # Shrink title font for unusually long titles.
    while len(lines) > 5 and font_title.size > 27:
        font_title = get_font(font_title.size - 3, bold=True)
        lines = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and draw.textbbox((0, 0), candidate, font=font_title)[2] > max_text_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)

    y_start = 270 - (len(lines) * 29)
    for line in lines:
        draw.text((width // 2, y_start), line, font=font_title,
                  fill=(255, 255, 255, 255), anchor="mm", stroke_width=1,
                  stroke_fill=(20, 18, 45, 180))
        y_start += 66

    if year:
        draw.text((width // 2, y_start + 18), year, font=font_genre,
                  fill=(218, 212, 255, 255), anchor="mm")

    genre_text = "  •  ".join(
        part.strip() for part in str(genres).split("|")
        if part.strip() and part.strip() != "(no genres listed)"
    )
    if genre_text:
        if len(genre_text) > 34:
            genre_text = genre_text[:31].rstrip() + "..."
        draw.text((width // 2, height - 145), genre_text, font=font_genre,
                  fill=(205, 215, 241, 255), anchor="mm")

    draw.rounded_rectangle((85, height - 112, width - 85, height - 68),
                           radius=18, fill=(255, 255, 255, 24),
                           outline=(220, 215, 255, 100), width=1)
    draw.text((width // 2, height - 90), "ORIGINAL POSTER UNAVAILABLE",
              font=font_footer, fill=(225, 221, 255, 255), anchor="mm")
    return image


try:
    movies, ratings, data = load_data()
except Exception as exc:
    st.error(f"Could not load movie data: {exc}")
    st.stop()

# This small JSON file keeps fetched poster URLs between app restarts.
poster_cache = read_poster_cache()

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: radial-gradient(ellipse at 8% 0%,rgba(119,83,255,.20),transparent 34%), radial-gradient(ellipse at 90% 12%,rgba(32,195,180,.10),transparent 30%), linear-gradient(145deg,#090d19,#10172a 55%,#0b1120); font-family:Arial, sans-serif; color:#f5f7ff; }
    h1,h2,h3 { font-family:Arial, sans-serif!important; letter-spacing:-.035em; color:#f7f8ff!important; }
    h1 { font-size:2.6rem!important; }
    p,label,.stCaption { color:#b9c4dc; }
    [data-testid="stSidebar"] { background:rgba(9,14,27,.96); border-right:1px solid rgba(255,255,255,.09); }
    [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(31,41,68,.96),rgba(19,27,47,.96)); border:1px solid rgba(255,255,255,.09); padding:18px 20px; border-radius:18px; }
    [data-testid="stMetricLabel"] { color:#aebbd6!important; }
    [data-testid="stMetricValue"] { color:#fff!important; font-weight:700; }
    .stButton>button { border:1px solid rgba(255,255,255,.14); border-radius:11px; padding:.62rem 1rem; font-weight:650; }
    .stButton>button[kind="primary"] { background:linear-gradient(100deg,#7865ff,#4e8cff); color:#fff; border:0; }
    .stTextInput input,.stTextArea textarea,.stSelectbox [data-baseweb="select"] { border-radius:10px!important; }
    hr { border-color:rgba(255,255,255,.12)!important; }
    .hero { padding:1.7rem 1.9rem; border-radius:22px; border:1px solid rgba(255,255,255,.12); background:linear-gradient(115deg,rgba(110,82,255,.19),rgba(22,31,54,.9) 48%,rgba(24,137,139,.12)); margin:.4rem 0 1.3rem 0; }
    .kicker { color:#a99dff; text-transform:uppercase; letter-spacing:.17em; font-size:.76rem; font-weight:700; }
    .muted { color:#aebbd4; line-height:1.7; }
    .movie-card { background:linear-gradient(145deg,rgba(27,36,61,.95),rgba(16,23,41,.95)); border:1px solid rgba(255,255,255,.09); padding:1rem 1.05rem; border-radius:16px; margin-bottom:.65rem; min-height:140px; }
    .pill { display:inline-block; background:rgba(123,101,255,.16); color:#c9c0ff; border:1px solid rgba(153,136,255,.2); border-radius:20px; padding:3px 9px; font-size:.75rem; margin:2px 4px 2px 0; }
    .auth-panel { background:linear-gradient(145deg,rgba(25,34,57,.98),rgba(13,20,37,.98)); border:1px solid rgba(255,255,255,.12); padding:1.5rem; border-radius:22px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Login / signup gate
if "user_id" not in st.session_state:
    st.markdown(
        '<div class="hero"><div class="kicker">Your personal movie space</div><h1>🎬 CineScope</h1><p class="muted">Discover your next favourite. Explore ratings, find similar films, and share your opinion.</p></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("## Movies worth talking about")
        st.markdown('<div class="movie-card"><h3>⭐ Find the favourites</h3><p class="muted">Explore highly rated and popular movies.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="movie-card"><h3>🍿 Discover something new</h3><p class="muted">Get recommendations based on shared genres.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="movie-card"><h3>💬 Share your take</h3><p class="muted">Rate a movie and leave a review from your account.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)
        login_tab, signup_tab = st.tabs(["Log in", "Create account"])
        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email address")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Log in to CineScope", type="primary", width="stretch")
            if submit:
                try:
                    if login_user(email, password):
                        st.rerun()
                    else:
                        st.error("Email or password is incorrect.")
                except Exception as exc:
                    st.error("Could not connect to the database. Check DB_PASSWORD in this terminal.")
                    st.caption(f"Technical detail: {exc}")
        with signup_tab:
            with st.form("signup_form"):
                name = st.text_input("Your name")
                new_email = st.text_input("Email address", key="signup_email")
                new_password = st.text_input("Password (at least 8 characters)", type="password", key="signup_password")
                signup = st.form_submit_button("Create my account", type="primary", width="stretch")
            if signup:
                if not name.strip() or not new_email.strip() or not new_password:
                    st.warning("Please complete all fields.")
                elif len(new_password) < 8:
                    st.warning("Choose a password with at least 8 characters.")
                elif "@" not in new_email or "." not in new_email:
                    st.warning("Please enter a valid email address.")
                else:
                    try:
                        success, message = register_user(name, new_email, new_password)
                        st.success(message) if success else st.error(message)
                    except Exception as exc:
                        st.error("Could not connect to the database. Check DB_PASSWORD in this terminal.")
                        st.caption(f"Technical detail: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Sidebar
st.sidebar.markdown("# 🎬 CineScope")
st.sidebar.caption(f"Welcome, {st.session_state.get('display_name', 'Movie fan')}")
page = st.sidebar.radio("Explore", ["Discover", "Find Similar Movies", "Ratings & Reviews", "Ratings Analysis"])
if st.sidebar.button("Log out", width="stretch"):
    logout_user()
    st.rerun()
st.sidebar.divider()
st.sidebar.caption("MOVIE DISCOVERY · POWERED BY MOVIELENS")

st.markdown(
    '<div class="hero"><div class="kicker">Movie discovery, made personal</div><h1>Find your next favourite film.</h1><p class="muted">Browse community ratings, discover genre matches, and share your own reviews — all in one place.</p></div>',
    unsafe_allow_html=True,
)

if page == "Discover":
    search_col, genre_col, sort_col = st.columns([1.6, 1, 1])
    with search_col:
        search_text = st.text_input("Search movies", placeholder="Search by movie title…", label_visibility="collapsed")
    genre_values = sorted({g for values in movies["genres"].fillna("").astype(str) for g in values.split("|") if g and g != "(no genres listed)"})
    with genre_col:
        genre_filter = st.selectbox("Genre", ["All genres"] + genre_values, label_visibility="collapsed")
    with sort_col:
        sort_mode = st.selectbox("Sort movies", ["Top rated", "Most popular", "Title A–Z"], label_visibility="collapsed")

    filtered = data.copy()
    if search_text.strip():
        filtered = filtered[filtered["title"].str.contains(search_text.strip(), case=False, na=False, regex=False)]
    if genre_filter != "All genres":
        filtered = filtered[filtered["genres"].fillna("").str.contains(genre_filter, regex=False)]
    if sort_mode == "Top rated":
        filtered = filtered.sort_values(["average_rating", "total_ratings"], ascending=False)
    elif sort_mode == "Most popular":
        filtered = filtered.sort_values(["total_ratings", "average_rating"], ascending=False)
    else:
        filtered = filtered.sort_values("title")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies", f"{len(movies):,}")
    c2.metric("MovieLens ratings", f"{len(ratings):,}")
    c3.metric("MovieLens users", f"{ratings['userId'].nunique():,}")
    c4.metric("Genres", f"{len(genre_values)}")

    st.subheader("🎞️ Movie collection")
    st.caption(f"Showing {min(9, len(filtered)):,} of {len(filtered):,} matching movies")
    visible = filtered.head(9)
    if visible.empty:
        st.info("No movies match those filters. Try another title or genre.")
    else:
        poster_rows = [(int(row.movieId), str(row.title)) for row in visible.itertuples()]
        poster_sources = get_cached_poster_sources(poster_rows, poster_cache)
        missing_rows = [item for item in poster_rows if item[0] not in poster_sources and not (isinstance(poster_cache.get(str(item[0])), dict) and poster_cache[str(item[0])].get("checked_at") and time.time() - float(poster_cache[str(item[0])]["checked_at"]) < 86400)]
        if missing_rows and os.getenv("TMDB_API_KEY", "").strip():
            if st.button(f"Load real posters for these {len(missing_rows)} movies", key="load_discover_posters"):
                with st.spinner("Fetching posters once and saving them for future visits..."):
                    poster_cache, found, not_found = fetch_and_cache_posters(missing_rows, poster_cache)
                st.success(f"Fetched {found} posters. {not_found} movies do not currently have a TMDB poster.")
                st.rerun()
        elif missing_rows:
            st.caption("Add your TMDB API key to load real posters. Title-based CineScope posters are shown meanwhile.")
        for start in range(0, len(visible), 3):
            cols = st.columns(3, gap="medium")
            chunk = visible.iloc[start:start + 3]
            for col, (_, row) in zip(cols, chunk.iterrows()):
                with col:
                    source = poster_sources.get(int(row["movieId"]))
                    if source:
                        st.image(source, width="stretch")
                    else:
                        st.image(
                            make_placeholder_poster(str(row["title"]), str(row["genres"])),
                            width="stretch",
                        )
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    st.markdown(f"### {row['title']}")
                    genre_html = "".join(f"<span class='pill'>{g}</span>" for g in str(row["genres"]).split("|")[:3] if g and g != "(no genres listed)")
                    st.markdown(genre_html or "Genre unknown", unsafe_allow_html=bool(genre_html))
                    st.markdown(f"#### ⭐ {row['average_rating']:.2f} / 5")
                    st.caption(f"{int(row['total_ratings']):,} MovieLens ratings")
                    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Find Similar Movies":
    st.subheader("🍿 Find movies with a similar vibe")
    titles = sorted(movies["title"].dropna().unique().tolist())
    chosen_title = st.selectbox("Choose a movie you like", titles)
    chosen = movies[movies["title"] == chosen_title].iloc[0]
    chosen_genres = set(str(chosen["genres"]).split("|")) - {"(no genres listed)", ""}
    st.markdown(f"**Your pick:** {chosen_title}")
    st.caption("Genres: " + (", ".join(sorted(chosen_genres)) if chosen_genres else "Not listed"))
    if st.button("Recommend similar movies", type="primary"):
        candidates = data[(data["movieId"] != chosen["movieId"]) & (data["total_ratings"] >= 10)].copy()
        candidates["shared_genres"] = candidates["genres"].fillna("").apply(lambda value: len(chosen_genres.intersection(set(str(value).split("|")))))
        recs = candidates[candidates["shared_genres"] > 0].sort_values(["shared_genres", "average_rating", "total_ratings"], ascending=False).head(6)
        if recs.empty:
            st.info("No similar movies found for this selection.")
        else:
            rows = [(int(row.movieId), str(row.title)) for row in recs.itertuples()]
            poster_sources = get_cached_poster_sources(rows, poster_cache)
            missing_rows = [item for item in rows if item[0] not in poster_sources]
            if missing_rows and os.getenv("TMDB_API_KEY", "").strip():
                if st.button("Load real posters for recommendations", key="load_recommendation_posters"):
                    with st.spinner("Fetching recommendation posters..."):
                        poster_cache, found, not_found = fetch_and_cache_posters(missing_rows, poster_cache)
                    st.success(f"Fetched {found} posters; {not_found} unavailable.")
                    st.rerun()
            for _, row in recs.iterrows():
                rec_col, info_col = st.columns([1, 3], gap="medium")
                with rec_col:
                    source = poster_sources.get(int(row["movieId"]))
                    if source:
                        st.image(source, width="stretch")
                    else:
                        st.image(
                            make_placeholder_poster(str(row["title"]), str(row["genres"])),
                            width="stretch",
                        )
                with info_col:
                    st.markdown(f"<div class='movie-card'><h3>{row['title']}</h3><p class='muted'>{row['genres']}</p><b>⭐ {row['average_rating']:.2f}/5</b> · {int(row['total_ratings']):,} MovieLens ratings</div>", unsafe_allow_html=True)

elif page == "Ratings & Reviews":
    st.subheader("⭐ Rate a movie and share your thoughts")
    st.markdown("Your rating and review are saved to your account. You can update them later.")
    options = movies[["movieId", "title"]].drop_duplicates().sort_values("title")
    selected_title = st.selectbox("Choose a movie", options["title"].tolist(), key="review_movie")
    selected_id = int(options.loc[options["title"] == selected_title, "movieId"].iloc[0])
    selected_genres = str(movies.loc[movies["movieId"] == selected_id, "genres"].iloc[0])
    selected_source = get_cached_poster_sources([(selected_id, selected_title)], poster_cache).get(selected_id)
    if selected_source:
        st.image(selected_source, width=220)
    else:
        st.image(make_placeholder_poster(selected_title, selected_genres), width=220)
        if os.getenv("TMDB_API_KEY", "").strip() and st.button("Load this movie's real poster", key="load_review_poster"):
            with st.spinner("Fetching this poster..."):
                poster_cache, found, not_found = fetch_and_cache_posters([(selected_id, selected_title)], poster_cache)
            if found:
                st.success("Poster saved for future visits.")
            else:
                st.info("TMDB does not currently have a poster for this title.")
            st.rerun()
    rating_value = st.slider("Your rating", 0.5, 5.0, 4.0, 0.5)
    review_text = st.text_area("Your review (optional)", max_chars=2000)
    if st.button("Submit rating & review", type="primary"):
        uid = st.session_state["user_id"]
        rating_ok = save_movie_rating(uid, selected_id, rating_value)
        review_ok = True
        if review_text.strip():
            review_ok = save_movie_review(uid, selected_id, review_text)
        if rating_ok and review_ok:
            st.success("Your rating and review have been saved!")
        elif not rating_ok:
            st.error("Your rating could not be saved. Check the database connection and try again.")
        else:
            st.warning("Your rating was saved, but the review could not be saved.")

    st.divider()
    st.subheader("Community ratings")
    community = data[data["total_ratings"] > 0].sort_values(["total_ratings", "average_rating"], ascending=False).head(10)
    st.dataframe(community[["title", "genres", "average_rating", "total_ratings"]], width="stretch", hide_index=True)

    if st.button("Show my saved reviews"):
        conn = None
        cur = None
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                """SELECT m.title, ur.review_text, ur.updated_at
                   FROM user_reviews ur
                   JOIN movies m ON m.movieId = ur.movie_id
                   WHERE ur.user_id = %s
                   ORDER BY ur.updated_at DESC""",
                (st.session_state["user_id"],),
            )
            my_reviews = cur.fetchall()
            if my_reviews:
                for title, review, updated in my_reviews:
                    st.markdown(f"<div class='movie-card'><h3>{title}</h3><p class='muted'>{review}</p><small>Updated: {updated}</small></div>", unsafe_allow_html=True)
            else:
                st.info("You have not written any reviews yet.")
        except Exception as exc:
            st.error(f"Could not load saved reviews: {exc}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

elif page == "Ratings Analysis":
    st.subheader("📊 Understand the ratings")
    genre_data = movies[["movieId", "genres"]].copy()
    genre_data["genre"] = genre_data["genres"].fillna("").str.split("|")
    genre_data = genre_data.explode("genre")
    genre_ratings = genre_data.merge(ratings[["movieId", "rating"]], on="movieId", how="inner")
    summary = genre_ratings.groupby("genre").agg(average_rating=("rating", "mean"), total_ratings=("rating", "count")).reset_index()
    summary["average_rating"] = summary["average_rating"].round(2)
    summary = summary[(summary["genre"] != "(no genres listed)") & (summary["genre"] != "")].sort_values("average_rating", ascending=False)
    left, right = st.columns(2)
    with left:
        st.markdown("#### ⭐ Average rating by genre")
        st.bar_chart(summary.set_index("genre")[["average_rating"]])
    with right:
        st.markdown("#### 📈 Rating volume by genre")
        st.bar_chart(summary.sort_values("total_ratings", ascending=False).set_index("genre")[["total_ratings"]])
    st.subheader("Genre details")
    st.dataframe(summary, width="stretch", hide_index=True)

st.divider()
st.caption("CineScope · Movie discovery and ratings · Built with Python, Streamlit, Pandas and MySQL")
