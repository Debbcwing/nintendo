import re
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="🎮 Best Game Recommender", page_icon="🎮", layout="wide")

# ---- BACKGROUND IMAGE ----
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---- CORNER IMAGE (top-right) ----
def add_corner_image(image_file: str, width_px: int = 500):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    /* Container that floats above the app */
    .corner-img {{
        position: absolute;
        top: -4rem;
        right: 5rem;
        z-index: 1000;           /* above page content */
        pointer-events: none;    /* don't block clicks */
    }}
    .corner-img img {{
        width: {width_px}px;
        height: auto;
        border-radius: 12px;     /* optional: rounded */
        opacity: 0.65;           /* slight transparency */
    }}
    </style>
    <div class="corner-img">
        <img src="data:image/png;base64,{encoded}" />
    </div>
    """
    st.markdown(css, unsafe_allow_html=True)



st.title("🎮Best Game Recommender👾")
st.caption(" Use checkboxes to filter and find the best recommendations. **All prices are in EUR (€).**")

# ---------------------- HELPERS ----------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def pick_first_col(df: pd.DataFrame, candidates: list[str] | tuple[str, ...]):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols:
            return cols[c.lower()]
    for want in candidates:
        for c in df.columns:
            if want.lower() in c.lower():
                return c
    return None


def to_year(s):
    try:
        return pd.to_datetime(s, errors="coerce").dt.year
    except Exception:
        return pd.to_datetime(pd.Series(s), errors="coerce").dt.year


def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = np.nanmin(s.values), np.nanmax(s.values)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

# ---------------------- DATA LOADING ----------------------
DEFAULT_CSV = "data/deku_cleanest.csv"

csv_source = DEFAULT_CSV

try:
    df_raw = load_csv(csv_source)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

df = df_raw.copy()

# ---------------------- COLUMN NORMALIZATION ----------------------
TITLE_COL = pick_first_col(df, ("title", "name", "game", "game_title")) or df.columns[0]
GENRE_COL = pick_first_col(df, ("genres", "genre", "tags"))
PLATFORM_COL = pick_first_col(df, ("platform", "platforms", "system"))
CRITIC_COL = pick_first_col(df, ("critic_score", "metascore", "meta_score"))
USER_COL = pick_first_col(df, ("user_score", "rating_user", "user_rating"))
PRICE_COL = pick_first_col(df, ("price", "current_price", "price_eur"))
PLAYTIME_COL = pick_first_col(df, ("playtime_hours", "hours", "avg_playtime"))
DATE_COL = pick_first_col(df, ("release_date", "released", "date"))

if DATE_COL:
    df["_year"] = to_year(df[DATE_COL])
else:
    df["_year"] = np.nan

if GENRE_COL:
    def split_genres(x):
        if pd.isna(x):
            return []
        return re.split(r"[|/,;]+\\s*", str(x))
    df["_genres_list"] = df[GENRE_COL].apply(split_genres)
else:
    df["_genres_list"] = [[] for _ in range(len(df))]

for col in [CRITIC_COL, USER_COL, PRICE_COL, PLAYTIME_COL]:
    if col and col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------- FILTERS ----------------------
st.sidebar.header("Filters")
query = st.sidebar.text_input("Search title contains", "").strip()

selected_genres = []
if GENRE_COL:
    all_genres = sorted({g for lst in df["_genres_list"] for g in lst if g})
    if all_genres:
        with st.sidebar.expander("Filter by Genres (checkboxes)"):
            for g in all_genres:
                if st.checkbox(g, key=f"genre_{g}"):
                    selected_genres.append(g)

selected_platforms = []
if PLATFORM_COL:
    plats = sorted({p.strip() for p in df[PLATFORM_COL].dropna().astype(str).str.split(r"[/,;]").explode().unique()})
    if plats:
        with st.sidebar.expander("Filter by Platforms (checkboxes)"):
            for p in plats:
                if st.checkbox(p, key=f"plat_{p}"):
                    selected_platforms.append(p)

# Apply filters
flt = df.copy()

if query:
    flt = flt[flt[TITLE_COL].astype(str).str.contains(re.escape(query), case=False, na=False)]

if selected_genres:
    flt = flt[flt["_genres_list"].apply(lambda lst: any(g in lst for g in selected_genres))]

if selected_platforms and PLATFORM_COL:
    patt = re.compile("|".join(map(re.escape, selected_platforms)))
    flt = flt[flt[PLATFORM_COL].astype(str).str.contains(patt, na=False)]

# ---------------------- SCORING ----------------------
st.sidebar.header("Scoring Weights")
w_price   = st.sidebar.slider("Weight: Price (lower is better)", 0.0, 1.0, 0.3)
w_critic  = st.sidebar.slider("Weight: Critic score", 0.0, 1.0, 0.4)
w_user    = st.sidebar.slider("Weight: User score", 0.0, 1.0, 0.3)

price_norm = minmax(flt[PRICE_COL].fillna(flt[PRICE_COL].max())) if PRICE_COL else pd.Series(0.0, index=flt.index)
price_term = 1.0 - price_norm

if CRITIC_COL:
    critic_norm = (flt[CRITIC_COL].fillna(flt[CRITIC_COL].median()) / 100.0).clip(0, 1)
else:
    critic_norm = pd.Series(0.0, index=flt.index)

if USER_COL:
    usr = flt[USER_COL].astype(float)
    if usr.max(skipna=True) > 10:
        usr = usr / 10.0
    user_norm = usr.fillna(usr.median()).clip(0, 1)
else:
    user_norm = pd.Series(0.0, index=flt.index)

flt["rec_score"] = (
    (w_price   * price_term) +
    (w_critic  * critic_norm)+
    (w_user    * user_norm)
)

if flt["rec_score"].max() - flt["rec_score"].min() > 1e-9:
    flt["rec_score"] = 100 * minmax(flt["rec_score"])
else:
    flt["rec_score"] = 0

# ---------------------- RESULTS ----------------------
N = st.sidebar.number_input("How many top games?", min_value=5, max_value=50, value=10, step=5)

if len(flt) == 0:
    st.warning("No games match your filters. Try relaxing them.")
    st.stop()

ranked = flt.sort_values("rec_score", ascending=False).head(int(N))

tab_rec, tab_viz, tab_price = st.tabs(["Recommendations", "Top Picks", "Price ⚔️ Rating"])

with tab_rec:
    st.subheader("Top Recommended Games")
    for _, row in ranked.reset_index(drop=True).iterrows():
        with st.container():
            st.markdown(f"### {row[TITLE_COL]}")
            meta = []
            if GENRE_COL and row.get(GENRE_COL):
                meta.append(f"**Genre:** {row[GENRE_COL]}")
            if PLATFORM_COL and row.get(PLATFORM_COL):
                meta.append(f"**Platform:** {row[PLATFORM_COL]}")
            if DATE_COL and not pd.isna(row.get("_year")):
                meta.append(f"**Year:** {int(row['_year'])}")
            st.caption(" | ".join(meta))

            cols = st.columns(4)
            if CRITIC_COL: cols[0].metric("Critic", f"{row.get(CRITIC_COL, np.nan):,.0f}")
            if USER_COL:
                usr_val = row.get(USER_COL, np.nan)
                if pd.notna(usr_val) and usr_val > 10: usr_val = usr_val / 10.0
                cols[1].metric("User", f"{usr_val:,.1f}")
            if PRICE_COL: cols[2].metric("Price (EUR)", f"€{row.get(PRICE_COL, np.nan):,.2f}")
            cols[3].metric("Score", f"{row['rec_score']:,.1f}")
            st.progress(min(1.0, float(row['rec_score'])/100.0))

with tab_viz:
    st.markdown("---")
    st.subheader("Visualize Recommendations")

    # Leaderboard (horizontal bar)
    try:
        fig_bar = px.bar(
            ranked.sort_values("rec_score", ascending=True),
            x="rec_score",
            y=TITLE_COL,
            color=GENRE_COL if GENRE_COL else None,
            orientation="h",
            text=ranked.sort_values("rec_score", ascending=True)["rec_score"].round(1),
            labels={"rec_score": "Recommendation Score", TITLE_COL: "Title"},
            title="Top Picks — Leaderboard",
        )
        fig_bar.update_traces(textposition="outside", cliponaxis=False)
        fig_bar.update_layout(height=600, yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception:
        pass

    # Value map (Price vs Score)

with tab_price:
    st.markdown("---")
    st.subheader("Visualize Recommendations")

    try:
        fig_bar = px.bar(
            ranked.sort_values("rec_score", ascending=True),
            x="rec_score",
            y=TITLE_COL,
            color=GENRE_COL if GENRE_COL else None,
            orientation="h",
            text=ranked.sort_values("rec_score", ascending=True)["rec_score"].round(1),
            labels={"rec_score": "Recommendation Score", TITLE_COL: "Title"},
            title="Top Picks — Leaderboard",
        )
        fig_bar.update_traces(textposition="outside", cliponaxis=False)
        fig_bar.update_layout(height=600, yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception:
        pass


    if PRICE_COL and USER_COL:
        u = pd.to_numeric(flt[USER_COL], errors="coerce")
        if u.max(skipna=True) > 10:
            u = u / 10.0
        flt["_user10"] = u
        st.markdown("Every dot represents a game. Hover over dots to see details. You can also filter games using the checkboxes on the left.")
   
        fig_user = px.scatter(
            flt,
            x=PRICE_COL,
            y="_user10",
            hover_name=TITLE_COL,
            color=GENRE_COL if GENRE_COL else None,
            hover_data=[PLATFORM_COL] if PLATFORM_COL else None,
            labels={PRICE_COL: "Price (EUR)", "_user10": "User Score (0–10)"},
            title="Price ⚔️ User Score",
        )
        fig_user.update_xaxes(tickprefix="€")
        st.plotly_chart(fig_user, use_container_width=True)

    # --- 2) Price vs Critic Score ---
    if PRICE_COL and CRITIC_COL:
        c = pd.to_numeric(flt[CRITIC_COL], errors="coerce")
        if c.max(skipna=True) <= 1.0:
            c = c * 100.0
        flt["_critic100"] = c.clip(0, 100)

        fig_crit = px.scatter(
            flt,
            x=PRICE_COL,
            y="_critic100",
            hover_name=TITLE_COL,
            color=GENRE_COL if GENRE_COL else None,
            hover_data=[PLATFORM_COL] if PLATFORM_COL else None,
            labels={PRICE_COL: "Price (EUR)", "_critic100": "Critic Score (0–100)"},
            title="Price ⚔️ Critic Score",
        )
        fig_crit.update_xaxes(tickprefix="€")
        st.plotly_chart(fig_crit, use_container_width=True)



    if st.checkbox("Show full results table"):
        cols_to_show = [c for c in [TITLE_COL, GENRE_COL, PLATFORM_COL, DATE_COL, CRITIC_COL, USER_COL, PRICE_COL, "rec_score"] if c and (c in flt.columns or c == "rec_score")]
        st.dataframe(ranked[cols_to_show], use_container_width=True, hide_index=True)
