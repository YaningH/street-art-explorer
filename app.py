import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import ast

st.set_page_config(
    page_title="Belgium Street Art Explorer",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>

* { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
.stApp { background-color: #F9FCFF; color: #1A1A1A; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 3rem 3rem 3rem; max-width: 100%; }

.hero-wrap {
    position: relative;
    padding: 0 0 1rem 0;
    margin-bottom: 0;
}

.hero-title {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: clamp(80px, 12vw, 160px);
    font-weight: 300;
    line-height: 1;
    color: #000000;
    letter-spacing: 0;
    margin: 0;
    padding: 0;
    user-select: none;
}

.hero-count {
    position: absolute;
    bottom: 1rem;
    right: 0;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 1rem;
    color: #000000;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #C8C5BF;
    gap: 100px;
    margin-bottom: 3px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    color: #1A1A1A;
    background: transparent;
    border: none;
    padding: 0.6rem 2.5rem 0.6rem 0;
    margin-right: 0;
}
            
.stTabs [data-baseweb="tab"]:hover {
    color: #1A1A1A;
    transition: 0.5s;
}

.stTabs [aria-selected="true"] {
    color: #1A1A1A;
    border-bottom: 2px solid #1A1A1A;
    background: transparent;
    font-weight: 500;
    margin-bottom: -1px;
}

.stTabs [data-baseweb="tab-highlight"] { background: transparent; }
.stTabs [data-baseweb="tab-border"] { 
            display: none; 
        }

.tab-description {
    font-size: 1rem;
    color: #1A1A1A;
    line-height: 1.3;
    max-width: 600px;
    margin-bottom: 50px;
}
            
.stSelectbox > label, .stTextInput > label {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 0.7rem;
    color: #1A1A1A;
}

.stSelectbox [data-baseweb="select"] > div,
.stTextInput input {
    background: transparent;
    border: none;
    border-bottom: 1px solid #1A1A1A;
    border-radius: 0;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 0.85rem;
    color: #1A1A1A;
    box-shadow: none;
    padding: 0.2rem 0;
}

.stSelectbox [data-baseweb="select"] > div:focus-within {
    border-bottom: 1px solid #1A1A1A ;
}
            
.stSelectbox [data-baseweb="select"] svg {
    fill: #888;
}

[data-baseweb="popover"] * {
    color: #1A1A1A;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 0.8rem;
    border-radius: 0;
}

li[role="option"] {
    background-color: #EEECEA;
    color: #1A1A1A;
}

li[role="option"]:hover {
    background-color: #E0DDD8;
}

[data-testid="stMetric"] {
    background: transparent;
    border: 1px solid #C8C5BF;
    padding: 0.5rem;
}

[data-testid="stMetricLabel"] {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 0.7rem;
    color: #C8C5BF;
}

[data-testid="stMetricValue"] {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 1.2rem;
    color: #1A1A1A;
    font-weight: 400;
    padding-left: 10px;
}
            
.legend {
    display: flex;
    gap: 1.5rem;
    margin: 0.5rem 0 0.75rem 0;
    font-size: 0.72rem;
    color: #555;
    align-items: center;
}

.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
}

.archive-card {
    border-top: 1px solid #C8C5BF;
    padding: 0.85rem 0;
}

.archive-card-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.5rem;
}

.archive-card-title {
    font-size: 0.82rem;
    font-weight: 500;
    color: #1A1A1A;
}

.archive-card-link {
    font-size: 0.7rem;
    color: #1A1A1A;
    text-decoration: none;
    letter-spacing: 0.04em;
}

.archive-data-row {
    display: grid;
    grid-template-columns: 140px 1fr 1fr 1fr;
    gap: 0.25rem 1rem;
    margin-bottom: 0.2rem;
    font-size: 0.72rem;
    color: #555;
}

.archive-data-row.desc-row {
    grid-template-columns: 140px 1fr;
    align-items: start;
    margin-top: 0.3rem;
}

.archive-field-label { font-size: 0.68rem; color: #AAA; }
.archive-collection-label { font-size: 0.7rem; letter-spacing: 0.06em; color: #888; padding: 1rem 0 0.25rem 0; }
.archive-search-label { font-size: 0.72rem; color: #888; margin-bottom: 0.5rem; }
            
[data-testid="stFileUploader"] {
    background: transparent;
    border: none;
    padding: 0;
}

[data-testid="stFileUploader"] section {
    background-color: transparent !important;
    border: 1px dashed #C8C5BF !important;
    border-radius: 0 !important;
}

[data-testid="stFileUploadDropzone"] {
    background-color: transparent !important;
}

[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #1A1A1A !important;
}

[data-testid="stFileUploader"] section button {
    background: #1A1A1A !important;
    color: #F9FCFF !important;
    border: none !important;
    border-radius: 0 !important;
    font-size: 0.75rem !important;
}

[data-testid="stFileUploader"] section button:hover {
    opacity: 0.7 !important;
    transition: 0.3s;
}
            
[data-testid="stFileUploader"] * {
    color: #1A1A1A !important;
}

[data-testid="stFileUploader"] > div > div:last-child {
    background: transparent !important;
    border: 1px solid #C8C5BF !important;
}

[data-testid="stFileUploader"] > div > div:last-child button {
    background: transparent !important;
    border: none !important;
}

[data-testid="stFileUploader"] > div > div:last-child button svg {
    fill: #1A1A1A !important;
    stroke: #1A1A1A !important;
}

.ai-caption-label {
    font-size: 0.8rem;
    color: #888;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
}

.ai-caption-text {
    font-size: 0.85rem;
    color: #1A1A1A;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}

.section-label {
    font-size: 0.8rem;
    color: #888;
    margin-top: 50px;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

def clean_html(text):
    if pd.isna(text):
        return ""
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ").strip()


def parse_list_field(value):
    if pd.isna(value):
        return []
    try:
        result = ast.literal_eval(value)
        if isinstance(result, list):
            return [str(v).strip() for v in result]
        return [str(result).strip()]
    except Exception:
        return [str(value).strip()]


def get_year(date_str):
    if pd.isna(date_str):
        return ""
    try:
        return str(pd.to_datetime(date_str).year)
    except:
        return ""

@st.cache_data
def load_data():
    df = pd.read_csv("data/Month-2026-05.csv", low_memory=False)
    df["description_clean"] = df["description"].apply(clean_html)
    df["title"] = df["title"].fillna("Untitled")
    df["styles"] = df["attribute_artwork_style"].apply(parse_list_field)
    df["city_display"] = df["city_id"].str.replace("-", " ").str.title()
    df["year"] = df["attribute_date_created"].apply(get_year)
    df["search_text"] = (
        df["title"].fillna("") + " "
        + df["description_clean"] + " "
        + df["attribute_festival"].fillna("") + " "
        + df["artist1_title"].fillna("")
    )
    return df

# BLIP model
@st.cache_resource(show_spinner="Loading model...")
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model


def caption_image(image: Image.Image, processor, model) -> str:
    inputs = processor(image.convert("RGB"), return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(out[0], skip_special_tokens=True)


# Similarity search
def find_similar_artworks(caption: str, df: pd.DataFrame, top_n: int = 6):
    corpus = list(df["search_text"].fillna("")) + [caption]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10_000)
    tfidf = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
    top_idx = scores.argsort()[::-1][:top_n]
    results = df.iloc[top_idx].copy()
    results["similarity_score"] = scores[top_idx]
    return results


# Map builder
def build_map(df: pd.DataFrame) -> folium.Map:
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="CartoDB positron",
    )
    for _, row in df.iterrows():
        color = "#2A7A2A" if row["status"] == "active" else "#AAAAAA"
        popup_html = f"""
        <div style="width:210px;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;
                    font-size:11px;line-height:1.7;">
            <div style="font-weight:500;margin-bottom:4px;">{row['title']}</div>
            <div style="color:#666;">{row.get('artist1_title','Unknown')}</div>
            <div style="color:#999;font-size:10px;">{row.get('city_display','')}</div>
            <div style="margin-top:8px;">
                <a href="{row['url']}" target="_blank"
                   style="color:#1A1A1A;font-size:10px;letter-spacing:0.04em;">
                   View on Street Art Cities
                </a>
            </div>
        </div>
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=230),
            tooltip=row["title"],
        ).add_to(m)
    return m


def render_archive_card(row):
    title   = row.get("title", "Untitled")
    artist1 = row.get("artist1_title", "") or ""
    artist2 = row.get("artist2_title", "") or ""
    artist3 = row.get("artist3_title", "") or ""
    city    = row.get("city_display", "") or ""
    country = row.get("country", "") or ""
    lat     = row.get("latitude", "")
    lon     = row.get("longitude", "")
    address = row.get("address", "") or ""
    desc    = row.get("description_clean", "") or ""
    url     = row.get("url", "#")
    year    = row.get("year", "") or ""

    lat_str = f"{lat:.7f}" if isinstance(lat, float) else str(lat)
    lon_str = f"{lon:.7f}" if isinstance(lon, float) else str(lon)

    st.markdown(f"""
    <div class="archive-card">
        <div class="archive-card-header">
            <div class="archive-card-title">{title}</div>
            <a class="archive-card-link" href="{url}" target="_blank">View →</a>
        </div>
        <div class="archive-data-row">
            <div class="archive-field-label">Artist</div>
            <div>{artist1}</div>
            <div>{artist2}</div>
            <div>{artist3}</div>
        </div>
        <div class="archive-data-row">
            <div class="archive-field-label">City / Country</div>
            <div>{city}</div>
            <div>{country}</div>
            <div>{year}</div>
        </div>
        <div class="archive-data-row">
            <div class="archive-field-label">Coordinates</div>
            <div>{lat_str}</div>
            <div>{lon_str}</div>
            <div style="color:#888;font-size:0.68rem;">{address[:50]}</div>
        </div>
        <div class="archive-data-row desc-row">
            <div class="archive-field-label">Description</div>
            <div style="line-height:1.55;color:#444;">{desc[:400] + "..." if len(desc) > 400 else desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    df = load_data()

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">Belgium Street<br>Art Explorer</div>
        <div class="hero-count">7,199 works</div>
    </div>
    """, unsafe_allow_html=True)

    tab_map, tab_archive, tab_search = st.tabs(["Map", "Archive", "Search"])

    # Map
    with tab_map:
        st.markdown('<p class="tab-description">A collection of street artworks across Belgium. Filter by city and status to explore works in specific locations, or click any pin to view details.</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            cities = ["All cities"] + sorted(df["city_display"].unique().tolist())
            selected_city = st.selectbox("City", cities)
        with col2:
            selected_status = st.selectbox("Status", ["All", "Active only", "Removed only"])
        with col3:
            map_df_count = df.copy()
            if selected_city != "All cities":
                map_df_count = map_df_count[map_df_count["city_display"] == selected_city]
            st.metric("Total results", f"{len(map_df_count):,}")

        map_df = df.copy()
        if selected_city != "All cities":
            map_df = map_df[map_df["city_display"] == selected_city]
        if selected_status == "Active only":
            map_df = map_df[map_df["status"] == "active"]
        elif selected_status == "Removed only":
            map_df = map_df[map_df["status"] == "removed"]

        st.markdown("""
        <div class="legend">
            <span><span class="legend-dot" style="background:#2A7A2A;"></span> Active</span>
            <span><span class="legend-dot" style="background:#AAAAAA;"></span> Removed</span>
        </div>
        """, unsafe_allow_html=True)

        display_df = map_df if len(map_df) <= 3000 else map_df.sample(3000, random_state=42)

        st_folium(build_map(display_df), use_container_width=True, height=560)

    # Archive
    with tab_archive:
        st.markdown('<p class="tab-description">Browse and search the full archive of Belgian street art. Filter by artist, location, or keyword to find specific works.</p>', unsafe_allow_html=True)
        st.markdown('<div class="archive-search-label">Search by</div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            city_query = st.selectbox("Location", ["All"] + sorted(df["city_display"].unique().tolist()), key="arc_city")
        with col_b:
            artists = ["All"] + sorted(df["artist1_title"].dropna().unique().tolist())
            artist_query = st.selectbox("Artist", artists, key="arc_artist")
        with col_c:
            keywords = ["All"] + sorted(set(
                k for sublist in df["styles"].tolist() for k in sublist if k
            ))
            keyword_query = st.selectbox("Style", keywords, key="arc_keyword")

        results = df.copy()
        if artist_query != "All":
            results = results[results["artist1_title"] == artist_query]
        if city_query != "All":
            results = results[results["city_display"] == city_query]
        if keyword_query != "All":
            results = results[results["styles"].apply(lambda x: keyword_query in x)]

        total = len(results)
        col_show, col_page, col_empty = st.columns([0.5, 0.5, 3])
        with col_show:
            per_page = st.selectbox("Show", [10, 25, 50, 100], index=1, key="per_page")
        with col_page:
            total_pages = max(1, -(-total // per_page))
            page = st.selectbox("Page", range(1, total_pages + 1), key="arc_page")

        start = (page - 1) * per_page
        end = start + per_page

        st.markdown(f'<div class="archive-collection-label">Collection &nbsp; {total:,} works &nbsp; — &nbsp; Page {page} of {total_pages}</div>', unsafe_allow_html=True)

        if results.empty:
            st.caption("No artworks match your filters.")
        else:
            for _, row in results.iloc[start:end].iterrows():
                render_archive_card(row)

    # Search
    with tab_search:
        st.markdown('<p class="tab-description">Upload a photo of any street art and the AI will analyze it and find visually similar works from the dataset.</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload a street art photo", type=["jpg", "jpeg", "png", "webp"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            col_img, col_result = st.columns([1, 1])

            with col_img:
                st.image(image, use_container_width=True)

            with col_result:
                with st.spinner("Analysing image..."):
                    processor, model = load_blip()
                    caption = caption_image(image, processor, model)

                st.markdown(f"""
                    <div class="ai-caption-label">Description</div>
                    <div class="ai-caption-text">{caption}</div>
                <p style="font-size:0.62rem;color:#999;letter-spacing:0.06em;">
                    Model: Salesforce/blip-image-captioning-base
                </p>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Similar works in the dataset</div>', unsafe_allow_html=True)

            similar = find_similar_artworks(caption, df)
            for _, row in similar.iterrows():
                render_archive_card(row)


if __name__ == "__main__":
    main()