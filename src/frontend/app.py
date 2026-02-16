import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from api_client import ClipStreamClient
from datetime import timedelta

# --- CONFIGURATION ---
PAGE_TITLE = "ClipStream"
PAGE_ICON = "🎬"

client = ClipStreamClient(base_url="http://localhost:8000")

st.set_page_config(
    page_title=PAGE_TITLE, 
    page_icon=PAGE_ICON, 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Responsive & Mobile Friendly) ---
st.markdown("""
<style>
    .main-title { font-size: 3rem; color: #FF4B4B; text-align: center; font-weight: 700; }
    .sub-title { text-align: center; color: #555; margin-bottom: 2rem; }
    div.stButton > button:first-child { height: 3em; font-weight: bold; }
    /* Mobile optimization for badges */
    @media (max-width: 640px) {
        .main-title { font-size: 2rem; }
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER: CONFIDENCE BADGES ---
def get_confidence_style(score: float):
    if score >= 0.30: return "🟢 High", "green"
    elif score >= 0.27: return "🟡 Medium", "orange"
    else: return "🔴 Low", "red"

# --- 1. URL PARAMETER HANDLING (Share Logic) ---
# We read parameters on load to pre-fill widgets
query_params = st.query_params
default_query = query_params.get("q", "")
# Handle list for categories (Streamlit params are strings, need to parse if multiple)
default_cats = query_params.get_all("cat") if "cat" in query_params else []

# --- SESSION STATE ---
if "search_history" not in st.session_state: st.session_state.search_history = []
if "last_results" not in st.session_state: st.session_state.last_results = []
# Trigger search automatically if URL param exists and we haven't searched yet
if "auto_run" not in st.session_state: 
    st.session_state.auto_run = True if default_query else False

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("### ⚙️ Search Filters")
    
    selected_cats = st.multiselect(
        "Categories",
        options=["anime", "movie"],
        default=[c for c in default_cats if c in ["anime", "movie"]],
        placeholder="All Categories"
    )
    
    selected_years = st.multiselect(
        "Years",
        options=[2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017],
        default=[], 
        placeholder="All Years"
    )

    st.divider()
    
    min_confidence = st.slider(
        "Min Confidence Threshold", 
        min_value=0.25, max_value=1.0, value=0.27, step=0.01
    )
    
    sort_option = st.radio(
        "Sort Results By:",
        options=["Highest Confidence", "Chronological (Time)"],
        index=0
    )
    sort_api_value = "time" if sort_option == "Chronological (Time)" else "confidence"

    st.divider()
    st.caption(f"v0.5.0 | {PAGE_TITLE} Team")

# --- MAIN INTERFACE ---
st.markdown('<div class="main-title">Find the exact moment.</div>', unsafe_allow_html=True)

# Search Form
with st.form("search_form"):
    col1, col2 = st.columns([5, 1])
    with col1:
        # Pre-fill with URL param if available
        query = st.text_input("Search Query", value=default_query, placeholder="e.g., 'Eren Yeager transformation'")
    with col2:
        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

# --- SEARCH LOGIC ---
# Run if button clicked OR if we have a URL query pending (auto_run)
if submitted or (st.session_state.auto_run and query):
    # Reset auto_run so we don't loop
    st.session_state.auto_run = False
    
    # Update History
    if query not in st.session_state.search_history:
        st.session_state.search_history.append(query)
    
    # 2. Sync URL Parameters (State Persistance)
    st.query_params["q"] = query
    if selected_cats:
        st.query_params["cat"] = selected_cats
    else:
        # Clear param if empty
        if "cat" in st.query_params: del st.query_params["cat"]

    with st.spinner("🧠 Analyzing video embeddings..."):
        response = client.search(
            query=query,
            categories=selected_cats,
            years=selected_years,
            sort_by=sort_api_value,
            top_k=20 
        )

    if response.get("error"):
        # 3. Custom No Results / Error Page
        st.empty() # Clear spinner
        st.session_state.last_results = []
        
        with st.container(border=True):
            st.markdown(f"### 🚫 {response['message']}")
            st.markdown("---")
            st.markdown("**Suggestions:**")
            st.markdown("- Try broader keywords (e.g., *'baseball'* instead of *'Ohtani 2023'*).")
            st.markdown("- Lower the **Confidence Threshold** in the sidebar.")
            st.markdown("- **Try these example queries:**")
            
            # Quick-click suggestions
            s_col1, s_col2, s_col3 = st.columns(3)
            if s_col1.button("⚾ Baseball Home Run"):
                st.query_params["q"] = "Baseball Home Run"
                st.rerun()
            if s_col2.button("⚔️ Anime Fight"):
                st.query_params["q"] = "Anime Fight"
                st.rerun()
            if s_col3.button("🌆 City Skyline"):
                st.query_params["q"] = "City Skyline"
                st.rerun()

    else:
        results = response.get("results", [])
        st.session_state.last_results = results

# --- RENDER RESULTS ---
if st.session_state.last_results:
    # Filter by Slider
    filtered_results = [r for r in st.session_state.last_results if r['score'] >= min_confidence]
    
    st.divider()
    
    # 4. Share Link Feature
    # Construct a shareable link (simulated since we are localhost)
    share_url = f"http://localhost:8501/?q={query.replace(' ', '+')}"
    if selected_cats:
        for cat in selected_cats:
            share_url += f"&cat={cat}"
            
    with st.expander("🔗 Share these results"):
        st.code(share_url, language="text")
        st.caption("Copy this link to share your search configuration with others.")

    if not filtered_results:
        st.warning(f"Matches found, but none met your confidence threshold of **{min_confidence}**.")
    else:
        st.subheader(f"Top Matches ({len(filtered_results)})")
        cols = st.columns(3)
        
        for idx, result in enumerate(filtered_results):
            with cols[idx % 3]:
                with st.container(border=True):
                    # 1. Show the Public Google Drive Thumbnail
                    thumb = result.get("thumbnail_url")
                    if thumb:
                        try:
                            # 1. Fetch the image data on the server side
                            response = requests.get(thumb)
                            response.raise_for_status() # Check for HTTP errors
                            
                            # 2. Convert to an image object
                            image_data = Image.open(BytesIO(response.content))
                            
                            # 3. Display the image object directly
                            st.image(image_data, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                    else:
                        st.image("https://placehold.co/600x400/png?text=No+Thumbnail", use_container_width=True)
                    
                    # 2. Metadata Labels
                    badge_text, badge_color = get_confidence_style(result['score'])
                    st.markdown(f"**{result['video_id']}**")
                    st.markdown(f":{badge_color}[**{badge_text}**] `{result['score']:.4f}`")
                    
                    timestamp = str(timedelta(seconds=int(result['start_time'])))
                    st.caption(f"📍 Starts at: **{timestamp}**")
                    
                    # 3. Play Video Logic
                    # We use an expander or a conditional block to show the video player
                    if st.button("▶️ Play Video", key=f"btn_{idx}", use_container_width=True):
                        if "youtube.com" in result['video_url']:
                            # Streamlit handles YouTube embedding automatically
                            st.video(result['video_url'])
                        else:
                            st.warning("No YouTube source available for this clip.")