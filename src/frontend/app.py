import streamlit as st
import requests
import concurrent.futures
from PIL import Image
from io import BytesIO
from api_client import ClipStreamClient
from datetime import timedelta

# --- CONFIGURATION ---
PAGE_TITLE = "ClipStream"
PAGE_ICON = "🎬"

client = ClipStreamClient(base_url="https://clipstream-api.onrender.com")

st.set_page_config(
    page_title=PAGE_TITLE, 
    page_icon=PAGE_ICON, 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Responsive & Mobile Friendly) ---
st.markdown("""
<style>
    /* Force vertical scrollbar to always exist (prevents width oscillation in HF iframe) */
    html {
        overflow-y: scroll;
    }

    /* Prevent any horizontal overflow feedback loop */
    body {
        overflow-x: hidden;
    }

    .main-title { 
        font-size: 3rem; 
        color: #FF4B4B; 
        text-align: center; 
        font-weight: 700; 
    }

    .sub-title { 
        text-align: center; 
        color: #555; 
        margin-bottom: 2rem; 
    }

    div.stButton > button:first-child { 
        height: 3em; 
        font-weight: bold; 
    }

    /* Mobile optimization */
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

# Cache the image data so we don't re-download on every click
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_image_from_url(url: str):
    if not url:
        return None
    try:
        # Use a timeout so the app doesn't hang on bad links
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception:
        return None

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
        options=["anime", "amv"],
        default=[c for c in default_cats if c in ["anime", "amv"]],
        placeholder="All Categories"
    )
    
    selected_years = st.multiselect(
        "Years",
        options=[2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015],
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
        query = st.text_input("Search Query", value=default_query, placeholder="e.g., A horse girl running on grass")
    with col2:
        submitted = st.form_submit_button("Search", type="primary", width='stretch')

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

    with st.status("🚀 Connecting to backend...", expanded=True) as status:
        try:
            # 1. The Wake-Up Ping
            # Hit the base URL to force Render to wake up. 
            # We give it a 120-second timeout to handle the cold boot.
            status.write("😴 Checking server status (waking it up if it's asleep)...")
            requests.get("https://clipstream-api.onrender.com/", timeout=120)
            
            # 2. The Actual Search
            status.update(label="🧠 Analyzing video embeddings...", state="running")
            response = client.search(
                query=query,
                categories=selected_cats,
                years=selected_years,
                sort_by=sort_api_value,
                top_k=10 
            )
            
            # 3. Success (collapses the status box)
            status.update(label="✅ Search complete!", state="complete", expanded=False)
            
        except requests.exceptions.Timeout:
            status.update(label="❌ Server took too long to wake up.", state="error")
            response = {"error": True, "message": "Backend is still waking up. Please click search again!"}
        except Exception as e:
            status.update(label="❌ Connection failed.", state="error")
            response = {"error": True, "message": f"Connection error: {str(e)}"}

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
            if s_col1.button("Two middle school girl chilling in a camping place"):
                st.query_params["q"] = "Two middle school girl chilling in a camping place"
                st.rerun()
            if s_col2.button("An anime fight scene with explosions"):
                st.query_params["q"] = "An anime fight scene with explosions"
                st.rerun()
            if s_col3.button("A horse girl running on grass"):
                st.query_params["q"] = "a horse girl running on grass"
                st.rerun()

    else:
        results = response.get("results", [])
        st.session_state.last_results = results

# --- RENDER RESULTS ---
if st.session_state.last_results:
    # Filter results first
    filtered_results = [r for r in st.session_state.last_results if r['score'] >= min_confidence]
    
    # Parallel Fetching: Download all images simultaneously
    # This reduces total load time from sum(all_latencies) to max(single_latency)
    image_urls = [r.get("thumbnail_url") for r in filtered_results]
    
    # "8" workers allows 8 images to download at the exact same time
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        loaded_images = list(executor.map(fetch_image_from_url, image_urls))

    st.divider()
    
    if not filtered_results:
        st.warning(f"Matches found, but none met your confidence threshold.")
    else:
        st.subheader(f"Top Matches ({len(filtered_results)})")
        cols = st.columns(3)
        
        for idx, result in enumerate(filtered_results):
            with cols[idx % 3]:
                with st.container(border=True):
                    # Playback State Logic
                    play_key = f"is_playing_{result['scene_id']}"
                    if play_key not in st.session_state:
                        st.session_state[play_key] = False

                    if st.session_state[play_key]:
                        # Video Player
                        # The 'start_time' parameter in st.video handles the seeking logic
                        st.video(
                            result['video_url'].split('&')[0], 
                            start_time=int(result['start_time'])
                        )
                        if st.button("⏹️ Close Clip", key=f"close_{idx}"):
                            st.session_state[play_key] = False
                            st.rerun()
                    else:
                        # Display the Pre-Fetched Image
                        img_obj = loaded_images[idx]
                        if img_obj:
                            st.image(img_obj, width='stretch')
                        else:
                            # Fallback placeholder
                            st.image("https://placehold.co/600x400/png?text=No+Thumbnail", width='stretch')
                        
                        # Metadata & Buttons
                        badge_text, badge_color = get_confidence_style(result['score'])
                        st.markdown(f"**{result['video_id']}**")
                        st.markdown(f":{badge_color}[**{badge_text}**] `{result['score']:.4f}`")
                        
                        timestamp = str(timedelta(seconds=int(result['start_time'])))
                        st.caption(f"📍 Starts at: **{timestamp}**")
                        
                        if st.button("▶️ Play Video", key=f"btn_{idx}", width='stretch'):
                            st.session_state[play_key] = True
                            st.rerun()