# Streamlit App

import streamlit as st
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from streamlit_star_rating import st_star_rating
from graphviz import Digraph


# Import functions from other files
import Model_Helper
from Model_Helper import (
    load_model, 
    calc_review_sentiment_score,
    score_to_stars,
    render_stars
)

import Dataset_Helper
from Dataset_Helper import (
    load_movie_sentiment_ranking_dataset, 
    load_imdb_movie_ranking_dataset, 
    # load_movie_reviews_dataset, 
    create_sentiment_ranking_dataset
)



st.set_page_config(page_title="Movie Review Sentiment Demo", layout="wide")

# --- Session State Initialization ---
if "page" not in st.session_state:
    st.session_state.page = 1



st.title("Analyzing the True Sentiment of Movie Reviews")

movie_ranking_tab1, sentiment_model_tab2, project_background_tab3 = st.tabs(["📊 Movie Rankings", "🧠 Model Inference", "📋 Project Background"])

# Load Model
MODEL_NAME = "andrewoh/RoBERTa-finetuned-hotel-reviews-sentiment-analysis"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = load_model()

# Load Datasets
MOVIE_SENTIMENT_RANKING_DS = load_movie_sentiment_ranking_dataset()
IMDB_MOVIE_RANKS_DS = load_imdb_movie_ranking_dataset()
# IMDB_MOVIE_REVIEWS_DS = load_movie_reviews_dataset()
SENTIMENT_RANKING_DS = create_sentiment_ranking_dataset(MOVIE_SENTIMENT_RANKING_DS, IMDB_MOVIE_RANKS_DS)


# _______________________
# TAB FOR MOVIE RANKINGS
# -----------------------
with movie_ranking_tab1:
    # Header for Ranking Section
    st.markdown(
        "<div style='font-size:28px; font-weight:600; margin-bottom:1rem;'>🎬 Top 250 Movie Sentiment Rankings</div>",
        unsafe_allow_html=True
    )

# _______________________
# FILTER OUT RANKINGS
# -----------------------
    # Filter Datset
    filters_col, results_col = st.columns([1, 5])
    with filters_col:
        st.markdown(
            "<div style='font-size:24px; font-weight:600; margin-bottom:0.25rem;'>Filters</div>",
            unsafe_allow_html=True
        )

        # Search
        search_query = st.text_input("Search by title")

        # Minimum sentiment score
        min_sentiment = st.slider(
            "Minimum Sentiment Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

        # Minimum IMDb rating
        min_imdb = st.slider(
            "Minimum IMDb Rating",
            min_value=0.0,
            max_value=10.0,
            value=8.0,
            step=0.1
        )

        # Genre filter
        all_genres = sorted({
            g.strip()
            for genres in SENTIMENT_RANKING_DS["Genres"].dropna()
            for g in genres.split(",")
        })

        selected_genres = st.multiselect(
            "Genres",
            all_genres
        )

        # Content rating filter
        content_ratings = sorted(
            SENTIMENT_RANKING_DS["Content Rating"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_ratings = st.multiselect(
            "Content Rating",
            content_ratings
        )

        st.markdown("---")

        # Sorting
        sort_by = st.selectbox(
            "Sort by",
            [
                "Sentiment Model Rank",
                "IMDb Rank"
            ]
        )

        sort_order = st.radio(
            "Order",
            ["Ascending", "Descending"]
        )

        page_size = st.selectbox(
            "Movies per page",
            [5, 10, 25, 50, 250],
            index=1
        )



    # Creates copy of dataset before applying filtrations
    FILTERED_SENTIMENT_RANKING_DS = SENTIMENT_RANKING_DS.copy()

    # Search
    if search_query:
        FILTERED_SENTIMENT_RANKING_DS = FILTERED_SENTIMENT_RANKING_DS[
            FILTERED_SENTIMENT_RANKING_DS["Title"].str.contains(search_query, case=False, na=False)
        ]

    # Minimum sentiment
    FILTERED_SENTIMENT_RANKING_DS = FILTERED_SENTIMENT_RANKING_DS[
        FILTERED_SENTIMENT_RANKING_DS["Sentiment Model Score"] >= min_sentiment
    ]

    # Minimum IMDb
    FILTERED_SENTIMENT_RANKING_DS = FILTERED_SENTIMENT_RANKING_DS[
        FILTERED_SENTIMENT_RANKING_DS["IMDb Rating"] >= min_imdb
    ]

    # Genre filter
    if selected_genres:
        selected_genres_set = set(selected_genres)
        FILTERED_SENTIMENT_RANKING_DS = FILTERED_SENTIMENT_RANKING_DS[
            FILTERED_SENTIMENT_RANKING_DS["Genres"].apply(
                lambda g: (
                    isinstance(g, str)
                    and bool(
                        {x.strip() for x in g.split(",")}
                        .intersection(selected_genres_set)
                    )
                )
            )
        ]

    # Content rating filter
    if selected_ratings:
        FILTERED_SENTIMENT_RANKING_DS = FILTERED_SENTIMENT_RANKING_DS[
            FILTERED_SENTIMENT_RANKING_DS["Content Rating"].isin(selected_ratings)
        ]

    # Sorting
    ascending = sort_order == "Ascending"
    FILTERED_SENTIMENT_RANKING_DS = FILTERED_SENTIMENT_RANKING_DS.sort_values(
        by=sort_by,
        ascending=ascending
    ).reset_index(drop=True)

    # Pages
    total_movies = len(FILTERED_SENTIMENT_RANKING_DS)
    total_pages = max(1, (total_movies - 1) // page_size + 1)
    # Clamp page within bounds
    st.session_state.page = max(
        1,
        min(st.session_state.page, total_pages)
    )




# _______________________
# DISPLAY RESULTS
# -----------------------
    with results_col:
        start = (st.session_state.page - 1) * page_size
        end = start + page_size

        PAGED_FILTERED_SENTIMENT_RANKING_DS = FILTERED_SENTIMENT_RANKING_DS.iloc[start:end]


        st.markdown(
            "<div style='font-size:24px; font-weight:600; margin-bottom:0.25rem;'>Rankings</div>",
            unsafe_allow_html=True
        )
        
        st.caption(
            f"Showing {start + 1}–{min(end, total_movies)} "
            f"of {total_movies} movies"
        )

        # Subheader Columns
        header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([1, 1, 2, 3, 3])
        with header_col1:
            st.markdown("<div style='font-size:22px; font-weight:600;'>Rank</div>", unsafe_allow_html=True)
        with header_col2:
            st.markdown("<div style='font-size:22px; font-weight:600;'>Poster</div>", unsafe_allow_html=True)
        with header_col3:
            st.markdown("<div style='font-size:22px; font-weight:600;'>Movie</div>", unsafe_allow_html=True)
        with header_col4:
            st.markdown("<div style='font-size:22px; font-weight:600;'>Details</div>", unsafe_allow_html=True)
        with header_col5:
            st.markdown("<div style='font-size:22px; font-weight:600;'>Description</div>", unsafe_allow_html=True)


        # Sentiment Model Ranking Columns
        for _, row in PAGED_FILTERED_SENTIMENT_RANKING_DS.iterrows():
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 3, 3])
            # 1)Sentiment Model Rank
            with col1:
                st.markdown(
                    f"<div style='font-size:20px; font-weight:600;'>{row['Sentiment Model Rank']}</div>",
                    unsafe_allow_html=True
                )
            # 2)Movie Poster
            with col2:
                st.markdown(
                    f"""
                    <a href="{row['Movie_Link']}" target="_blank">
                        <img src="{row['Movie_Image_URL']}" width="120"/>
                    </a>
                    """,
                    unsafe_allow_html=True
                )
            # 3) Movie Title (clickable) + description
            with col3:
                st.markdown(
                    f"""
                    <a href="{row['Movie_Link']}" target="_blank"
                    style="font-size:24px; font-weight:700; text-decoration:underline;">
                        {row['Title']}
                    </a>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(f"**Sentiment Score:** {row['Sentiment Model Score']:.1f}")
                st.markdown(f"**Reviews Analyzed:** {row['# Reviews']:,}")
            # 4)Extra metadata
            with col4:
                st.markdown(f"**Genres:** {row['Genres']}")
                st.markdown(f"**Content Rating:** {row['Content Rating']}")
                st.markdown(f"**Runtime:** {row['Duration']}")
                st.markdown(f"**IMDb Rank:** #{row['IMDb Rank']}")
                st.markdown(f"**IMDb Rating:** ⭐ {row['IMDb Rating']}")
            # 5)Movie Description
            with col5:
                st.markdown(f"{row['Description']}")
            st.divider()
        
        # --- Bottom Pagination Controls ---
        nav = st.columns([1, 2, 2, 1])

        # ◀ Previous page
        with nav[0]:
            if st.button("◀", disabled=st.session_state.page == 1):
                st.session_state.page -= 1
                st.rerun()

        # Centered page text
        with nav[1]:
            st.markdown(
                f"<div style='text-align:center; font-weight:500;'>"
                f"Page {st.session_state.page} of {total_pages}"
                f"</div>",
                unsafe_allow_html=True
            )

        # 🔽 Page selector dropdown (narrow column)
        with nav[2]:
            selected_page = st.selectbox(
                "Jump to page",
                options=list(range(1, total_pages + 1)),
                index=st.session_state.page - 1,
                label_visibility="collapsed",
                key="page_selector"
            )

            if selected_page != st.session_state.page:
                st.session_state.page = selected_page
                st.rerun()

        # ▶ Next page
        with nav[3]:
            if st.button("▶", disabled=st.session_state.page == total_pages):
                st.session_state.page += 1
                st.rerun()


# _______________________
# TAB FOR SENTIMENT MODEL
# -----------------------
with sentiment_model_tab2:
    st.header("Run the Sentiment Model")

    # Ask user to give movie review
    review_text = st.text_area(
        "Enter a movie review:",
        placeholder="Type or paste a movie review here...",
        height=200
    )
    # Track length of review
    char_count = len(review_text)
    st.caption(f"{char_count} characters")
    st.caption("Recommended: under ~2,000 characters for best results.")

    # Give user warning if review is too long
    tokens = TOKENIZER(review_text, add_special_tokens=True)["input_ids"]
    token_count = len(tokens)
    if token_count > 512:
        st.warning("⚠️ WARNING: Your review is longer than the model can handle, so part of it will be trimmed. For best results, try shortening it.")


    
    # Run the sentiment model
    if st.button("Run Model"):
        st.divider()
        if review_text.strip():
            with st.spinner("Running model..."):
                results = MODEL(review_text)[0]
                sentiment_score = calc_review_sentiment_score(results)
            
            st.subheader("Model Sentiment Analysis")
            col1, col2 = st.columns([1, 1])

            # Big score
            with col1:
                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <h1 style="margin-bottom:0;">
                            {sentiment_score:.2f}
                            <span style="font-size:0.45em; color:gray;"> / 100</span>
                        </h1>
                        <p style="color:gray; margin-top:4px;">Sentiment Score</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # Stars + label
            with col2:
                num_star_rating = score_to_stars(sentiment_score)
                # Center using columns (reliable)
                left, center, right = st.columns([1, 1, 1])
                with center:
                    stars = st_star_rating(
                        "",
                        read_only=True,
                        maxValue = 5,
                        defaultValue=num_star_rating,
                        size=32,
                    )
                    # stars
                    st.markdown(
                        f"""
                        <div style="text-align:center;">
                            <p style="color:gray; margin-top:6px;">
                                {num_star_rating} / 5 stars
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )



            # ----------------------------
            # Breakdown section
            # ----------------------------
            st.markdown("### Sentiment Score Breakdown")


            label_map = {
                "LABEL_0": "⭐ <b>Very Negative</b>",
                "LABEL_1": "⭐⭐ <b>Negative</b>",
                "LABEL_2": "⭐⭐⭐ <b>Neutral</b>",
                "LABEL_3": "⭐⭐⭐⭐ <b>Positive</b>",
                "LABEL_4": "⭐⭐⭐⭐⭐ <b>Very Positive</b>",
            }

            for r in results[::-1]:  # show positive → negative
                st.markdown(
                    f"{label_map.get(r['label'], r['label'])}: {r['score']:.2%}",
                    unsafe_allow_html=True
                )
                st.progress(r["score"])

        else:
            st.warning("Please enter some text.")
    
        
# --------------------------
# TAB FOR PROJECT BACKGROUND 
# --------------------------
with project_background_tab3:
    st.header("Project Background")

    # ---- Goal ----
    st.markdown("### Goal")
    st.markdown(
        """
        The goal of this project is to re-rank IMDb’s Top 250 movies using **only review text**, 
        eliminating reliance on numerical rating values. Numerical ratings are inherently subjective. Users 
        interpret scales differently, and identical scores may represent vastly different opinions from person to person.

        By analyzing sentiment directly from written reviews, this project produces a more **objective and 
        interpretable ranking** grounded in natural language rather than explicit numeric input.
        """
    )

    # ---- Background ----
    st.markdown("### Background")
    st.markdown(
        """
        IMDb’s Top 250 ranking system aggregates millions of user ratings into a single numerical score per movie. 
        While effective at scale, this approach discards the **semantic richness of review text**. Written reviews 
        encode emotional intensity, contextual qualifiers, and nuanced critique that numerical ratings fail to capture.

        To address this limitation, every available review associated with IMDb’s Top 250 movies was scraped and 
        treated as an **independent signal of audience sentiment**. Instead of averaging star ratings, this project 
        infers sentiment directly from language, allowing rankings to be driven by **what users say rather than what 
        number they select**.
        """
    )

    # ---- Methodology ----
    st.markdown("### Methodology")
    st.markdown(
        """
        This project employs a custom **transformer-based sentiment analysis pipeline** built by fine-tuning 
        **RoBERTa-large** for the movie-review domain. Domain adaptation is critical for accurately capturing 
        film-specific language, mixed sentiment, and nuanced critique common in cinematic reviews.

        Each review is processed by the model to generate a **probability distribution over ordered sentiment 
        classes**. These probabilities are converted into a **continuous sentiment score**, enabling fine-grained 
        comparison across reviews. Review-level scores are then aggregated at the movie level to compute a final 
        **sentiment-based ranking**.
        """
    )


    st.markdown("### Sentiment Ranking Pipeline")

    dot = Digraph()
    dot.attr(
        rankdir="LR",
        size="24,20",
        fontname="Helvetica",
        fontsize="30"   # ⬅️ graph-level font size
    )

    # Node styling
    # dot.attr(
    #     "node",
    #     shape="box",
    #     style="rounded,filled",
    #     fillcolor="#F5F7FA",
    #     color="#4A4A4A",
    #     fontname="Helvetica",
    #     fontsize="16"   # ⬅️ node text size
    # )

    # Edge styling
    dot.attr(
        "edge",
        fontname="Helvetica",
        fontsize="20"
    )

    dot.node("A", "IMDb Top 250 Reviews\n(Web Scraping)")
    dot.node("B", "Text Preprocessing\n& Tokenization")
    dot.node("C", "RoBERTa-Large\nFine-Tuned Sentiment Model")
    dot.node("D", "Class Probabilities\n→ Continuous Score")
    dot.node("E", "Movie-Level\nScore Aggregation")
    dot.node("F", "Final Sentiment-Based\nMovie Rankings")

    dot.edges([
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("E", "F"),
    ])

    st.graphviz_chart(dot)