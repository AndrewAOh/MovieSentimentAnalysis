# Load the datasets for rankings and reviews

import streamlit as st
import pandas as pd
from transformers import pipeline

# File Paths
MOVIE_RANKING_FP = "Dataset/Top250_MovieSentiment_Sorted.csv"
TOP_250_MOVIES_FP = "Dataset/IMDB_Top250_Movies.csv"
# MOVIE_REVIEWS_FP = "Dataset/IMDB_MovieReviews.csv"


@st.cache_data
def load_movie_sentiment_ranking_dataset():
    movie_sentiment_ranking_dataset = pd.read_csv(MOVIE_RANKING_FP)
    return clean_movie_titles(movie_sentiment_ranking_dataset)

@st.cache_data
def load_imdb_movie_ranking_dataset():
    imdb_movie_ranking_dataset = pd.read_csv(TOP_250_MOVIES_FP)
    imdb_movie_ranking_dataset = clean_movie_titles(imdb_movie_ranking_dataset)
    return clean_movie_descriptions(imdb_movie_ranking_dataset)

# @st.cache_data
# def load_movie_reviews_dataset():
#     movie_reviews_dataset = pd.read_csv(MOVIE_REVIEWS_FP)
#     return clean_movie_titles(movie_reviews_dataset)

def clean_movie_titles(df):
    df = df.copy()
    df["Movie_Title"] = df["Movie_Title"].str.replace("&apos;", "'", regex=False)
    return df

def clean_movie_descriptions(df):
    df = df.copy()
    df["Movie_Description"] = df["Movie_Description"].str.replace("&apos;", "'", regex=False)
    df["Movie_Description"] = df["Movie_Description"].str.replace("&quot;", '"', regex=False)
    return df


@st.cache_data
def create_sentiment_ranking_dataset(MOVIE_SENTIMENT_RANKING_DS, IMDB_MOVIE_RANKS_DS):
    # Combine datasets
    Sentiment_Ranking_Dataset = MOVIE_SENTIMENT_RANKING_DS.merge(
        IMDB_MOVIE_RANKS_DS,
        on=["Movie_Title", "Movie_Rank", "Movie_Review_Rating"],
        how="inner"
    )

    # Keep only desired final columns
    columns_to_keep = [
        'Movie_Title', 'average_sentiment_score', 'Movie_Rank', 'num_reviews', 
        'Movie_Review_Rating', 'Movie_Description', 'Movie_Content_Rating', 
        'Movie_Genres', 'Movie_Runtime_Duration_Minutes', 'Movie_Link', 
        'Movie_Image_URL'
    ]
    Sentiment_Ranking_Dataset = Sentiment_Ranking_Dataset[columns_to_keep]

    # Change Column Names
    renamed_columns = {
        'Movie_Title': 'Title', 'average_sentiment_score': 'CineScore', 
        'Movie_Rank': 'IMDb Rank', 'num_reviews': '# Reviews', 'Movie_Review_Rating': 'IMDb Rating', 
        'Movie_Description': "Description", 'Movie_Content_Rating': "Content Rating", 
        'Movie_Genres': "Genres", 'Movie_Runtime_Duration_Minutes': 'Duration'
    }
    Sentiment_Ranking_Dataset = Sentiment_Ranking_Dataset.rename(columns=renamed_columns)

    # Create Column for Sentiment Model Ranking
    Sentiment_Ranking_Dataset["CineScore Rank"] = Sentiment_Ranking_Dataset.index + 1

    # Adjust scoring of sentiment model (x100)
    Sentiment_Ranking_Dataset['CineScore'] = Sentiment_Ranking_Dataset['CineScore'] * 100

    # Runtime conversion from minutes → hours & minutes
    Sentiment_Ranking_Dataset["Duration"] = (
        (Sentiment_Ranking_Dataset["Duration"] // 60).astype(int).astype(str)
        + "h "
        + (Sentiment_Ranking_Dataset["Duration"] % 60).astype(int).astype(str)
        + "min"
    )

    # Fill in empty content ratings
    Sentiment_Ranking_Dataset["Content Rating"] = (
        Sentiment_Ranking_Dataset["Content Rating"]
        .fillna("None")
    )



    return Sentiment_Ranking_Dataset
