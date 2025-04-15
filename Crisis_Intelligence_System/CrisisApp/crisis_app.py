# Crisis Intelligence MVP Notebook (Streamlit-ready)

# === 1. Imports === #
import pandas as pd
import numpy as np
import re
from datetime import datetime
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from transformers import pipeline
from streamlit_folium import st_folium

# === 2. Load Preprocessed Data === #
tweets = pd.read_csv("social_media_with_temporal_score.csv")

# Disaster Prediction Placeholder (from hackathon model)
def predict_disasters():
    return pd.DataFrame({
        'Zone': ['Zone A', 'Zone C'],
        'Disaster': ['Flood', 'Power Outage'],
        'Probability': [0.87, 0.65],
        'Recommendation': ['Evacuate Zone A', 'Deploy backup generators to Zone C']
    })

# === 3. Fake Tweet Detection (Text + Temporal) === #
@st.cache_data
def classify_tweets(tweets_df):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["real news", "fake news"]

    results = []
    for text in tweets_df['text'].fillna('').astype(str).tolist():
        result = classifier(text, labels)
        results.append({
            'text': text,
            'text_pred': result['labels'][0],
            'text_conf': round(result['scores'][0], 3)
        })

    text_df = pd.DataFrame(results)
    combined = tweets_df.merge(text_df, on='text', how='left')

    def final_label(row):
        if row['temporal_score'] == 1 and row['text_pred'] == 'real news':
            return 'Likely Real'
        elif row['temporal_score'] == 0 and row['text_pred'] == 'fake news':
            return 'Likely Fake'
        else:
            return 'Uncertain'

    combined['final_label'] = combined.apply(final_label, axis=1)
    return combined

# === 4. Streamlit Interface === #
st.set_page_config(layout="wide")
st.title("🪖 Crisis Intelligence System")
st.markdown("Combining real-time disaster predictions and misinformation detection")

# Disaster Section
st.header("🌩️ Predicted Disasters")
disaster_df = predict_disasters()
st.dataframe(disaster_df, use_container_width=True)

# Tweet Detection Section with toggle
st.header("🔎 Social Media Intelligence")
tweet_data = classify_tweets(tweets)

show_map = st.toggle("Show Tweets as Map", value=True)

if show_map:
    st.subheader("🌐 Real vs Fake Tweet Map")
    m = folium.Map(location=[37.7, -122.2], zoom_start=9)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in tweet_data.iterrows():
        color = 'green' if row['final_label'] == 'Likely Real' else 'red' if row['final_label'] == 'Likely Fake' else 'orange'
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.4,
            popup=f"{row['final_label']}: {row['text'][:100]}"
        ).add_to(marker_cluster)

    legend_html = """
    <div style='position: fixed; bottom: 50px; left: 50px; width: 160px; z-index:9999; font-size:14px;
    background-color: white; border:2px solid grey; border-radius:5px; padding: 10px;'>
    <b>Legend</b><br>
    <i style='background:green; width:10px; height:10px; display:inline-block;'></i> Real<br>
    <i style='background:red; width:10px; height:10px; display:inline-block;'></i> Fake<br>
    <i style='background:orange; width:10px; height:10px; display:inline-block;'></i> Uncertain
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=1000, height=550)
else:
    st.dataframe(tweet_data[['text', 'zone', 'timestamp', 'text_pred', 'temporal_score', 'final_label']], use_container_width=True)