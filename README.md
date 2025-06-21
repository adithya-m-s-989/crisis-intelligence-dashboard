# 🛡️ Crisis Intelligence System Dashboard

A multi-layered interactive dashboard for real-time crisis monitoring and intelligence. This project integrates social media signals, infrastructure data, and sensor-based disaster events into a unified geospatial interface to support fast, data-driven emergency response.

---

## 🔗 Live Demo

🌐 **Try it now:**
[🚀 View Deployed App]
https://crisis-intelligence-dashboard.onrender.com/

---

## 📊 Features

* 🔥 **Disaster Incident Mapping**
  Visualize sensor-detected disasters (fires, floods, earthquakes, etc.) across zones.

* 🏥 **Infrastructure Impact Analysis**
  View affected hospitals, shelters, and fire stations with real-time severity status.

* 🐦 **Tweet Intelligence Layer**
  Classify tweets as *Likely Real* or *Possibly Fake* using temporal modeling and NLP heuristics.

* 🤭 **Smart Filtering**
  Filter by disaster type, infrastructure, zone, date, and tweet credibility — all in one row.

* 🗺️ **Map View with Icons**
  Interactive Mapbox visualization with custom icons for facilities.

---

## 📁 File Structure

```
.
├── app.py                             # Main Dash app with callback logic
├── social_media_with_temporal_score.csv  # Tweet data with credibility scores
├── sensor_readings.csv               # Disaster events from sensors
├── final_df.csv                      # Infrastructure data
├── assets/                           # (Optional) CSS styling or images
└── README.md
```

---

## 🧠 How It Works

* **Tweets** are classified using a temporal score to assess credibility.
* **Sensors** report disaster types with severity scores.
* **Infrastructure** records are cross-tagged with predicted impact and recommended actions.
* The **dashboard syncs all three** on a central Mapbox map using Dash.

---

## 🚀 Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/your-username/crisis-intelligence-dashboard.git
cd crisis-intelligence-dashboard
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install dash plotly pandas dash-bootstrap-components
```

### 3. Run the app

```bash
python app.py
```

Then open your browser at `http://localhost:8050`

---

## 🛠️ Tech Stack

* [Dash](https://dash.plotly.com/) + [Plotly](https://plotly.com/) – interactive dashboard and maps
* [Pandas](https://pandas.pydata.org/) – data transformation
* [Mapbox](https://www.mapbox.com/) – custom icons and base map
* [Render](https://render.com/) – deployment ready

---
## 🙌 Acknowledgements

* UC Davis MSBA 2025 cohort project inspiration
* Public disaster datasets (e.g. CalFire, USGS)
* Hugging Face for NLP models powering tweet scoring

---

> “In times of crisis, accurate information is the most valuable asset.”
