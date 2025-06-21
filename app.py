import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import re
import os

app = Dash(__name__)

# ---- Load Data ----
tweets = pd.read_csv("social_media_with_temporal_score.csv")
disaster_df = pd.read_csv("sensor_readings.csv")
infra_df = pd.read_csv("final_df.csv")

# ---- Preprocess Tweets ----
tweets['label'] = tweets['temporal_score'].apply(lambda x: 'Likely Real' if x == 1 else 'Possibly Fake')
tweets['timestamp'] = pd.to_datetime(tweets['timestamp'], errors='coerce')
tweets = tweets.dropna(subset=['timestamp'])
tweets['date'] = tweets['timestamp'].dt.date
tweets['zone'] = tweets['zone'].fillna(tweets['text'].str.extract(r'(Zone\s+[A-Z])')[0])

# ---- Preprocess Disaster Data ----
disaster_df['timestamp'] = pd.to_datetime(disaster_df['timestamp'], errors='coerce')
disaster_df = disaster_df.dropna(subset=['timestamp'])
disaster_df['date'] = disaster_df['timestamp'].dt.date
if 'zone' not in disaster_df.columns:
    disaster_df['zone'] = disaster_df['disaster'].astype(str).str.extract(r'(Zone\s+[A-Z])')[0]
disaster_df['zone'] = disaster_df['zone'].fillna('Unknown')

# ---- Preprocess Infrastructure Data ----
infra_df['timestamp'] = pd.to_datetime(infra_df['timestamp'], errors='coerce')
infra_df = infra_df.dropna(subset=['timestamp'])
infra_df['date'] = infra_df['timestamp'].dt.date
infra_df['latitude'] = infra_df['infra_latitude']
infra_df['longitude'] = infra_df['infra_longitude']
infra_df['zone'] = infra_df['zone'].fillna('Unknown')
facility_map = {
    'fire_station': 'Fire Station',
    'hospital': 'Hospital',
    'shelter': 'Shelter'
}
infra_df['facility'] = infra_df['infrastructure_type'].astype(str).str.lower().map(facility_map).fillna('Unknown')

# ---- Dash App Setup ----
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
app.title = "Crisis Intelligence Dashboard"

app.layout = dbc.Container([
    html.H2("🛡 Crisis Intelligence System", className="my-4 text-primary"),
    html.P("Multi-layered crisis visualization and intelligence", className="text-secondary"),

    dcc.Tabs(id="tabs", value="disasters", children=[
        dcc.Tab(label='🌩 Disaster Incidents', value='disasters'),
        dcc.Tab(label='🏥 Infrastructure Impact', value='infrastructure'),
        dcc.Tab(label='📣 Tweet Intelligence', value='tweets'),
    ], style={"backgroundColor": "#f8f9fa", "borderRadius": "8px", "padding": "10px"}),

    html.Div(id="filters"),
    dcc.Graph(id="map", style={"backgroundColor": "white", "padding": "10px", "borderRadius": "10px"}),
    html.Div(id="summary-cards"),
    html.Div(id="zone-table")
], fluid=True, style={"padding": "20px", "backgroundColor": "#f0f2f5"})

# ---- Filters ----
@app.callback(
    Output("filters", "children"),
    Input("tabs", "value")
)
def update_filters(tab):
    tweet_style = {"display": "block"} if tab == "tweets" else {"display": "none"}
    disaster_style = {"display": "block"} if tab == "disasters" else {"display": "none"}
    infra_style = {"display": "block"} if tab == "infrastructure" else {"display": "none"}

    return dbc.Row([
        dbc.Col([
            html.Label("Tweet Type"),
            dcc.Dropdown(
                id="tweet-select",
                options=[{'label': l, 'value': l} for l in ['All', 'Likely Real', 'Possibly Fake']],
                value="All"
            )
        ], md=3),
        dbc.Col([
            html.Label("Disaster Type"),
            dcc.Dropdown(
                id="disaster-select",
                options=[{'label': d, 'value': d} for d in ['All', 'Flood', 'Fire', 'Earthquake', 'Hurricane']],
                value="All"
            )
        ], md=3),
        dbc.Col([
            html.Label("Infrastructure"),
            dcc.Dropdown(
                id="infra-select",
                options=[{'label': i, 'value': i} for i in ['All', 'Hospital', 'Shelter', 'Fire Station']],
                value="All"
            )
        ], md=3),
        dbc.Col([
            html.Label("Zone"),
            dcc.Dropdown(
                id='zone-select',
                options=[{'label': z, 'value': z} for z in ['All', 'Zone A', 'Zone B', 'Zone C', 'Zone D']],
                value='All'
            )
        ], md=3),
        dbc.Col([
    html.Div([
        html.Label("Date Range"),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=min(tweets['date'].min(), disaster_df['date'].min(), infra_df['date'].min()),
            end_date=max(tweets['date'].max(), disaster_df['date'].max(), infra_df['date'].max()),
            display_format='YYYY-MM-DD',
            style={"width": "100%"}
        )
    ])
], md=3)
], className="my-3")

# ---- Map + Summary + Table ----
@app.callback(
    Output("map", "figure"),
    Output("summary-cards", "children"),
    Output("zone-table", "children"),
    Input("tabs", "value"),
    Input("zone-select", "value"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date"),
    Input("tweet-select", "value"),
    Input("disaster-select", "value"),
    Input("infra-select", "value"),
)
def update_map(tab, zone, start_date, end_date, tweet_type, disaster_type, infra_type):
    cards = []
    table = None

    if tab == "tweets":
        df = tweets.copy()
        if tweet_type != "All":
            df = df[df["label"] == tweet_type]
        if zone != "All":
            df = df[df["zone"] == zone]
        df = df[(df["date"] >= pd.to_datetime(start_date).date()) & (df["date"] <= pd.to_datetime(end_date).date())]

        fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="label", hover_name="text", zoom=8)
        cards = [
    dbc.Card(
        dbc.CardBody([
            html.H5("🧵 Total Tweets", className="card-title"),
            html.H3(len(df), className="card-text")
        ]),
        className="m-2 shadow-sm",
        style={"backgroundColor": "#ffffff", "borderRadius": "12px"}
    ),
    dbc.Card(
        dbc.CardBody([
            html.H5("✅ Likely Real", className="card-title"),
            html.H3((df['label'] == 'Likely Real').sum(), className="card-text")
        ]),
        className="m-2 shadow-sm",
        style={"backgroundColor": "#ffffff", "borderRadius": "12px"}
    ),
    dbc.Card(
        dbc.CardBody([
            html.H5("❌ Possibly Fake", className="card-title"),
            html.H3((df['label'] == 'Possibly Fake').sum(), className="card-text")
        ]),
        className="m-2 shadow-sm",
        style={"backgroundColor": "#ffffff", "borderRadius": "12px"}
    )
]

    elif tab == "disasters":
        df = disaster_df.copy()
        if disaster_type != "All":
            df = df[df["disaster"].str.lower() == disaster_type.lower()]
        if zone != "All":
            df = df[df["zone"] == zone]
        df = df[(df["date"] >= pd.to_datetime(start_date).date()) & (df["date"] <= pd.to_datetime(end_date).date())]

        fig = px.scatter_mapbox(
            df, lat="latitude", lon="longitude", color="disaster", hover_name="disaster",
            hover_data={"disaster": True, "zone": True, "severity": True, "latitude": False, "longitude": False},
            zoom=8
        )

        severity_counts = df['severity'].value_counts()
        cards = [
            dbc.Card(dbc.CardBody([html.H5("🌪️ Total Incidents"), html.H3(len(df))]), className="m-2")
        ] + [
            dbc.Card(dbc.CardBody([html.H5(f"{level}"), html.H3(count)]), className="m-2")
            for level, count in severity_counts.items()
        ]

        zone_summary = df.groupby(["zone", "severity"]).size().unstack(fill_value=0).reset_index()
        table = dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in zone_summary.columns],
            data=zone_summary.to_dict("records"),
            style_table={"margin": "20px", "overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "10px"},
            style_header={"backgroundColor": "#eee", "fontWeight": "bold"},
        )

    else:  # infrastructure
        df = infra_df.copy()
        if infra_type != "All":
            df = df[df["facility"] == infra_type]
        if zone != "All":
            df = df[df["zone"] == zone]
        df = df[(df["date"] >= pd.to_datetime(start_date).date()) & (df["date"] <= pd.to_datetime(end_date).date())]

        fig = px.scatter_mapbox(
            df, lat="latitude", lon="longitude", color="facility", hover_name="name",
            hover_data={"disaster": True, "predicted_impact": True, "recommended_action": True, "latitude": False, "longitude": False},
            zoom=8
        )

        impact_summary = df.groupby(["facility", "predicted_impact"]).size().unstack(fill_value=0).reset_index()
        table = dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in impact_summary.columns],
            data=impact_summary.to_dict("records"),
            style_table={"margin": "20px", "overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "10px"},
            style_header={"backgroundColor": "#eee", "fontWeight": "bold"},
        )

    fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600)
    return fig, dbc.Row(cards, className="d-flex flex-wrap"), table

# ---- Run ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
