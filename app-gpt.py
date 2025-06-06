import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

# Configuration
config = {
    'fuel_sensor_column': 'An1',
    'mv_empty': 173,
    'liters_empty': 0,
    'mv_full': 5062,
    'liters_full': 375,
    'filtration_level': 5,
    'min_drain_volume': 10,
    'min_fill_volume': 10
}

# Convert millivolts to liters
def convert_mv_to_liters(mv):
    slope = (config['liters_full'] - config['liters_empty']) / (config['mv_full'] - config['mv_empty'])
    liters = config['liters_empty'] + slope * (mv - config['mv_empty'])
    return max(0, min(liters, config['liters_full']))

# Identify fuel events
def detect_events(df):
    df['Dttime_ist'] = pd.to_datetime(df['Dttime_ist'])
    df = df.sort_values(by='Dttime_ist').reset_index(drop=True)
    df['fuel_liters'] = df[config['fuel_sensor_column']].apply(convert_mv_to_liters)

    df['fuel_diff'] = df['fuel_liters'].diff()
    events = []

    for i, row in df.iterrows():
        if abs(row['fuel_diff']) >= config['min_drain_volume']:
            event_type = 'Drain' if row['fuel_diff'] < 0 else 'Filling'
            stationary = df.at[i-1, 'Speed'] < 2 if i > 0 else False
            events.append({
                'time': row['Dttime_ist'],
                'type': event_type,
                'volume': abs(round(row['fuel_diff'], 2)),
                'stationary': stationary,
                'latitude': row['Latitude'],
                'longitude': row['Longitude']
            })

    return pd.DataFrame(events)

# Streamlit UI
st.title("Fuel Event Analyzer")
uploaded_files = st.file_uploader("Upload Excel Files", accept_multiple_files=True)

if uploaded_files:
    all_events = pd.DataFrame()
    for uploaded_file in uploaded_files:
        df = pd.read_excel(uploaded_file)
        events_df = detect_events(df)
        events_df['source_file'] = uploaded_file.name
        all_events = pd.concat([all_events, events_df])
else:
    df = pd.read_excel('table.xlsx')
    all_events = detect_events(df)

st.dataframe(all_events)

fig = px.scatter_mapbox(
    all_events,
    lat="latitude",
    lon="longitude",
    color="type",
    size="volume",
    hover_name="time",
    zoom=10,
    mapbox_style="open-street-map"
)

st.plotly_chart(fig, use_container_width=True)