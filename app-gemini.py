import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

# --- Core Logic Functions (Adapted from previous script) ---

def convert_mv_to_liters(millivolts, config):
    """Converts raw millivolt sensor readings to liters based on linear calibration."""
    mv1 = config['calibration']['mv_empty']
    ltr1 = config['calibration']['liters_empty']
    mv2 = config['calibration']['mv_full']
    ltr2 = config['calibration']['liters_full']
    
    if (mv2 - mv1) == 0:
        return 0
    slope = (ltr2 - ltr1) / (mv2 - mv1)
    fuel_liters = ltr1 + slope * (millivolts - mv1)
    return max(0, fuel_liters)

def detect_fuel_events(df, config):
    """
    Main function to process a dataframe and detect all fuel events.
    Identifies: True Drains, False Drains, True Fillings, False Fillings.
    """
    st.write(f"Processing data for IMEI: `{df['Imei'].iloc[0]}`...")

    # --- Pre-processing ---
    df['Dttime_ist'] = pd.to_datetime(df['Dttime_ist'], errors='coerce')
    df.dropna(subset=['Dttime_ist'], inplace=True)
    df = df.sort_values(by='Dttime_ist').reset_index(drop=True)
    
    fuel_sensor_column = config['fuel_sensor_column']
    df['fuel_level_liters'] = df[fuel_sensor_column].apply(
        lambda mv: convert_mv_to_liters(mv, config)
    )

    # --- Initial Filtering ---
    filtered_data = []
    if not df.empty:
        last_valid_reading = df.iloc[0]
        filtered_data.append(last_valid_reading)
        for _, current_reading in df.iloc[1:].iterrows():
            if abs(current_reading['fuel_level_liters'] - last_valid_reading['fuel_level_liters']) >= config['filtration_level']:
                filtered_data.append(current_reading)
                last_valid_reading = current_reading
    if not filtered_data:
        st.write("No data left after initial filtering.")
        return [], pd.DataFrame()
        
    filtered_df = pd.DataFrame(filtered_data).reset_index(drop=True)

    # --- Advanced Event Detection Loop ---
    all_events = []
    potential_event_info = None
    time_stationary_seconds = 0

    for i in range(1, len(filtered_df)):
        prev = filtered_df.iloc[i-1]
        curr = filtered_df.iloc[i]
        
        time_diff_seconds = (curr['Dttime_ist'] - prev['Dttime_ist']).total_seconds()

        if prev['Speed'] == 0:
            time_stationary_seconds += time_diff_seconds
        else:
            time_stationary_seconds = 0

        fuel_change = curr['fuel_level_liters'] - prev['fuel_level_liters']
        fuel_drop = -fuel_change
        fuel_fill = fuel_change

        # --- Check for new potential events (Drains or Fillings) ---
        if potential_event_info is None:
            is_stationary_long_enough = time_stationary_seconds >= config['min_stay_time_before_event']
            is_in_motion_detection_allowed = config['detect_events_in_motion']
            
            event_type = None
            
            if fuel_drop >= config['min_drain_volume']:
                event_type = "Drain"
            elif fuel_fill >= config['min_fill_volume']:
                event_type = "Filling"

            if event_type and (is_stationary_long_enough or is_in_motion_detection_allowed):
                potential_event_info = {
                    "type": event_type,
                    "start_index": i,
                    "start_time": prev['Dttime_ist'],
                    "start_fuel_level": prev['fuel_level_liters'],
                    "start_location": {"latitude": prev['Latitude'], "longitude": prev['Longitude']},
                    "was_stationary": is_stationary_long_enough
                }

        # --- If tracking a potential event, decide its fate ---
        if potential_event_info:
            start_fuel = potential_event_info['start_fuel_level']
            event_type = potential_event_info['type']
            
            if abs(curr['fuel_level_liters'] - start_fuel) < config['filtration_level']:
                event_class = "False"
                volume_change = abs(start_fuel - curr['fuel_level_liters'])
                all_events.append({
                    "Timestamp": potential_event_info['start_time'],
                    "Event": f"{event_class} {event_type}",
                    "Volume (L)": round(volume_change, 2),
                    "Details": "Fuel level returned to normal"
                })
                potential_event_info = None
            else:
                start_index = potential_event_info['start_index']
                time_since_event_started = (curr['Dttime_ist'] - filtered_df.iloc[start_index]['Dttime_ist']).total_seconds()
                
                if time_since_event_started >= config['timeout_to_confirm_event']:
                    event_class = "True"
                    volume_change = abs(curr['fuel_level_liters'] - start_fuel)
                    all_events.append({
                        "Timestamp": potential_event_info['start_time'],
                        "Event": f"{event_class} {event_type}",
                        "Volume (L)": round(volume_change, 2),
                        "Details": f"Confirmed after {config['timeout_to_confirm_event']}s timeout"
                    })
                    potential_event_info = None

    return all_events, df

def process_and_display(df, config, filename):
    """Encapsulates the full analysis and display logic for a given dataframe."""
    st.header(f"Results for: `{filename}`")
    try:
        events, processed_df = detect_fuel_events(df, config)
        
        # Display table only if events are found, otherwise show an info message.
        if events:
            st.subheader("Detected Fuel Events")
            events_df = pd.DataFrame(events)
            st.dataframe(events_df)
        else:
            st.info("No significant fuel events were detected with the current settings.")
        
        # Always plot the chart if there is data to process.
        if processed_df is not None and not processed_df.empty:
            st.subheader("Fuel Level Chart")
            fig = go.Figure()

            # Plot the main fuel level line
            fig.add_trace(go.Scatter(
                x=processed_df['Dttime_ist'],
                y=processed_df['fuel_level_liters'],
                mode='lines', name='Fuel Level', line=dict(color='blue', width=2)
            ))
            
            # Add markers only if events were detected
            if events:
                event_markers = {
                    "True Drain": {"color": "red", "symbol": "triangle-down"},
                    "False Drain": {"color": "orange", "symbol": "triangle-down-open"},
                    "True Filling": {"color": "green", "symbol": "triangle-up"},
                    "False Filling": {"color": "lightgreen", "symbol": "triangle-up-open"}
                }

                for event_type, style in event_markers.items():
                    plot_df = events_df[events_df['Event'] == event_type]
                    if not plot_df.empty:
                        merged_df = pd.merge_asof(
                            plot_df.sort_values('Timestamp'), 
                            processed_df[['Dttime_ist', 'fuel_level_liters']].sort_values('Dttime_ist'),
                            left_on='Timestamp', right_on='Dttime_ist', direction='nearest'
                        )
                        fig.add_trace(go.Scatter(
                            x=merged_df['Timestamp'], y=merged_df['fuel_level_liters'],
                            mode='markers', name=event_type,
                            marker=dict(color=style['color'], size=12, symbol=style['symbol'], line=dict(width=2, color='DarkSlateGrey')),
                            hoverinfo='text',
                            text=[f"{row['Event']}<br>{row['Volume (L)']} L" for _, row in merged_df.iterrows()]
                        ))

            fig.update_layout(
                title=f"Fuel Analysis for IMEI: {df['Imei'].iloc[0]}",
                xaxis_title="Date and Time", yaxis_title="Fuel Level (Liters)",
                legend_title="Events", hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing {filename}: {e}")

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Wialon Fuel Logic Analyzer")

with st.sidebar:
    st.header("1. Upload Files")
    uploaded_files = st.file_uploader("Upload one or more Excel files", type="xlsx", accept_multiple_files=True)
    st.header("2. Configure Parameters")
    fuel_sensor_column = st.selectbox("Fuel Sensor Column", ["An1", "An2", "An3", "An4"], index=0)
    st.subheader("Algorithm Tuning")
    filtration_level = st.slider("Filtration Level (Liters)", 0.0, 50.0, 5.0, 0.5)
    min_drain_volume = st.slider("Minimum Drain Volume (Liters)", 0.0, 50.0, 10.0, 0.5)
    min_fill_volume = st.slider("Minimum Filling Volume (Liters)", 0.0, 50.0, 10.0, 0.5)
    min_stay_time = st.slider("Min Stationary Time (seconds)", 0, 600, 300)
    timeout = st.slider("Confirmation Timeout (seconds)", 0, 600, 180)
    detect_in_motion = st.checkbox("Detect Events in Motion", value=False)

# --- Main App Body ---
config = {
    "fuel_sensor_column": fuel_sensor_column,
    "calibration": {"mv_empty": 173, "liters_empty": 0, "mv_full": 5062, "liters_full": 375},
    "filtration_level": filtration_level,
    "min_drain_volume": min_drain_volume,
    "min_fill_volume": min_fill_volume,
    "min_stay_time_before_event": min_stay_time,
    "timeout_to_confirm_event": timeout,
    "detect_events_in_motion": detect_in_motion
}

if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_excel(uploaded_file)
        process_and_display(df, config, uploaded_file.name)
else:
    try:
        st.info("No files uploaded. Attempting to load default file 'table.xlsx'...")
        default_df = pd.read_excel("table.xlsx")
        process_and_display(default_df, config, "table.xlsx")
    except FileNotFoundError:
        st.warning("Default file 'table.xlsx' not found.")
        st.info("Please upload Excel files using the sidebar to begin analysis.")
    except Exception as e:
        st.error(f"An error occurred while loading the default file: {e}")

