import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# --- Core Logic Functions ---

# convert_mv_to_liters, apply_magnitude_filter, apply_median_filter, 
# and apply_adaptive_median_filter functions remain the same as you provided.
# I am including them here for a complete, runnable file.

def convert_mv_to_liters(millivolts, config):
    slope = config['calibration']['slope']
    intercept = config['calibration']['intercept']
    fuel_liters = (millivolts * slope) + intercept
    return max(0, fuel_liters)

def apply_magnitude_filter(df, config):
    st.write(f"Applying Magnitude Threshold with level: {config['filtration_level']}L")
    filtered_data = []
    if not df.empty:
        last_valid_reading = df.iloc[0]
        filtered_data.append(last_valid_reading)
        for _, current_reading in df.iloc[1:].iterrows():
            fuel_diff = abs(current_reading['fuel_level_liters'] - last_valid_reading['fuel_level_liters'])
            if fuel_diff >= config['filtration_level']:
                filtered_data.append(current_reading)
                last_valid_reading = current_reading
    return pd.DataFrame(filtered_data).reset_index(drop=True)

def apply_median_filter(df, config):
    window_size = config['median_window_size']
    if window_size < 3: return df
    if window_size % 2 == 0: window_size += 1
    st.write(f"Applying Median Filter with window size: {window_size}")
    df['fuel_level_liters'] = df['fuel_level_liters'].rolling(window=window_size, center=True, min_periods=1).median()
    df.dropna(subset=['fuel_level_liters'], inplace=True)
    return df.reset_index(drop=True)

def apply_adaptive_median_filter(df, config):
    # This function remains as you provided it.
    column_name = 'fuel_level_liters'
    min_window, max_window = config['adaptive_min_window'], config['adaptive_max_window']
    st.write(f"Applying True Adaptive Median Filter with min/max windows: {min_window}/{max_window}")
    if min_window % 2 == 0: min_window += 1
    if max_window % 2 == 0: max_window += 1
    data = df[column_name].to_numpy()
    pad_size = max_window // 2
    padded_data = np.pad(data, (pad_size, pad_size), mode='reflect')
    filtered_data = np.copy(data)
    for i in range(len(data)):
        center_index = i + pad_size
        current_window_size = min_window
        while current_window_size <= max_window:
            half_window = current_window_size // 2
            window = padded_data[center_index - half_window : center_index + half_window + 1]
            z_min, z_max, z_med = np.min(window), np.max(window), np.median(window)
            if z_min < z_med < z_max:
                center_point_value = padded_data[center_index]
                if not (z_min < center_point_value < z_max):
                    filtered_data[i] = z_med
                break
            else:
                current_window_size += 2
                if current_window_size > max_window:
                    filtered_data[i] = z_med
                    break
    df[column_name] = filtered_data
    return df.reset_index(drop=True)

# --- NEW `detect_fuel_events` with Start/End Confirmation ---
def detect_fuel_events(df, config):
    """
    Analyzes processed fuel data to detect and verify fuel events (Drains and Fillings).
    This version identifies a confirmed start and end time for each event by checking
    for fuel level stabilization.
    """
    st.write(f"Processing data for IMEI: `{df['Imei'].iloc[0]}`...")

    # --- Step 1: Data Preparation and Calibration ---
    df['Dttime_ist'] = pd.to_datetime(df['Dttime_ist'], errors='coerce')
    df.dropna(subset=['Dttime_ist'], inplace=True)
    df = df.sort_values(by='Dttime_ist').reset_index(drop=True)
    
    fuel_sensor_column = config['fuel_sensor_column']
    df['fuel_level_liters'] = df[fuel_sensor_column].apply(lambda mv: convert_mv_to_liters(mv, config))
    df['fuel_level_liters'] = df['fuel_level_liters'].replace(0, np.nan).ffill()

    raw_calibrated_df = df.copy()

    # --- Step 2: Filtering ---
    if config['filtering_algorithm'] == 'Median Filter':
        filtered_df = apply_median_filter(df.copy(), config)
    elif config['filtering_algorithm'] == 'Adaptive Median Filter':
        filtered_df = apply_adaptive_median_filter(df.copy(), config)
    else:
        # Default to magnitude or no filter if another type is selected
        filtered_df = apply_magnitude_filter(df.copy(), config)


    if filtered_df.empty:
        return [], pd.DataFrame(), raw_calibrated_df
        
    # --- Step 3: Event Detection & Verification ---
    all_events = []
    potential_event = None
    time_stationary_seconds = 0

    for i in range(1, len(filtered_df)):
        prev = filtered_df.iloc[i - 1]
        curr = filtered_df.iloc[i]
        
        time_diff_seconds = (curr['Dttime_ist'] - prev['Dttime_ist']).total_seconds()
        time_stationary_seconds = time_stationary_seconds + time_diff_seconds if prev['Speed'] == 0 else 0
        fuel_change = curr['fuel_level_liters'] - prev['fuel_level_liters']

        if potential_event is None:
            is_stationary_long_enough = time_stationary_seconds >= config['min_stay_time_before_event']
            can_detect_event = is_stationary_long_enough or config['detect_events_in_motion']
            
            event_type = None
            if fuel_change >= config['min_fill_volume']:
                event_type = "Filling"
            elif -fuel_change >= config['min_drain_volume']:
                event_type = "Drain"

            if event_type and can_detect_event:
                potential_event = {
                    "type": event_type,
                    "start_time": prev['Dttime_ist'],
                    "start_fuel": prev['fuel_level_liters'],
                    "last_change_time": curr['Dttime_ist'],
                    "potential_end_fuel": curr['fuel_level_liters'],
                }
        else:
            if abs(curr['fuel_level_liters'] - potential_event['start_fuel']) < config['false_event_threshold']:
                potential_event = None
                continue

            is_level_changing = abs(curr['fuel_level_liters'] - potential_event['potential_end_fuel']) > 1.0
            if is_level_changing:
                potential_event['potential_end_fuel'] = curr['fuel_level_liters']
                potential_event['last_change_time'] = curr['Dttime_ist']

            time_since_last_change = (curr['Dttime_ist'] - potential_event['last_change_time']).total_seconds()
            
            if time_since_last_change >= config['timeout_to_confirm_event']:
                start_fuel = potential_event['start_fuel']
                end_fuel = potential_event['potential_end_fuel']
                volume_change = abs(end_fuel - start_fuel)
                
                min_volume = config['min_fill_volume'] if potential_event['type'] == 'Filling' else config['min_drain_volume']
                
                if volume_change >= min_volume:
                    all_events.append({
                        "Event": potential_event['type'],
                        "Start Time": potential_event['start_time'],
                        "End Time": potential_event['last_change_time'],
                        "Start Fuel (L)": round(start_fuel, 2),
                        "End Fuel (L)": round(end_fuel, 2),
                        "Volume (L)": round(volume_change, 2),
                        "Details": f"Confirmed after {time_since_last_change:.0f}s stabilization."
                    })

                potential_event = None

    return all_events, filtered_df, raw_calibrated_df


# --- FIXED `process_and_display` to show Event Start/End ---
def process_and_display(df, config, filename):
    """
    A wrapper function that calls the detection logic and then creates all the
    Streamlit UI elements (tables, charts) to display the results.
    This version is updated to display events with confirmed start and end times.
    """
    st.header(f"Results for: `{filename}`")
    try:
        events, processed_df, raw_df = detect_fuel_events(df, config)
        
        if events:
            st.subheader("Detected Fuel Events")
            events_df = pd.DataFrame(events)
            st.dataframe(events_df)
        else:
            st.info("No significant fuel events were detected with the current settings.")
        
        if processed_df is not None and not processed_df.empty:
            st.subheader("Fuel Level Chart")
            fig = go.Figure()

            if config.get('show_raw_data', False) and raw_df is not None:
                fig.add_trace(go.Scatter(
                    x=raw_df['Dttime_ist'], y=raw_df['fuel_level_liters'],
                    mode='lines', name='Fuel Level (Raw)',
                    line=dict(color='rgba(173, 216, 230, 0.6)', width=1.5, dash='dot')
                ))

            fig.add_trace(go.Scatter(
                x=processed_df['Dttime_ist'], y=processed_df['fuel_level_liters'],
                mode='lines', name='Fuel Level (Processed)', line=dict(color='blue', width=2)
            ))
            
            if events:
                events_df = pd.DataFrame(events)
                for _, event in events_df.iterrows():
                    event_type = event['Event']
                    start_time = event['Start Time']
                    end_time = event['End Time']
                    volume = event['Volume (L)']
                    
                    if event_type == 'Filling':
                        color = "rgba(0, 255, 0, 0.2)" # Light Green
                        name = f"Filling (+{volume:.2f} L)"
                    else:
                        color = "rgba(255, 0, 0, 0.2)" # Light Red
                        name = f"Drain (-{volume:.2f} L)"

                    fig.add_vrect(
                        x0=start_time,
                        x1=end_time,
                        fillcolor=color,
                        layer="below",
                        line_width=0,
                        annotation_text=name,
                        annotation_position="top left"
                    )

            fig.update_layout(
                title=f"Fuel Analysis for IMEI: {df['Imei'].iloc[0]}",
                xaxis_title="Date and Time", yaxis_title="Fuel Level (Liters)",
                legend_title="Data Series",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing {filename}: {e}")