import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# --- Core Logic Functions ---

def convert_mv_to_liters(millivolts, config):
    """
    Converts raw millivolt sensor readings to a fuel volume in Liters.
    This function uses a linear conversion based on two calibration points:
    one for an empty tank and one for a full tank.

    Args:
        millivolts (float): The raw sensor reading in millivolts.
        config (dict): The application's configuration dictionary containing calibration data.

    Returns:
        float: The calculated fuel level in Liters. Returns 0 if the tank is empty or below.
    """
    # Retrieve calibration points from the configuration
    mv_empty = config['calibration']['mv_empty']
    liters_empty = config['calibration']['liters_empty']
    mv_full = config['calibration']['mv_full']
    liters_full = config['calibration']['liters_full']
    
    # Avoid division by zero if calibration points are identical
    if (mv_full - mv_empty) == 0:
        return 0
    
    # Calculate the slope of the line (Liters per millivolt)
    slope = (liters_full - liters_empty) / (mv_full - mv_empty)
    
    # Apply the linear formula: y = y1 + m(x - x1)
    fuel_liters = liters_empty + slope * (millivolts - mv_empty)
    
    # Ensure the calculated fuel level cannot be negative
    return max(0, fuel_liters)

def apply_magnitude_filter(df, config):
    """
    Applies the Magnitude Threshold filter.
    This method discards data points unless the change in fuel level from the
    last accepted point exceeds a specified threshold (`Filtration Level`).
    It's effective for ignoring minor fuel sloshing.

    Args:
        df (pd.DataFrame): The DataFrame with the 'fuel_level_liters' column.
        config (dict): The application's configuration dictionary.

    Returns:
        pd.DataFrame: A new DataFrame containing only the significant data points.
    """
    st.write(f"Applying Magnitude Threshold with level: {config['filtration_level']}L")
    filtered_data = []
    # Proceed only if the DataFrame is not empty
    if not df.empty:
        # The first data point is always considered valid
        last_valid_reading = df.iloc[0]
        filtered_data.append(last_valid_reading)

        # Iterate through the rest of the data
        for _, current_reading in df.iloc[1:].iterrows():
            # Calculate the absolute difference from the last saved point
            fuel_diff = abs(current_reading['fuel_level_liters'] - last_valid_reading['fuel_level_liters'])
            
            # If the change is significant, keep the point and update the reference
            if fuel_diff >= config['filtration_level']:
                filtered_data.append(current_reading)
                last_valid_reading = current_reading
    
    return pd.DataFrame(filtered_data).reset_index(drop=True)

def apply_median_filter(df, config):
    """
    Applies a standard rolling Median Filter.
    This method smooths the data by replacing each point with the median of the
    values within a sliding window. It's effective at removing sharp, single-point spikes.

    Args:
        df (pd.DataFrame): The DataFrame with the 'fuel_level_liters' column.
        config (dict): The application's configuration dictionary.

    Returns:
        pd.DataFrame: The DataFrame with the 'fuel_level_liters' column smoothed.
    """
    window_size = config['median_window_size']
    # A median filter requires a window size of at least 3
    if window_size < 3:
        st.write("Median window size is too small. Skipping filter.")
        return df
    # The window size must be an odd number for the median to be centered correctly
    if window_size % 2 == 0:
        window_size += 1
        st.write(f"Window size must be odd. Adjusting to: {window_size}")

    st.write(f"Applying Median Filter with window size: {window_size}")
    
    # Apply the rolling median. `center=True` ensures the output aligns with the original data.
    df['fuel_level_liters'] = df['fuel_level_liters'].rolling(
        window=window_size, center=True, min_periods=1
    ).median()
    
    # Remove any NaN values that might be created at the edges
    df.dropna(subset=['fuel_level_liters'], inplace=True)
    return df.reset_index(drop=True)

def apply_adaptive_median_filter(df, config):
    """
    Applies a true adaptive median filter by dynamically adjusting its window size
    for each data point to remove noise while preserving signal features.

    Args:
        df (pd.DataFrame): The DataFrame with the 'fuel_level_liters' column.
        config (dict): The application's configuration dictionary.

    Returns:
        pd.DataFrame: The DataFrame with the 'fuel_level_liters' column filtered.
    """
    column_name = 'fuel_level_liters'
    min_window = config['adaptive_min_window']
    max_window = config['adaptive_max_window']
    
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
            
            # STAGE A: Check if the median is an outlier itself
            if z_min < z_med < z_max:
                # STAGE B: Median is good, now check the original center point
                center_point_value = padded_data[center_index]
                if not (z_min < center_point_value < z_max):
                    # Center point is the outlier, replace it with the window's median
                    filtered_data[i] = z_med
                # If center point is not an outlier, it keeps its original value from the copy
                break  # Exit while loop, point is processed
            else:
                # Median is an outlier (e.g., z_min or z_max), so expand the window and retry
                current_window_size += 2
                if current_window_size > max_window:
                    # Max window reached, unable to find a non-outlier median.
                    # As a last resort, keep the median of the largest window.
                    filtered_data[i] = z_med
                    break

    df[column_name] = filtered_data
    return df.reset_index(drop=True)

def detect_fuel_events(df, config):
    """
    This is the main analysis function. It orchestrates the pre-processing, filtering,
    and event detection logic.

    Args:
        df (pd.DataFrame): The raw DataFrame from the uploaded file.
        config (dict): The application's configuration dictionary.

    Returns:
        tuple: A tuple containing (list of detected events, processed DataFrame, raw DataFrame).
    """
    st.write(f"Processing data for IMEI: `{df['Imei'].iloc[0]}`...")

    # --- Step 1: Data Preparation and Calibration ---
    # Convert timestamp column to datetime objects for proper sorting and calculations
    df['Dttime_ist'] = pd.to_datetime(df['Dttime_ist'], errors='coerce')
    df.dropna(subset=['Dttime_ist'], inplace=True) # Remove rows with invalid dates
    df = df.sort_values(by='Dttime_ist').reset_index(drop=True)
    
    # Apply the calibration formula to get fuel levels in Liters
    fuel_sensor_column = config['fuel_sensor_column']
    df['fuel_level_liters'] = df[fuel_sensor_column].apply(
        lambda mv: convert_mv_to_liters(mv, config)
    )
    
    # Store a copy of the raw but calibrated data for later comparison on the chart
    raw_calibrated_df = df.copy()

    # --- Step 2: Initial Filtering (The User's Choice) ---
    # Conditionally call the appropriate filtering function based on the user's selection
    if config['filtering_algorithm'] == 'Magnitude Threshold':
        filtered_df = apply_magnitude_filter(df.copy(), config)
    elif config['filtering_algorithm'] == 'Median Filter':
        filtered_df = apply_median_filter(df.copy(), config)
    elif config['filtering_algorithm'] == 'Adaptive Median Filter':
        filtered_df = apply_adaptive_median_filter(df.copy(), config)
    else:
        # If no filter is somehow selected, use the raw data
        filtered_df = df

    # If filtering removed all data, exit early
    if filtered_df.empty:
        return [], pd.DataFrame(), raw_calibrated_df
        
    # --- Step 3: Event Detection & Verification ---
    # This loop iterates through the PROCESSED data to find events.
    all_events = []
    potential_event_info = None # This will store info about a drain/fill we are currently tracking
    time_stationary_seconds = 0 # A running timer for how long the vehicle has been parked

    for i in range(1, len(filtered_df)):
        # Get the previous and current data points for comparison
        prev = filtered_df.iloc[i-1]
        curr = filtered_df.iloc[i]
        
        # Calculate time elapsed since the last point
        time_diff_seconds = (curr['Dttime_ist'] - prev['Dttime_ist']).total_seconds()

        # Update the stationary timer
        if prev['Speed'] == 0:
            time_stationary_seconds += time_diff_seconds
        else:
            time_stationary_seconds = 0 # Reset timer if the vehicle moves

        # Calculate the change in fuel level
        fuel_change = curr['fuel_level_liters'] - prev['fuel_level_liters']
        fuel_drop = -fuel_change
        fuel_fill = fuel_change

        # --- Logic to Identify a NEW Potential Event ---
        # This block only runs if we are not already tracking an event
        if potential_event_info is None:
            # Check if the vehicle has been stationary long enough for a reliable reading
            is_stationary_long_enough = time_stationary_seconds >= config['min_stay_time_before_event']
            is_in_motion_detection_allowed = config['detect_events_in_motion']
            event_type = None
            
            # Check if the drop/fill volume exceeds the configured minimum
            if fuel_drop >= config['min_drain_volume']: event_type = "Drain"
            elif fuel_fill >= config['min_fill_volume']: event_type = "Filling"

            # If we have a significant volume change AND the motion conditions are met...
            if event_type and (is_stationary_long_enough or is_in_motion_detection_allowed):
                # ...we log it as a "potential event" and start the confirmation timer.
                potential_event_info = {
                    "type": event_type, "start_index": i, "start_time": prev['Dttime_ist'],
                    "start_fuel_level": prev['fuel_level_liters'],
                    "start_location": {"latitude": prev['Latitude'], "longitude": prev['Longitude']},
                    "was_stationary": is_stationary_long_enough
                }
        # --- Logic to Confirm or Cancel an EXISTING Potential Event ---
        else:
            start_fuel = potential_event_info['start_fuel_level']
            event_type = potential_event_info['type']
            
            # CANCELLATION condition: Did the fuel level return to normal?
            if abs(curr['fuel_level_liters'] - start_fuel) < config['false_event_threshold']:
                # If so, it was just sloshing. Log as a "False" event.
                event_class = "False"
                volume_change = abs(start_fuel - curr['fuel_level_liters'])
                all_events.append({
                    "Timestamp": potential_event_info['start_time'], "Event": f"{event_class} {event_type}",
                    "Volume (L)": round(volume_change, 2), "Details": "Fuel level returned to normal"
                })
                # Reset and stop tracking this event
                potential_event_info = None
            else:
                # CONFIRMATION condition: Has enough time passed without the fuel level returning?
                start_index = potential_event_info['start_index']
                time_since_event_started = (curr['Dttime_ist'] - filtered_df.iloc[start_index]['Dttime_ist']).total_seconds()
                
                if time_since_event_started >= config['timeout_to_confirm_event']:
                    # If so, the change is persistent. Log as a "True" event.
                    event_class = "True"
                    volume_change = abs(curr['fuel_level_liters'] - start_fuel)
                    all_events.append({
                        "Timestamp": potential_event_info['start_time'], "Event": f"{event_class} {event_type}",
                        "Volume (L)": round(volume_change, 2),
                        "Details": f"Confirmed after {config['timeout_to_confirm_event']}s timeout"
                    })
                    # Reset and stop tracking this event
                    potential_event_info = None

    # Return all results for display
    return all_events, filtered_df, raw_calibrated_df

def process_and_display(df, config, filename):
    """
    A wrapper function that calls the detection logic and then creates all the
    Streamlit UI elements (tables, charts) to display the results.
    """
    st.header(f"Results for: `{filename}`")
    try:
        # Run the main detection logic
        events, processed_df, raw_df = detect_fuel_events(df, config)
        
        # Display the results table only if events were found
        if events:
            st.subheader("Detected Fuel Events")
            st.dataframe(pd.DataFrame(events))
        else:
            st.info("No significant fuel events were detected with the current settings.")
        
        # Display the chart if there is data to plot
        if processed_df is not None and not processed_df.empty:
            st.subheader("Fuel Level Chart")
            fig = go.Figure()
            
            # NEW: Logic to draw colored background rectangles based on motion status
            stoppage_color = "rgba(100, 100, 0, 0.3)"
            if config.get('color_motion_status', False):
                in_stationary_block = False
                block_start_time = None
                # Use the raw dataframe for accurate motion status before any filtering
                for i, row in raw_df.iterrows():
                    # Start of a new stationary block
                    if row['Speed'] == 0 and not in_stationary_block:
                        in_stationary_block = True
                        block_start_time = row['Dttime_ist']
                    # End of a stationary block
                    elif row['Speed'] > 0 and in_stationary_block:
                        fig.add_vrect(
                            x0=block_start_time, x1=row['Dttime_ist'],
                            fillcolor=stoppage_color, layer="below", line_width=0
                        )
                        in_stationary_block = False
                # Handle case where the data ends while still in a stationary block
                if in_stationary_block:
                    fig.add_vrect(
                        x0=block_start_time, x1=raw_df['Dttime_ist'].iloc[-1],
                        fillcolor=stoppage_color, layer="below", line_width=0
                    )


            # Plot the raw (unfiltered) data if the user has checked the box
            if config.get('show_raw_data', False) and raw_df is not None:
                fig.add_trace(go.Scatter(
                    x=raw_df['Dttime_ist'], y=raw_df['fuel_level_liters'],
                    mode='lines', name='Fuel Level (Raw)',
                    line=dict(color='rgba(173, 216, 230, 0.6)', width=1.5, dash='dot')
                ))

            # Plot the main processed (filtered) data
            fig.add_trace(go.Scatter(
                x=processed_df['Dttime_ist'], y=processed_df['fuel_level_liters'],
                mode='lines', name='Fuel Level (Processed)', line=dict(color='blue', width=2)
            ))
            
            # If events were found, plot them as markers on the chart
            if events:
                event_markers = {
                    "True Drain": {"color": "red", "symbol": "triangle-down"},
                    "False Drain": {"color": "orange", "symbol": "triangle-down-open"},
                    "True Filling": {"color": "green", "symbol": "triangle-up"},
                    "False Filling": {"color": "lightgreen", "symbol": "triangle-up-open"}
                }
                for event_type, style in event_markers.items():
                    plot_df = pd.DataFrame(events)[pd.DataFrame(events)['Event'] == event_type]
                    if not plot_df.empty:
                        # Find the closest point in time on the processed data line to place the marker
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

            # Finalize chart layout
            fig.update_layout(
                title=f"Fuel Analysis for IMEI: {df['Imei'].iloc[0]}",
                xaxis_title="Date and Time", yaxis_title="Fuel Level (Liters)",
                legend_title="Events", hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing {filename}: {e}")

# --- Streamlit UI Definition ---
# This part of the script defines the user interface (sidebar, sliders, etc.)

st.set_page_config(layout="wide")
st.title("Wialon Fuel Logic Analyzer")

# Custom CSS for styling the parameter blocks in the sidebar
st.markdown("""
<style>
    .param-block {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .filter-params {
        background-color: #e8f0fe;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar UI Elements ---
with st.sidebar:
    st.header("1. Upload Files")
    uploaded_files = st.file_uploader("Upload one or more Excel files", type="xlsx", accept_multiple_files=True)
    st.header("2. Configure Parameters")
    fuel_sensor_column = "An1" #st.selectbox("Fuel Sensor Column", ["An1", "An2", "An3", "An4"], index=0)
    
    # --- Filtering Parameters Block (Light Blue) ---
    st.divider()
    st.subheader("Filtering Algorithm")
    filtering_algorithm = st.radio(
        "Algorithm",
        (#'Magnitude Threshold', 
         'Median Filter', 
         'Adaptive Median Filter'),
        index=0, label_visibility="collapsed"
    )
    
    # Initialize parameter variables
    filtration_level = 0
    median_window_size = 0
    adaptive_min_window = 0
    adaptive_max_window = 0

    # Conditionally show the relevant sliders for the chosen algorithm
    if filtering_algorithm == 'Magnitude Threshold':
        filtration_level = st.slider("Filtration Level (Liters)", 0.0, 50.0, 5.0, 0.5)
    elif filtering_algorithm == 'Median Filter':
        median_window_size = st.slider("Window Size", 3, 101, 15, 2, help="Must be an odd number.")
    elif filtering_algorithm == 'Adaptive Median Filter':
        adaptive_min_window = st.slider("Min Window Size", 3, 21, 3, 2)
        adaptive_max_window = st.slider("Max Window Size", 5, 101, 11, 2)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Event Detection Parameters Block (Gray) ---
    st.divider()
    st.subheader("Event Detection Tuning")
    min_drain_volume = st.slider("Minimum Drain Volume (Liters)", 0.0, 50.0, 10.0, 0.5)
    min_fill_volume = st.slider("Minimum Filling Volume (Liters)", 0.0, 50.0, 10.0, 0.5)
    min_stay_time = st.slider("Min Stationary Time (seconds)", 0, 600, 300)
    timeout = st.slider("Confirmation Timeout (seconds)", 0, 600, 180)
    false_event_threshold = st.slider("False Event Threshold (Liters)", 0.0, 10.0, 2.0, 0.5, help="Threshold to cancel a potential event if fuel level returns to normal.")
    detect_in_motion = st.checkbox("Detect Events in Motion", value=False)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("Chart Options")
    show_raw_data = st.checkbox("Show Raw Fuel Data on Chart", value=True)
    color_motion_status = st.checkbox("Highlight Parkings", value=False)


# --- Main Application Body ---
# This part runs the analysis based on the UI settings.

# Gather all configurations into a single dictionary to pass to functions
config = {
    "fuel_sensor_column": fuel_sensor_column,
    "filtering_algorithm": filtering_algorithm,
    "calibration": {"mv_empty": 173, "liters_empty": 0, "mv_full": 5062, "liters_full": 375},
    "filtration_level": filtration_level,
    "median_window_size": median_window_size,
    "adaptive_min_window": adaptive_min_window,
    "adaptive_max_window": adaptive_max_window,
    "min_drain_volume": min_drain_volume,
    "min_fill_volume": min_fill_volume,
    "min_stay_time_before_event": min_stay_time,
    "timeout_to_confirm_event": timeout,
    "false_event_threshold": false_event_threshold,
    "detect_events_in_motion": detect_in_motion,
    "show_raw_data": show_raw_data,
    "color_motion_status": color_motion_status,
}

# Main logic: Process uploaded files or a default file
if uploaded_files:
    # If files are uploaded, loop through each one
    for uploaded_file in uploaded_files:
        df = pd.read_excel(uploaded_file)
        process_and_display(df, config, uploaded_file.name)
else:
    # If no files are uploaded, try to use a local default file
    try:
        st.info("No files uploaded. Attempting to load default file 'table.xlsx'...")
        default_df = pd.read_excel("table.xlsx")
        process_and_display(default_df, config, "table.xlsx")
    except FileNotFoundError:
        st.warning("Default file 'table.xlsx' not found.")
        st.info("Please upload Excel files using the sidebar to begin analysis.")
    except Exception as e:
        st.error(f"An error occurred while loading the default file: {e}")
