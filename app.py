import streamlit as st
import pandas as pd
import numpy as np

# Import the main controller function from our new OOP-based utils file
from tools_realtime import process_and_display, caliberate

CALIBERATION_SCOPE = (375.0 - 0.0) / (5062.0 - 173.0)
CALIBERATION_INTERCEPT = 0.0 - CALIBERATION_SCOPE * 173.0

# --- NEW: Initialize a list to hold all events from the session ---
if 'session_events' not in st.session_state:
    st.session_state.session_events = []
    
# --- Streamlit UI Definition ---
st.set_page_config(layout="wide")
st.title("Wialon Fuel Logic Analyzer (OOP Version)")

# --- Initialize Session State for Calibration ---
# This ensures that our calibration parameters persist across reruns.
if 'calibration_params' not in st.session_state:
    # Set default parameters initially. These are based on a linear fit
    # between two common calibration points: (173mV, 0L) and (5062mV, 375L).
    
    st.session_state.calibration_params = {
        "slope": CALIBERATION_SCOPE,
        "intercept": CALIBERATION_INTERCEPT
    }

# --- Sidebar UI Elements ---
with st.sidebar:
    st.header("1. Upload Files")
    
    # --- Calibration File Uploader ---
    calibration_file = st.file_uploader(
        "Upload Calibration File (Optional)",
        type=["xlsx", "csv"],
        help="Must contain 'Voltage' and 'Fuel' columns. All rows will be used for linear regression."
    )

    # --- Logic to parse the calibration file and perform linear regression ---
    st.session_state.calibration_params['slope'], st.session_state.calibration_params['intercept'] = caliberate(calibration_file) 
    if st.session_state.calibration_params['slope'] is None or st.session_state.calibration_params['intercept'] is None:
        st.session_state.calibration_params['slope'] = CALIBERATION_SCOPE
        st.session_state.calibration_params['intercept'] = CALIBERATION_INTERCEPT
    
    # --- Main Data File Uploader ---
    uploaded_files = st.file_uploader(
        "Upload one or more Raw Sensor Data files", 
        type=["xlsx", "csv"], 
        accept_multiple_files=True
    )

    st.header("2. Configure Parameters")

    # This is a fixed column name for the sensor data
    fuel_sensor_column = "An1" 
    
    st.divider()
    st.subheader("Filtering Algorithm")
    # Note: 'Magnitude Threshold' is now a valid option
    filtering_algorithm = st.radio(
        "Algorithm",
        ('Magnitude Threshold', 
         'Median Filter', 'Adaptive Median Filter'),
        index=1, label_visibility="collapsed"
    )
    
    # Initialize parameter variables to zero
    filtration_level = 0
    median_window_size = 0
    adaptive_min_window = 0
    adaptive_max_window = 0

    # Conditionally show sliders based on the chosen algorithm
    if filtering_algorithm == 'Magnitude Threshold':
        filtration_level = st.slider("Filtration Level (Liters)", 0.0, 50.0, 5.0, 0.5)
    elif filtering_algorithm == 'Median Filter':
        median_window_size = st.slider("Window Size", 3, 101, 5, 2, help="Must be an odd number.")
    elif filtering_algorithm == 'Adaptive Median Filter':
        adaptive_min_window = st.slider("Min Window Size", 3, 21, 3, 2, help="Must be an odd number.")
        adaptive_max_window = st.slider("Max Window Size", 5, 101, 11, 2, help="Must be an odd number.")
    
    st.divider()
    st.subheader("Event Detection Tuning")
    min_drain_volume = st.slider("Minimum Drain Volume (Liters)", 0.0, 50.0, 10.0, 0.5)
    min_fill_volume = st.slider("Minimum Filling Volume (Liters)", 0.0, 50.0, 10.0, 0.5)
    min_stay_time = st.slider("Min Stationary Time (seconds)", 0, 600, 100)
    timeout = st.slider("Confirmation Timeout (seconds)", 0, 600, 180, help="Time the fuel level must stabilize to confirm an event's end.")
    false_event_threshold = st.slider("False Event Threshold (Liters)", 0.0, 10.0, 2.0, 0.5, help="Threshold to cancel an event if fuel level returns to normal.")
    detect_in_motion = st.checkbox("Detect Events in Motion", value=False)
    
    st.subheader("Chart Options")
    show_raw_data = st.checkbox("Show Raw Fuel Data on Chart", value=True)

# --- Main Application Body ---
# Gather all configurations from the UI into a single dictionary
config = {
    "fuel_sensor_column": fuel_sensor_column,
    "filtering_algorithm": filtering_algorithm,
    "calibration": st.session_state.calibration_params,
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
}

# --- Main Logic: Process uploaded files or a default file ---
if uploaded_files:
    # Clear previous session results when new files are uploaded
    st.session_state.session_events = []
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            # process_and_display now returns the events from the file
            file_events = process_and_display(df, config, uploaded_file.name)
            
            # If events were found, add them to our session list
            if file_events:
                events_df = pd.DataFrame(file_events)
                # Add a column to identify the source file for each event
                events_df['Source File'] = uploaded_file.name
                st.session_state.session_events.append(events_df)
        except Exception as e:
            st.error(f"Could not process file: {uploaded_file.name}. Error: {e}")
else:
    st.info("Please upload one or more data files using the sidebar to begin analysis.")


# --- NEW: Display Download button if any events were found in the session ---
if st.session_state.session_events:
    st.sidebar.divider()
    st.sidebar.header("3. Download Combined Report")

    # Combine all collected event DataFrames into one
    combined_df = pd.concat(st.session_state.session_events, ignore_index=True)

    # Use a helper function to convert the DataFrame to CSV bytes
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(combined_df)

    st.sidebar.download_button(
       label="Download All Events as CSV",
       data=csv_data,
       file_name="combined_fuel_events.csv",
       mime="text/csv",
    )
    # Also display the combined table in the sidebar for a quick preview
    st.sidebar.subheader("Combined Preview")
    st.sidebar.dataframe(combined_df)
