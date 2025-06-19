import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

from utils_filling_interval import process_and_display

# --- Streamlit UI Definition ---
st.set_page_config(layout="wide")
st.title("Wialon Fuel Logic Analyzer")

# --- Initialize Session State ---
# This ensures that our calibration parameters persist across reruns.
if 'calibration_params' not in st.session_state:
    # Set default parameters initially by performing a regression on the default points.
    # Default points: (173, 0) and (5062, 375)
    slope = (375.0 - 0.0) / (5062.0 - 173.0)
    intercept = 0.0 - slope * 173.0
    st.session_state.calibration_params = {
        "slope": slope,
        "intercept": intercept
    }

# --- Sidebar UI Elements ---
with st.sidebar:
    st.header("1. Upload Files")
    
        
    # --- NEW: Calibration File Uploader ---
    calibration_file = st.file_uploader(
        "Upload Calibration File (Optional)",
        type=["xlsx", "csv"],
        help="Must contain 'Voltage' and 'Fuel' columns. All rows will be used for linear regression."
    )

    # --- NEW: Logic to parse the calibration file and perform linear regression ---
    if calibration_file is not None:
        try:
            # Try reading as CSV, then fall back to Excel
            try:
                cal_df = pd.read_csv(calibration_file)
            except Exception:
                calibration_file.seek(0)
                cal_df = pd.read_excel(calibration_file)

            # Ensure the required columns exist and there's enough data for regression
            if 'Voltage' in cal_df.columns and 'Fuel' in cal_df.columns and len(cal_df) >= 2:
                # Perform linear regression using numpy
                x = cal_df['Voltage'].astype(float)
                y = cal_df['Fuel'].astype(float)
                slope, intercept = np.polyfit(x, y, 1)
                
                # Update the session state with the calculated parameters
                st.session_state.calibration_params['slope'] = slope
                st.session_state.calibration_params['intercept'] = intercept
                st.success("Calibration complete!")
            else:
                st.error("Calibration file must have 'Voltage' and 'Fuel' columns with at least 2 data points.")
        except Exception as e:
            st.error(f"Error reading calibration file: {e}")

    # Uploader for the main vehicle data file
    uploaded_files = st.file_uploader("Upload one or more Raw Sensor Data Excel files", type="xlsx", accept_multiple_files=True)

    st.header("2. Configure Parameters")
    
    # --- NEW: Display for active calibration parameters ---
    st.subheader("Sensor Calibration")
    st.info(f"Using Slope (L/mV): `{st.session_state.calibration_params['slope']:.4f}`\n\n"
            f"Using Intercept (L): `{st.session_state.calibration_params['intercept']:.2f}`")


    fuel_sensor_column = "An1" 
    
    st.divider()
    st.subheader("Filtering Algorithm")
    filtering_algorithm = st.radio(
        "Algorithm",
        (
            #'Magnitude Threshold', 
            'Median Filter', 
            'Adaptive Median Filter'),
        index=0, label_visibility="collapsed"
    )
    
    filtration_level = 0
    median_window_size = 0
    adaptive_min_window = 0
    adaptive_max_window = 0

    if filtering_algorithm == 'Magnitude Threshold':
        filtration_level = st.slider("Filtration Level (Liters)", 0.0, 50.0, 5.0, 0.5)
    elif filtering_algorithm == 'Median Filter':
        median_window_size = st.slider("Window Size", 3, 101, 5, 2, help="Must be an odd number.")
    elif filtering_algorithm == 'Adaptive Median Filter':
        adaptive_min_window = st.slider("Min Window Size", 3, 21, 3, 2)
        adaptive_max_window = st.slider("Max Window Size", 5, 101, 11, 2)
    
    st.divider()
    st.subheader("Event Detection Tuning")
    min_drain_volume = st.slider("Minimum Drain Volume (Liters)", 0.0, 50.0, 10.0, 0.5)
    min_fill_volume = st.slider("Minimum Filling Volume (Liters)", 0.0, 50.0, 10.0, 0.5)
    min_stay_time = st.slider("Min Stationary Time (seconds)", 0, 600, 100)
    timeout = st.slider("Confirmation Timeout (seconds)", 0, 600, 180)
    false_event_threshold = st.slider("False Event Threshold (Liters)", 0.0, 10.0, 2.0, 0.5)
    ignore_time_before_filling = st.slider("Ignore Dips Before Filling (sec)", 0, 600, 300, 60)
    detect_in_motion = st.checkbox("Detect Events in Motion", value=False)
    
    st.subheader("Chart Options")
    show_raw_data = st.checkbox("Show Raw Fuel Data on Chart", value=True)
    color_motion_status = st.checkbox("Highlight Parkings", value=False)


# --- Main Application Body ---
# Gather all configurations into a single dictionary to pass to functions
config = {
    "fuel_sensor_column": fuel_sensor_column,
    "filtering_algorithm": filtering_algorithm,
    "calibration": st.session_state.calibration_params, # Use the dynamic calibration data from session state
    "filtration_level": filtration_level,
    "median_window_size": median_window_size,
    "adaptive_min_window": adaptive_min_window,
    "adaptive_max_window": adaptive_max_window,
    "min_drain_volume": min_drain_volume,
    "min_fill_volume": min_fill_volume,
    "min_stay_time_before_event": min_stay_time,
    "timeout_to_confirm_event": timeout,
    "false_event_threshold": false_event_threshold,
    "ignore_time_before_filling": ignore_time_before_filling,
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
