import pandas as pd
from datetime import timedelta

def convert_mv_to_liters(millivolts, config):
    """
    Converts raw millivolt sensor readings to liters based on linear calibration.
    """
    mv1 = config['calibration']['mv_empty']
    ltr1 = config['calibration']['liters_empty']
    mv2 = config['calibration']['mv_full']
    ltr2 = config['calibration']['liters_full']
    
    if (mv2 - mv1) == 0:
        return 0

    slope = (ltr2 - ltr1) / (mv2 - mv1)
    fuel_liters = ltr1 + slope * (millivolts - mv1)
    return max(0, fuel_liters)

def detect_fuel_drains_advanced(df, config):
    """
    Main function to process the dataframe and detect fuel drains using
    Wialon's advanced time-based and motion-based logic.
    """
    print("Starting ADVANCED fuel drain detection process...")

    # --- Pre-processing Step ---
    # FIX: Use errors='coerce' to handle any invalid date formats gracefully.
    # This will turn unparseable dates into NaT (Not a Time).
    df['Dttime_ist'] = pd.to_datetime(df['Dttime_ist'], errors='coerce')

    # FIX: Remove rows where the date could not be parsed.
    initial_rows = len(df)
    df.dropna(subset=['Dttime_ist'], inplace=True)
    if len(df) < initial_rows:
        print(f"Warning: Dropped {initial_rows - len(df)} rows due to invalid date format in 'Dttime_ist'.")

    df = df.sort_values(by='Dttime_ist').reset_index(drop=True)
    fuel_sensor_column = config['fuel_sensor_column']
    df['fuel_level_liters'] = df[fuel_sensor_column].apply(
        lambda mv: convert_mv_to_liters(mv, config)
    )
    print(f"Data pre-processed. {len(df)} records found.")

    # --- Step 1: Data Filtering (Same as before) ---
    filtered_data = []
    if not df.empty:
        last_valid_reading = df.iloc[0]
        filtered_data.append(last_valid_reading)
        for _, current_reading in df.iloc[1:].iterrows():
            if abs(current_reading['fuel_level_liters'] - last_valid_reading['fuel_level_liters']) >= config['filtration_level']:
                filtered_data.append(current_reading)
                last_valid_reading = current_reading
    
    if not filtered_data:
        print("No data left after filtering. Exiting.")
        return []
        
    filtered_df = pd.DataFrame(filtered_data).reset_index(drop=True)
    print(f"Step 1: Filtering complete. Data reduced to {len(filtered_df)} records.")

    # --- Step 2 & 3: Advanced Drain Detection Loop ---
    confirmed_drains = []
    potential_drain_start_info = None
    time_stationary_seconds = 0

    for i in range(1, len(filtered_df)):
        prev = filtered_df.iloc[i-1]
        curr = filtered_df.iloc[i]
        
        # Calculate time elapsed since last data point
        time_diff_seconds = (curr['Dttime_ist'] - prev['Dttime_ist']).total_seconds()

        # Update stationary timer
        if prev['Speed'] == 0:
            time_stationary_seconds += time_diff_seconds
        else:
            time_stationary_seconds = 0 # Reset if vehicle moves

        # Check for a drop in fuel
        fuel_drop = prev['fuel_level_liters'] - curr['fuel_level_liters']
        
        # --- Stationary Drain Logic ---
        # A new potential drain starts if fuel drops significantly
        if fuel_drop >= config['min_drain_volume']:
            # Condition 1: Vehicle was stationary long enough before the drop
            is_stationary_long_enough = time_stationary_seconds >= config['min_stay_time_before_theft']
            
            # Condition 2: Or, we allow detection in motion
            is_in_motion_detection_allowed = config['detect_thefts_in_motion']

            if is_stationary_long_enough or is_in_motion_detection_allowed:
                if potential_drain_start_info is None:
                    # This is the start of a new potential drain
                    print(f"Potential drain started at {curr['Dttime_ist']}. Drop of {fuel_drop:.2f}L.")
                    potential_drain_start_info = {
                        "start_index": i,
                        "start_time": prev['Dttime_ist'],
                        "start_fuel_level": prev['fuel_level_liters'],
                        "start_location": {"latitude": prev['Latitude'], "longitude": prev['Longitude']},
                        "was_stationary": is_stationary_long_enough
                    }

        # If we are tracking a potential drain, check if it should be confirmed or discarded
        if potential_drain_start_info:
            start_index = potential_drain_start_info['start_index']
            time_since_drain_started = (curr['Dttime_ist'] - filtered_df.iloc[start_index]['Dttime_ist']).total_seconds()

            # Check for fuel level recovery
            if curr['fuel_level_liters'] >= potential_drain_start_info['start_fuel_level'] - (config['filtration_level'] / 2):
                print(f"Discarding potential drain at {curr['Dttime_ist']} due to fuel level recovery.")
                potential_drain_start_info = None # Discard the drain
            # Check for timeout
            elif time_since_drain_started >= config['timeout_to_confirm_theft']:
                final_drop = potential_drain_start_info['start_fuel_level'] - curr['fuel_level_liters']
                print(f"CONFIRMING drain at {curr['Dttime_ist']}. Final drop: {final_drop:.2f}L.")
                confirmed_drains.append({
                    "start_time": potential_drain_start_info['start_time'],
                    "end_time": curr['Dttime_ist'],
                    "volume_decrease_liters": round(final_drop, 2),
                    "start_fuel_level": round(potential_drain_start_info['start_fuel_level'], 2),
                    "end_fuel_level": round(curr['fuel_level_liters'], 2),
                    "location": potential_drain_start_info['start_location'],
                    "detection_mode": "Stationary" if potential_drain_start_info['was_stationary'] else "In Motion"
                })
                potential_drain_start_info = None # Reset after confirmation

    print(f"\nDetection loop finished. Found {len(confirmed_drains)} confirmed drains.")
    return confirmed_drains

if __name__ == '__main__':
    # --- CONFIGURATION ---
    config = {
        # File and Column Settings
        "data_file_path": "table.xlsx",
        "fuel_sensor_column": "An1",

        # Calibration Settings
        "calibration": {
            "mv_empty": 173, "liters_empty": 0,
            "mv_full": 5062, "liters_full": 375
        },

        # --- Advanced Algorithm Tuning (Based on Wialon Documentation) ---
        "filtration_level": 5,             # (Liters) Initial filter to remove noise.
        "min_drain_volume": 10,            # (Liters) The smallest drop to be considered a potential theft.
        "min_stay_time_before_theft": 300, # (Seconds) Vehicle must be stationary for this long before a drain is considered. 5 minutes.
        "timeout_to_confirm_theft": 180,   # (Seconds) The fuel level must stay low for this long to confirm the theft. 3 minutes.
        "detect_thefts_in_motion": False   # Set to True to detect drains even when the vehicle is moving (less reliable).
    }

    try:
        vehicle_data_df = pd.read_excel(config['data_file_path'])
        final_drains = detect_fuel_drains_advanced(vehicle_data_df, config)

        # --- Final Output ---
        if final_drains:
            print("\n--- CONFIRMED FUEL DRAINS DETECTED (ADVANCED LOGIC) ---")
            for i, drain in enumerate(final_drains, 1):
                print(f"\nDrain #{i}:")
                print(f"  Volume: {drain['volume_decrease_liters']} Liters")
                print(f"  Duration: {drain['start_time']} -> {drain['end_time']}")
                print(f"  Detection Mode: {drain['detection_mode']}")
                print(f"  Location (Lat, Lng): {drain['location']['latitude']}, {drain['location']['longitude']}")
                print(f"  Google Maps Link: https://www.google.com/maps?q={drain['location']['latitude']},{drain['location']['longitude']}")
        else:
            print("\n--- No fuel drains detected based on the current configuration. ---")

    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{config['data_file_path']}'")
    except KeyError as e:
        print(f"ERROR: A required column is missing from the Excel file: {e}")

