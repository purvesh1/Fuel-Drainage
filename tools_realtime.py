import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import deque

# ==============================================================================
# CLASS 1: REAL-TIME FILTERING LOGIC
# This class is unchanged and remains a dependency for the analyzer.
# ==============================================================================
class RealTimeFilter:
    """
    A stateful filter designed to process a stream of data points in real-time.
    It maintains an internal window of the most recent values.
    """
    def __init__(self, config: dict):
        self.config = config
        self.algorithm = config.get('filtering_algorithm', 'Median Filter')
        self._window = deque(maxlen=config.get('median_window_size', 3))
        self._last_accepted_value = None

    def filter(self, new_value: float) -> float:
        """Processes a single new data point and returns its filtered value."""
        if self.algorithm == 'Median Filter':
            self._window.append(new_value)
            return np.median(self._window)
        elif self.algorithm == 'Magnitude Threshold':
            if self._last_accepted_value is None:
                self._last_accepted_value = new_value
                return new_value
            if abs(new_value - self._last_accepted_value) >= self.config['filtration_level']:
                self._last_accepted_value = new_value
                return new_value
            else:
                return self._last_accepted_value
        elif self.algorithm == 'Adaptive Median Filter':
            st.warning("Adaptive Median Filter is not suitable for real-time streaming; using Median Filter instead.")
            self.algorithm = 'Median Filter'
            return self.filter(new_value)
        else:
            return new_value

# ==============================================================================
# CLASS 2: CORE ANALYSIS ENGINE (CORRECTED)
# ==============================================================================
class FuelAnalyzer:
    """
    Orchestrates the fuel data analysis using a real-time, streaming approach.
    """
    def __init__(self, df: pd.DataFrame, config: dict):
        self.raw_df = df.copy()
        self.config = config
        self.processed_data = [] 
        self.events = []
        
        # --- State variables are now instance attributes ---
        self.rt_filter = RealTimeFilter(self.config)
        self.potential_event = None
        self.time_stationary_seconds = 0
        # --- FIX: Added state to handle spurious zero readings ---
        self.last_good_fuel_value = None

    def run_analysis(self) -> dict:
        """Executes the full analysis pipeline using a single pass."""
        if self.raw_df.empty:
            return { "events": [], "processed_df": pd.DataFrame(), "raw_df": self.raw_df }
            
        self._prepare_data()
        self._process_data_stream()
        
        return {
            "events": self.events,
            "processed_df": pd.DataFrame(self.processed_data) if self.processed_data else pd.DataFrame(),
            "raw_df": self.raw_df
        }
        
    def _prepare_data(self):
        """Handles initial data cleaning and sorting."""
        st.write(f"Preparing data stream for IMEI: `{self.raw_df['Imei'].iloc[0]}`...")
        self.raw_df['Dttime_ist'] = pd.to_datetime(self.raw_df['Dttime_ist'], errors='coerce')
        self.raw_df.dropna(subset=['Dttime_ist'], inplace=True)
        self.raw_df = self.raw_df.sort_values(by='Dttime_ist').reset_index(drop=True)

    def _process_data_stream(self):
        """
        Processes the data as a stream by calling the single point processor
        in a loop. This now uses a while loop to better simulate a dynamic stream.
        """
        i = 1
        while i < len(self.raw_df):
            prev_raw = self.raw_df.iloc[i-1]
            curr_raw = self.raw_df.iloc[i]
            self._process_single_point(prev_raw, curr_raw)
            i += 1

    def _process_single_point(self, prev_raw: pd.Series, curr_raw: pd.Series):
        """
        Encapsulated logic to process one data point against the previous one.
        This method updates the state of the analyzer (e.g., events, filtered data).
        """
        # Step 1: Calibrate the current point's fuel value
        slope = self.config['calibration']['slope']
        intercept = self.config['calibration']['intercept']
        fuel_sensor_col = self.config['fuel_sensor_column']
        current_fuel_calibrated = max(0, (curr_raw[fuel_sensor_col] * slope) + intercept)

        # --- FIX: Handle spurious zero readings ---
        # This is the real-time equivalent of .replace(0, np.nan).ffill()
        if current_fuel_calibrated > 0:
            self.last_good_fuel_value = current_fuel_calibrated
        elif self.last_good_fuel_value is not None:
             # If current value is 0, use the last known good value
            current_fuel_calibrated = self.last_good_fuel_value
        # If the very first value is 0, we have to let it pass
        
        # Step 2: Filter the (now corrected) calibrated value in real-time
        current_fuel_filtered = self.rt_filter.filter(current_fuel_calibrated)
        
        # Step 3: Store the processed data point for later charting
        processed_point = curr_raw.to_dict()
        processed_point['fuel_level_liters'] = current_fuel_filtered
        self.processed_data.append(processed_point)

        # Step 4: Get the previous *filtered* value for event detection
        prev_fuel_filtered = self.processed_data[-2]['fuel_level_liters'] if len(self.processed_data) > 1 else current_fuel_filtered

        # Step 5: Update stationary time
        time_diff_seconds = (curr_raw['Dttime_ist'] - prev_raw['Dttime_ist']).total_seconds()
        self.time_stationary_seconds = self.time_stationary_seconds + time_diff_seconds if prev_raw['Speed'] == 0 else 0
        
        # CORE STATE MACHINE: Search for or manage an existing event
        if self.potential_event is None:
            # STATE 1: Searching for a new event
            is_stationary = self.time_stationary_seconds >= self.config['min_stay_time_before_event']
            can_detect = is_stationary or self.config['detect_events_in_motion']
            fuel_change = current_fuel_filtered - prev_fuel_filtered
            
            event_type = None
            if fuel_change >= self.config['min_fill_volume']: event_type = "Filling"
            elif -fuel_change >= self.config['min_drain_volume']: event_type = "Drain"

            if event_type and can_detect:
                self.potential_event = {
                    "type": event_type, "start_time": prev_raw['Dttime_ist'],
                    "start_fuel": prev_fuel_filtered, "last_change_time": curr_raw['Dttime_ist'],
                    "potential_end_fuel": current_fuel_filtered,
                }
        else:
            # STATE 2: Already tracking an event
            if abs(current_fuel_filtered - self.potential_event['start_fuel']) < self.config['false_event_threshold']:
                self.potential_event = None # Cancel event
                return

            if abs(current_fuel_filtered - self.potential_event['potential_end_fuel']) > 1.0:
                self.potential_event['potential_end_fuel'] = current_fuel_filtered
                self.potential_event['last_change_time'] = curr_raw['Dttime_ist']

            time_since_last_change = (curr_raw['Dttime_ist'] - self.potential_event['last_change_time']).total_seconds()
            
            if time_since_last_change >= self.config['timeout_to_confirm_event']:
                volume = abs(self.potential_event['potential_end_fuel'] - self.potential_event['start_fuel'])
                min_vol = self.config.get('min_fill_volume' if self.potential_event['type'] == 'Filling' else 'min_drain_volume')
                
                if volume >= min_vol:
                    self.events.append({
                        "Event": self.potential_event['type'], "Start Time": self.potential_event['start_time'],
                        "End Time": self.potential_event['last_change_time'], "Start Fuel (L)": round(self.potential_event['start_fuel'], 2),
                        "End Fuel (L)": round(self.potential_event['potential_end_fuel'], 2), "Volume (L)": round(volume, 2),
                    })
                self.potential_event = None

# ==============================================================================
# CLASS 3: UI AND REPORTING LOGIC 
# ==============================================================================
class ReportGenerator:
    """Generates all Streamlit UI components from analysis results."""
    def __init__(self, analysis_results: dict, config: dict):
        self.results = analysis_results
        self.config = config
        self.events = self.results['events']
        self.processed_df = self.results['processed_df']
        self.raw_df = self.results['raw_df']
        # Calibrate the raw data for charting comparison
        if not self.raw_df.empty:
            slope = self.config['calibration']['slope']
            intercept = self.config['calibration']['intercept']
            fuel_col = self.config['fuel_sensor_column']
            self.raw_df['fuel_level_liters'] = self.raw_df[fuel_col].apply(lambda mv: max(0, (mv * slope) + intercept))

    def display_full_report(self, filename: str):
        st.header(f"Analysis Results for: `{filename}`")
        self._display_events_table()
        self._display_fuel_chart()

    def _display_events_table(self):
        if self.events:
            st.subheader("Detected Fuel Events")
            st.dataframe(pd.DataFrame(self.events))
        else:
            st.info("No significant fuel events were detected with the current settings.")

    def _display_fuel_chart(self):
        if self.processed_df is None or self.processed_df.empty:
            return
            
        st.subheader("Fuel Level Chart")
        fig = go.Figure()

        if self.config.get('show_raw_data', False) and not self.raw_df.empty:
            fig.add_trace(go.Scatter(
                x=self.raw_df['Dttime_ist'], y=self.raw_df['fuel_level_liters'],
                mode='lines', name='Fuel Level (Raw Calibrated)',
                line=dict(color='rgba(173, 216, 230, 0.6)', width=1.5, dash='dot')
            ))

        fig.add_trace(go.Scatter(
            x=self.processed_df['Dttime_ist'], y=self.processed_df['fuel_level_liters'],
            mode='lines', name='Fuel Level (Real-Time Filtered)', line=dict(color='blue', width=2)
        ))
        
        if self.events:
            for event in self.events:
                color = "rgba(0, 255, 0, 0.2)" if event['Event'] == 'Filling' else "rgba(255, 0, 0, 0.2)"
                name = f"{event['Event']} ({event['Volume (L)']:.2f} L)"
                fig.add_vrect(
                    x0=event['Start Time'], x1=event['End Time'], fillcolor=color,
                    layer="below", line_width=0, annotation_text=name, annotation_position="top left"
                )

        fig.update_layout(
            title=f"Fuel Analysis for IMEI: {self.raw_df['Imei'].iloc[0]}",
            xaxis_title="Date and Time", yaxis_title="Fuel Level (Liters)",
            legend_title="Data Series", hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# MAIN CONTROLLER FUNCTION (Unchanged)
# ==============================================================================
def process_and_display(df: pd.DataFrame, config: dict, filename: str):
    """
    Main controller function that orchestrates the real-time analysis and reporting.
    """
    try:
        analyzer = FuelAnalyzer(df, config)
        results = analyzer.run_analysis()
        reporter = ReportGenerator(results, config)
        reporter.display_full_report(filename)
    except Exception as e:
        import traceback
        st.error(f"An error occurred during processing: {e}")
        st.error(traceback.format_exc())

    return reporter.events

def caliberate(calibration_file):
    """
    Parses a calibration file to calculate slope and intercept using linear regression.
    
    Args:
        calibration_file: The uploaded file object from Streamlit.

    Returns:
        A tuple (slope, intercept) on success.
        A tuple (None, None) on any failure.
    """
    if calibration_file is None:
        return None, None
        
    try:
        # Try reading as CSV, then fall back to Excel if it fails
        try:
            cal_df = pd.read_csv(calibration_file)
        except Exception:
            calibration_file.seek(0)
            cal_df = pd.read_excel(calibration_file)

        # Ensure the required columns exist and there's enough data for regression
        if 'Voltage' in cal_df.columns and 'Fuel' in cal_df.columns and len(cal_df) >= 2:
            x = cal_df['Voltage'].astype(float)
            y = cal_df['Fuel'].astype(float)
            slope, intercept = np.polyfit(x, y, 1)
            
            st.success("Calibration complete!")
            # --- Display for active calibration parameters ---
            st.subheader("Sensor Calibration")
            st.info(f"Slope (L/mV): `{st.session_state.calibration_params['slope']:.4f}`\n\n"
                    f"Intercept (L): `{st.session_state.calibration_params['intercept']:.2f}`")
            return slope, intercept
        else:
            st.error("Calibration file must have 'Voltage' and 'Fuel' columns with at least 2 data points.")
            return None, None
    except Exception as e:
        st.error(f"Error reading calibration file: {e}")
        return None, None
