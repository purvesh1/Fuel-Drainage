import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# ==============================================================================
# CLASS 1: FILTERING LOGIC
# This class groups all filtering algorithms from your code into a single unit.
# ==============================================================================
class Filter:
    """A utility class that applies a chosen filtering strategy to the fuel data."""

    @staticmethod
    def apply(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Applies the filtering algorithm specified in the config."""
        algorithm = config.get('filtering_algorithm', 'Magnitude Threshold')
        st.write(f"Applying filter: {algorithm}")
        
        if algorithm == 'Median Filter':
            return Filter._median_filter(df, config)
        elif algorithm == 'Adaptive Median Filter':
            return Filter._adaptive_median_filter(df, config)
        elif algorithm == 'Magnitude Threshold':
            return Filter._magnitude_filter(df, config)
        else:
            st.warning(f"Unknown filter '{algorithm}'. Returning unfiltered data.")
            return df

    @staticmethod
    def _magnitude_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Applies the Magnitude Threshold filter."""
        level = config['filtration_level']
        filtered_data = []
        if not df.empty:
            last_valid_reading = df.iloc[0]
            filtered_data.append(last_valid_reading)
            for _, current_reading in df.iloc[1:].iterrows():
                fuel_diff = abs(current_reading['fuel_level_liters'] - last_valid_reading['fuel_level_liters'])
                if fuel_diff >= level:
                    filtered_data.append(current_reading)
                    last_valid_reading = current_reading
        return pd.DataFrame(filtered_data).reset_index(drop=True)

    @staticmethod
    def _median_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Applies a standard rolling Median Filter."""
        window_size = config['median_window_size']
        if window_size < 3: return df
        if window_size % 2 == 0: window_size += 1
        
        df['fuel_level_liters'] = df['fuel_level_liters'].rolling(window=window_size, center=True, min_periods=1).median()
        df.dropna(subset=['fuel_level_liters'], inplace=True)
        return df.reset_index(drop=True)

    @staticmethod
    def _adaptive_median_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Applies a true adaptive median filter."""
        column_name = 'fuel_level_liters'
        min_window, max_window = config['adaptive_min_window'], config['adaptive_max_window']
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


# ==============================================================================
# CLASS 2: CORE ANALYSIS ENGINE
# This class contains the logic from your `detect_fuel_events` function.
# ==============================================================================
class FuelAnalyzer:
    """Orchestrates the fuel data analysis process from raw data to detected events."""
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config
        self.raw_calibrated_df = None
        self.processed_df = None
        self.events = []

    def run_analysis(self) -> dict:
        """Executes the full analysis pipeline."""
        self._prepare_data()
        self._filter_data()
        self._detect_events()
        
        return {
            "events": self.events,
            "processed_df": self.processed_df,
            "raw_df": self.raw_calibrated_df
        }

    def _convert_mv_to_liters(self, millivolts: float) -> float:
        """Applies the linear calibration formula."""
        slope = self.config['calibration']['slope']
        intercept = self.config['calibration']['intercept']
        fuel_liters = (millivolts * slope) + intercept
        return max(0, fuel_liters)

    def _prepare_data(self):
        """Handles data cleaning, sorting, and calibration."""
        st.write(f"Preparing data for IMEI: `{self.df['Imei'].iloc[0]}`...")
        self.df['Dttime_ist'] = pd.to_datetime(self.df['Dttime_ist'], errors='coerce')
        self.df.dropna(subset=['Dttime_ist'], inplace=True)
        self.df = self.df.sort_values(by='Dttime_ist').reset_index(drop=True)
        
        fuel_sensor_column = self.config['fuel_sensor_column']
        self.df['fuel_level_liters'] = self.df[fuel_sensor_column].apply(self._convert_mv_to_liters)
        self.df['fuel_level_liters'] = self.df['fuel_level_liters'].replace(0, np.nan).ffill()
        
        self.raw_calibrated_df = self.df.copy()

    def _filter_data(self):
        """Applies the selected filter to the prepared data."""
        if self.raw_calibrated_df is None:
            raise ValueError("Data has not been prepared. Run _prepare_data() first.")
        self.processed_df = Filter.apply(self.raw_calibrated_df.copy(), self.config)

    def _detect_events(self):
        """Contains the core state machine for detecting fuel events."""
        if self.processed_df is None or self.processed_df.empty:
            return

        potential_event = None
        time_stationary_seconds = 0

        for i in range(1, len(self.processed_df)):
            prev = self.processed_df.iloc[i-1]
            curr = self.processed_df.iloc[i]
            
            time_diff_seconds = (curr['Dttime_ist'] - prev['Dttime_ist']).total_seconds()
            time_stationary_seconds = time_stationary_seconds + time_diff_seconds if prev['Speed'] == 0 else 0
            
            if potential_event is None:
                is_stationary = time_stationary_seconds >= self.config['min_stay_time_before_event']
                can_detect = is_stationary or self.config['detect_events_in_motion']
                fuel_change = curr['fuel_level_liters'] - prev['fuel_level_liters']
                
                event_type = None
                if fuel_change >= self.config['min_fill_volume']: event_type = "Filling"
                elif -fuel_change >= self.config['min_drain_volume']: event_type = "Drain"

                if event_type and can_detect:
                    potential_event = {
                        "type": event_type, "start_time": prev['Dttime_ist'],
                        "start_fuel": prev['fuel_level_liters'], "last_change_time": curr['Dttime_ist'],
                        "potential_end_fuel": curr['fuel_level_liters'],
                    }
            else:
                if abs(curr['fuel_level_liters'] - potential_event['start_fuel']) < self.config['false_event_threshold']:
                    potential_event = None # Cancel event
                    continue

                if abs(curr['fuel_level_liters'] - potential_event['potential_end_fuel']) > 1.0:
                    potential_event['potential_end_fuel'] = curr['fuel_level_liters']
                    potential_event['last_change_time'] = curr['Dttime_ist']

                time_since_last_change = (curr['Dttime_ist'] - potential_event['last_change_time']).total_seconds()
                
                if time_since_last_change >= self.config['timeout_to_confirm_event']:
                    volume = abs(potential_event['potential_end_fuel'] - potential_event['start_fuel'])
                    min_vol = self.config.get('min_fill_volume' if potential_event['type'] == 'Filling' else 'min_drain_volume')
                    
                    if volume >= min_vol:
                        self.events.append({
                            "Event": potential_event['type'], "Start Time": potential_event['start_time'],
                            "End Time": potential_event['last_change_time'], "Start Fuel (L)": round(potential_event['start_fuel'], 2),
                            "End Fuel (L)": round(potential_event['potential_end_fuel'], 2), "Volume (L)": round(volume, 2),
                        })
                    potential_event = None

# ==============================================================================
# CLASS 3: UI AND REPORTING LOGIC
# This class handles the logic from your `process_and_display` function.
# ==============================================================================
class ReportGenerator:
    """Generates all Streamlit UI components from analysis results."""
    def __init__(self, analysis_results: dict, config: dict):
        self.results = analysis_results
        self.config = config
        self.events = self.results['events']
        self.processed_df = self.results['processed_df']
        self.raw_df = self.results['raw_df']

    def display_full_report(self, filename: str):
        """Displays the entire report including header, table, and chart."""
        st.header(f"Analysis Results for: `{filename}`")
        self._display_events_table()
        self._display_fuel_chart()

    def _display_events_table(self):
        """Displays the detected events in a table."""
        if self.events:
            st.subheader("Detected Fuel Events")
            st.dataframe(pd.DataFrame(self.events))
        else:
            st.info("No significant fuel events were detected with the current settings.")

    def _display_fuel_chart(self):
        """Creates and displays the Plotly fuel chart."""
        if self.processed_df is None or self.processed_df.empty:
            return
            
        st.subheader("Fuel Level Chart")
        fig = go.Figure()

        if self.config.get('show_raw_data', False) and self.raw_df is not None:
            fig.add_trace(go.Scatter(
                x=self.raw_df['Dttime_ist'], y=self.raw_df['fuel_level_liters'],
                mode='lines', name='Fuel Level (Raw)',
                line=dict(color='rgba(173, 216, 230, 0.6)', width=1.5, dash='dot')
            ))

        fig.add_trace(go.Scatter(
            x=self.processed_df['Dttime_ist'], y=self.processed_df['fuel_level_liters'],
            mode='lines', name='Fuel Level (Processed)', line=dict(color='blue', width=2)
        ))
        
        if self.events:
            for event in self.events:
                color = "rgba(0, 255, 0, 0.2)" if event['Event'] == 'Filling' else "rgba(255, 0, 0, 0.2)"
                name = f"{event['Event']} (+{event['Volume (L)']:.2f} L)" if event['Event'] == 'Filling' else f"{event['Event']} (-{event['Volume (L)']:.2f} L)"
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
# MAIN CONTROLLER FUNCTION
# This is the single entry point called by your main.py file.
# ==============================================================================
def process_and_display(df: pd.DataFrame, config: dict, filename: str):
    """
    Main controller function that orchestrates the analysis and reporting.
    """
    try:
        # 1. Instantiate the analyzer and run the full analysis.
        analyzer = FuelAnalyzer(df, config)
        results = analyzer.run_analysis()

        # 2. Instantiate the reporter with the analysis results.
        reporter = ReportGenerator(results, config)

        # 3. Display the complete report in the Streamlit app.
        reporter.display_full_report(filename)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

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
