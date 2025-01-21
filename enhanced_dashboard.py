# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:43:53 2024

@author: Adeka
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import time
import requests
import os
import logging
from monitoring_setup import MonitoringSystem

# Initialize monitoring system and logger
monitor = MonitoringSystem()
logger = logging.getLogger('dashboard')
logger.setLevel(logging.INFO)

# Create file handler if it doesn't exist
if not logger.handlers:
    log_file = os.path.join(monitor.logs_dir, 'dashboard.log')
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)







class ModelMonitor:
    def __init__(self, window_size=100):
        self.performance_history = deque(maxlen=window_size)
        self.last_update = None

    def update_metrics(self, actual, predicted):
        # Ensure arrays have the same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]

        try:
            mse = np.mean((actual - predicted) ** 2)
            mae = np.mean(np.abs(actual - predicted))
            r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)

            metrics = {
                'timestamp': datetime.now().isoformat(), #convert to ISO format string
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2)
            }

            self.performance_history.append(metrics)
            self.last_update = datetime.now()

            # Log metrics to monitoring system
            monitor.save_metrics(metrics)
            logger.info(f"Metrics updated: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def get_performance_trend(self):
        if not self.performance_history:
            return None
        return pd.DataFrame(list(self.performance_history))


class EnhancedDashboard:
    def __init__(self, api_url="http://localhost:8080", data_path="C:\\Users\\Adeka\\.spyder-py3\\processed_pems04_astgcn.npz"):
        self.api_url = api_url
        self.monitor = ModelMonitor()
        self.last_prediction_time = None
        self.load_data(data_path)

    def load_data(self, data_path):
        """Load data with monitoring"""
        try:
            if not os.path.exists(data_path):
                logger.error(f"Data file not found: {data_path}")
                st.error(f"Data file not found at: {data_path}")
                self.data = None
                return

            self.data = np.load(data_path, allow_pickle=True)
            logger.info("Data loaded successfully")
            st.success("Data loaded successfully")

            # Save data info to monitoring
            monitor.save_metrics({
                'timestamp': datetime.now().isoformat(),
                'event': 'data_loaded',
                'data_path': data_path
            })

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")
            self.data = None

    def get_predictions_and_actual(self, node_id, feature):
        if self.data is None:
            logger.error("No data available")
            st.error("No data available")
            return None, None

        try:
            feature_map = {'flow': 0, 'occupancy': 1, 'speed': 2}
            feature_idx = feature_map[feature]

            # Get actual values
            actual = self.data['test_data'].item()['y'][:, 0, node_id, feature_idx]

            try:
                # Get predictions from API
                sequence = self.data['test_data'].item()['X']
                adj_matrix = self.data['physical_adj']

                response = requests.post(
                    f"{self.api_url}/predict",
                    json={
                        "sequence": sequence.tolist(),
                        "adj_matrix": adj_matrix.tolist()
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    predictions = np.array(response.json()["prediction"])
                    predictions = predictions[0, node_id, 0, feature_idx]
                    predictions = np.repeat(predictions, len(actual))

                    # Debug output for API response
                    st.write("API Response:", response.json())  # Display API response in Streamlit
                    logger.info(f"API Response: {response.json()}")  # Log API response

                else:
                    logger.warning("Using simulated predictions due to API error")
                    st.error(f"API error: {response.status_code} - {response.text}")
                    predictions = actual + np.random.normal(0, 0.02, size=actual.shape)

            except Exception as e:
                logger.error(f"API error: {str(e)}")
                st.error(f"API error: {str(e)}")
                predictions = actual + np.random.normal(0, 0.02, size=actual.shape)

            # Update monitoring metrics
            self.monitor.update_metrics(actual, predictions)
            self.last_prediction_time = datetime.now()

            return actual, predictions

        except Exception as e:
            logger.error(f"Error in predictions: {str(e)}")
            st.error(f"Error in predictions: {str(e)}")
            return None, None

    def plot_comparison(self, node_id, feature):
        actual, predictions = self.get_predictions_and_actual(node_id, feature)

        if actual is None or predictions is None:
            return None

        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot predictions vs actual
            ax1.plot(actual[:100], label='Actual', alpha=0.7)
            ax1.plot(predictions[:100], label='Predicted', linestyle='--', alpha=0.7)
            ax1.set_title(f'Traffic {feature} for Node {node_id}: Actual vs Predicted')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel(feature.capitalize())
            ax1.legend()
            ax1.grid(True)

            # Plot performance trend
            perf_df = self.monitor.get_performance_trend()
            if perf_df is not None:
                ax2.plot(perf_df['timestamp'], perf_df['mse'], label='MSE')
                ax2.plot(perf_df['timestamp'], perf_df['mae'], label='MAE')
                ax2.set_title('Model Performance Trend')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Error Metrics')
                ax2.legend()
                ax2.grid(True)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error plotting comparison: {str(e)}")
            return None


def main():
    try:
        st.set_page_config(page_title="Enhanced Traffic Dashboard", layout="wide")
        st.title("Enhanced Traffic Dashboard")

        # Initialize dashboard
        dashboard = EnhancedDashboard()

        # Sidebar controls
        st.sidebar.header("Controls")
        node_id = st.sidebar.number_input("Select Node ID", min_value=0, max_value=306, value=105)
        feature = st.sidebar.selectbox("Select Feature", ['flow', 'occupancy', 'speed'])

        # Auto-refresh control
        refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 30)
        auto_refresh = st.sidebar.checkbox('Enable Auto-refresh', value=False)

        # Monitoring section
        st.sidebar.header("Monitoring")
        if st.sidebar.button("Check Model Health"):
            st.sidebar.info("Model health check coming soon...")

        # Main content
        st.header(f"Traffic Predictions for Node {node_id}")

        # Display plots
        fig = dashboard.plot_comparison(node_id, feature)
        if fig:
            st.pyplot(fig)

        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_rate)
            st.experimental_rerun()

    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting dashboard")
    main()
