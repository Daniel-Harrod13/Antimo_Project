"""
Dashboard Module

This module implements a Streamlit dashboard for visualizing commodity price predictions and market trends.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import json
from datetime import datetime, timedelta
import joblib

# Add the project root to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_processing.data_collector import CommodityDataCollector
from src.model_training.time_series_models import TimeSeriesModels

# Set page configuration
st.set_page_config(
    page_title="Commodity Market Price Prediction & Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))

# Initialize data collector and models
data_collector = CommodityDataCollector(data_dir=DATA_DIR)
ts_models = TimeSeriesModels(models_dir=MODELS_DIR)

# Function to load available commodities
@st.cache_data
def load_available_commodities():
    """Load the list of available commodities from the processed data directory."""
    commodities = []
    processed_dir = os.path.join(DATA_DIR, 'processed')
    
    if os.path.exists(processed_dir):
        for filename in os.listdir(processed_dir):
            if filename.endswith('_processed.csv'):
                commodity_name = filename.replace('_processed.csv', '').replace('_', ' ').title()
                commodities.append(commodity_name)
    
    return commodities

# Function to load commodity data
@st.cache_data
def load_commodity_data(commodity_name):
    """Load the processed data for a specific commodity."""
    file_path = os.path.join(
        DATA_DIR, 
        'processed', 
        f"{commodity_name.lower().replace(' ', '_')}_processed.csv"
    )
    
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    else:
        return None

# Function to load available models
@st.cache_data
def load_available_models():
    """Load the list of available trained models."""
    models = []
    
    if os.path.exists(MODELS_DIR):
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith('.pkl') or filename.endswith('.h5'):
                if not filename.endswith('_scaler.pkl') and not filename.endswith('_components.png'):
                    model_name = filename.split('.')[0]
                    models.append(model_name)
    
    return models

# Function to create price chart
def create_price_chart(data, commodity_name, ma_periods=None):
    """Create an interactive price chart with optional moving averages."""
    fig = go.Figure()
    
    # Add price data
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Close'],
            mode='lines',
            name=f'{commodity_name} Price',
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # Add moving averages if requested
    if ma_periods:
        for period in ma_periods:
            if f'MA_{period}' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data[f'MA_{period}'],
                        mode='lines',
                        name=f'{period}-Day MA',
                        line=dict(width=1.5)
                    )
                )
    
    # Update layout
    fig.update_layout(
        title=f'{commodity_name} Price History',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
        height=500
    )
    
    return fig

# Function to create technical indicators chart
def create_technical_indicators_chart(data, commodity_name):
    """Create a chart with technical indicators."""
    fig = make_subplots(
        rows=3, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price', 'Volatility (21-Day)', 'Momentum (14-Day)')
    )
    
    # Add price data to first subplot
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Close'],
            mode='lines',
            name=f'{commodity_name} Price',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    # Add volatility to second subplot
    if 'Volatility_21' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Volatility_21'],
                mode='lines',
                name='21-Day Volatility',
                line=dict(color='#ff7f0e', width=1.5)
            ),
            row=2, col=1
        )
    
    # Add momentum to third subplot
    if 'Momentum_14' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Momentum_14'],
                mode='lines',
                name='14-Day Momentum',
                line=dict(color='#2ca02c', width=1.5)
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{commodity_name} Technical Indicators',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
        height=700
    )
    
    return fig

# Function to create forecast chart
def create_forecast_chart(historical_data, forecast_data, commodity_name, model_name):
    """Create a chart showing historical data and forecast."""
    fig = go.Figure()
    
    # Add historical price data
    fig.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # Add forecast
    if 'dates' in forecast_data and 'forecast' in forecast_data:
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in forecast_data['dates']]
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=forecast_data['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            )
        )
        
        # Add confidence intervals if available
        if 'lower_bound' in forecast_data and 'upper_bound' in forecast_data:
            fig.add_trace(
                go.Scatter(
                    x=dates + dates[::-1],
                    y=forecast_data['upper_bound'] + forecast_data['lower_bound'][::-1],
                    fill='toself',
                    fillcolor='rgba(255,127,14,0.2)',
                    line=dict(color='rgba(255,127,14,0)'),
                    name='95% Confidence Interval'
                )
            )
    
    # Update layout
    fig.update_layout(
        title=f'{commodity_name} Price Forecast ({model_name})',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
        height=500
    )
    
    return fig

# Function to create model comparison chart
def create_model_comparison_chart(model_metrics, commodity_name):
    """Create a chart comparing different model performances."""
    models = [m['model_type'] for m in model_metrics]
    rmse_values = [m['rmse'] for m in model_metrics]
    mape_values = [m['mape'] for m in model_metrics]
    
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=('RMSE (Lower is Better)', 'MAPE % (Lower is Better)'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(
            x=models,
            y=rmse_values,
            name='RMSE',
            marker_color='#1f77b4'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=models,
            y=mape_values,
            name='MAPE %',
            marker_color='#ff7f0e'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'Model Performance Comparison for {commodity_name}',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

# Function to generate forecast
def generate_forecast(model_name, commodity_data, periods=30):
    """Generate forecast using a trained model."""
    # Extract model type from name
    model_parts = model_name.split('_')
    model_type = model_parts[-1]  # Last part is the model type (arima, prophet, lstm)
    commodity_name = '_'.join(model_parts[:-1])  # Everything before is the commodity name
    
    # Load the model
    if model_type == 'lstm':
        # For LSTM, we need both the model and the scaler
        model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
        scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return {"error": "Model files not found"}
        
        try:
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            
            # Prepare the most recent data
            look_back = 60  # Default value, should be stored with the model
            target_col = 'Close'  # Default value
            
            recent_data = commodity_data[target_col].values[-look_back:].reshape(-1, 1)
            scaled_data = scaler.transform(recent_data)
            
            # Initialize arrays for forecasting
            input_data = scaled_data.copy()
            forecasts = []
            
            # Generate forecasts one step at a time
            for _ in range(periods):
                # Reshape for LSTM input [1, look_back, 1]
                x_input = input_data[-look_back:].reshape(1, look_back, 1)
                
                # Predict the next value
                next_pred = model.predict(x_input, verbose=0)
                
                # Append to forecasts
                forecasts.append(next_pred[0, 0])
                
                # Update input data for next prediction
                input_data = np.append(input_data, next_pred)
                input_data = input_data.reshape(-1, 1)
            
            # Inverse transform to original scale
            forecast_values = scaler.inverse_transform(
                np.array(forecasts).reshape(-1, 1)
            )
            
            # Generate dates
            last_date = commodity_data['Date'].max()
            dates = [last_date + timedelta(days=i+1) for i in range(periods)]
            
            return {
                'forecast': forecast_values.flatten().tolist(),
                'dates': [d.strftime('%Y-%m-%d') for d in dates]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    elif model_type == 'prophet':
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            return {"error": "Model file not found"}
        
        try:
            model = joblib.load(model_path)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Extract the forecast for the future periods
            future_forecast = forecast.iloc[-periods:]
            
            return {
                'forecast': future_forecast['yhat'].tolist(),
                'lower_bound': future_forecast['yhat_lower'].tolist(),
                'upper_bound': future_forecast['yhat_upper'].tolist(),
                'dates': future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    elif model_type == 'arima' or model_type == 'sarima':
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            return {"error": "Model file not found"}
        
        try:
            model = joblib.load(model_path)
            
            # Generate forecast
            forecast = model.forecast(steps=periods)
            
            # Generate dates
            last_date = commodity_data['Date'].max()
            dates = [last_date + timedelta(days=i+1) for i in range(periods)]
            
            return {
                'forecast': forecast.tolist(),
                'dates': [d.strftime('%Y-%m-%d') for d in dates]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    else:
        return {"error": f"Unsupported model type: {model_type}"}

# Main dashboard layout
def main():
    """Main function to run the Streamlit dashboard."""
    # Title and description
    st.title("Commodity Market Price Prediction & Analysis Platform")
    st.markdown("""
    This dashboard provides tools for analyzing commodity price trends and forecasting future price movements
    using various time series forecasting models including ARIMA, Prophet, and LSTM.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Data Explorer", "Price Forecasting", "Model Performance", "About"])
    
    # Load available commodities
    commodities = load_available_commodities()
    
    if not commodities:
        st.warning("No commodity data found. Please run the data collection script first.")
        
        if st.button("Collect Sample Data"):
            st.info("Collecting sample data for Gold and Crude Oil (last 5 years)...")
            
            with st.spinner("Fetching data..."):
                # Collect data for Gold and Crude Oil
                raw_data = data_collector.fetch_historical_data(
                    symbols=['Gold', 'Crude Oil'],
                    period='5y'
                )
                
                # Preprocess the data
                processed_data = data_collector.preprocess_data(raw_data)
                
                # Generate and display summary
                summary = data_collector.get_data_summary()
                st.json(summary)
                
                st.success("Data collection completed! Please refresh the page.")
        
        return
    
    # Data Explorer Page
    if page == "Data Explorer":
        st.header("Commodity Price Data Explorer")
        
        # Select commodity
        selected_commodity = st.selectbox("Select Commodity", commodities)
        
        # Load data for selected commodity
        data = load_commodity_data(selected_commodity)
        
        if data is not None:
            # Display basic statistics
            st.subheader("Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Latest Price", f"${data['Close'].iloc[-1]:.2f}")
            
            with col2:
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                st.metric("Daily Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
            
            with col3:
                date_range = (data['Date'].max() - data['Date'].min()).days
                st.metric("Date Range", f"{date_range} days")
            
            with col4:
                st.metric("Data Points", f"{len(data)}")
            
            # Price chart
            st.subheader("Price History")
            
            # Moving average options
            ma_options = st.multiselect(
                "Add Moving Averages",
                [7, 21, 50, 100, 200],
                default=[50]
            )
            
            price_chart = create_price_chart(data, selected_commodity, ma_options)
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Technical indicators
            st.subheader("Technical Indicators")
            tech_chart = create_technical_indicators_chart(data, selected_commodity)
            st.plotly_chart(tech_chart, use_container_width=True)
            
            # Raw data table
            with st.expander("View Raw Data"):
                st.dataframe(data)
    
    # Price Forecasting Page
    elif page == "Price Forecasting":
        st.header("Commodity Price Forecasting")
        
        # Select commodity
        selected_commodity = st.selectbox("Select Commodity", commodities)
        
        # Load data for selected commodity
        data = load_commodity_data(selected_commodity)
        
        if data is not None:
            # Load available models for this commodity
            available_models = [m for m in load_available_models() 
                               if m.startswith(selected_commodity.lower().replace(' ', '_'))]
            
            if not available_models:
                st.warning(f"No trained models found for {selected_commodity}. Please train models first.")
                
                if st.button("Train Models"):
                    st.info(f"Training models for {selected_commodity}...")
                    
                    with st.spinner("Training ARIMA model..."):
                        # Prepare data for modeling
                        model_data = data.copy()
                        model_data = model_data.set_index('Date')
                        
                        # Train ARIMA model
                        arima_info = ts_models.train_arima_model(
                            data=model_data,
                            target_col='Close',
                            test_size=0.2,
                            auto_order=True,
                            commodity_name=selected_commodity
                        )
                        
                        st.write("ARIMA model trained:")
                        st.json(arima_info['metrics'])
                    
                    with st.spinner("Training Prophet model..."):
                        # Train Prophet model
                        prophet_info = ts_models.train_prophet_model(
                            data=data,
                            date_col='Date',
                            target_col='Close',
                            test_size=0.2,
                            forecast_periods=30,
                            commodity_name=selected_commodity
                        )
                        
                        st.write("Prophet model trained:")
                        st.json(prophet_info['metrics'])
                    
                    with st.spinner("Training LSTM model..."):
                        # Train LSTM model
                        lstm_info = ts_models.train_lstm_model(
                            data=model_data,
                            target_col='Close',
                            test_size=0.2,
                            look_back=60,
                            lstm_units=50,
                            epochs=50,
                            commodity_name=selected_commodity
                        )
                        
                        st.write("LSTM model trained:")
                        st.json(lstm_info['metrics'])
                    
                    st.success("Models trained successfully! Please refresh the page.")
                
                return
            
            # Select model
            selected_model = st.selectbox("Select Model", available_models)
            
            # Forecast parameters
            forecast_periods = st.slider("Forecast Periods (Days)", 7, 90, 30)
            
            # Generate forecast
            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    forecast = generate_forecast(selected_model, data, periods=forecast_periods)
                    
                    if 'error' in forecast:
                        st.error(f"Error generating forecast: {forecast['error']}")
                    else:
                        # Display forecast chart
                        forecast_chart = create_forecast_chart(
                            data, 
                            forecast, 
                            selected_commodity,
                            selected_model
                        )
                        st.plotly_chart(forecast_chart, use_container_width=True)
                        
                        # Display forecast data
                        with st.expander("View Forecast Data"):
                            forecast_df = pd.DataFrame({
                                'Date': forecast['dates'],
                                'Forecast': forecast['forecast']
                            })
                            
                            if 'lower_bound' in forecast and 'upper_bound' in forecast:
                                forecast_df['Lower Bound'] = forecast['lower_bound']
                                forecast_df['Upper Bound'] = forecast['upper_bound']
                            
                            st.dataframe(forecast_df)
    
    # Model Performance Page
    elif page == "Model Performance":
        st.header("Model Performance Comparison")
        
        # Select commodity
        selected_commodity = st.selectbox("Select Commodity", commodities)
        
        # Load available models for this commodity
        available_models = [m for m in load_available_models() 
                           if m.startswith(selected_commodity.lower().replace(' ', '_'))]
        
        if not available_models:
            st.warning(f"No trained models found for {selected_commodity}. Please go to the Price Forecasting page to train models.")
            return
        
        # Load model metrics
        model_metrics = []
        for model_name in available_models:
            # Extract model type
            model_type = model_name.split('_')[-1].upper()
            
            # Load model metrics from file
            metrics_file = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    model_metrics.append({
                        'model_name': model_name,
                        'model_type': model_type,
                        'rmse': metrics['rmse'],
                        'mape': metrics['mape']
                    })
            else:
                # If metrics file doesn't exist, use placeholder values
                model_metrics.append({
                    'model_name': model_name,
                    'model_type': model_type,
                    'rmse': 0.0,
                    'mape': 0.0
                })
        
        if model_metrics:
            # Display model comparison chart
            comparison_chart = create_model_comparison_chart(model_metrics, selected_commodity)
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Display metrics table
            st.subheader("Model Metrics")
            metrics_df = pd.DataFrame(model_metrics)
            metrics_df = metrics_df[['model_type', 'rmse', 'mape']]
            metrics_df.columns = ['Model Type', 'RMSE', 'MAPE (%)']
            st.dataframe(metrics_df)
            
            # Best model recommendation
            best_model = min(model_metrics, key=lambda x: x['rmse'])
            st.success(f"Recommended Model: {best_model['model_type']} (RMSE: {best_model['rmse']:.4f}, MAPE: {best_model['mape']:.2f}%)")
    
    # About Page
    elif page == "About":
        st.header("About This Project")
        
        st.markdown("""
        ## Commodity Market Price Prediction & Analysis Platform
        
        This platform implements an end-to-end machine learning pipeline for predicting commodity price movements 
        and analyzing market trends. It leverages various AI/ML techniques including time series forecasting, 
        deep learning, and natural language processing to provide insights for trading decisions.
        
        ### Features
        - Historical price data collection and preprocessing for multiple commodities
        - Advanced time series forecasting using statistical and deep learning models
        - Interactive dashboards for visualizing predictions and market trends
        - Performance evaluation of different trading strategies
        
        ### Models Implemented
        - ARIMA/SARIMA for statistical time series forecasting
        - Prophet for trend and seasonality decomposition
        - LSTM networks for sequence modeling
        
        ### Data Sources
        - Yahoo Finance API for historical price data
        
        ### Technologies Used
        - Python for data processing and model implementation
        - Pandas, NumPy for data manipulation
        - TensorFlow, Keras for deep learning models
        - Plotly, Streamlit for interactive visualization
        
        ### Future Enhancements
        - Integration of market sentiment analysis using NLP
        - Addition of more commodities and financial instruments
        - Implementation of trading strategy backtesting
        - Real-time data updates and alerts
        """)

if __name__ == "__main__":
    main() 