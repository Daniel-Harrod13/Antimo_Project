"""
Main script for the Commodity Market Price Prediction & Analysis Platform.

This script provides a command-line interface to run different components of the application.
"""

import os
import argparse
import logging
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    logger.info("Directory setup completed")

def collect_data(symbols=None, period='5y'):
    """Collect historical commodity price data."""
    from src.data_processing.data_collector import CommodityDataCollector
    
    logger.info(f"Collecting data for symbols: {symbols}, period: {period}")
    collector = CommodityDataCollector(data_dir='data')
    
    # Fetch data
    raw_data = collector.fetch_historical_data(symbols=symbols, period=period)
    
    # Preprocess data
    processed_data = collector.preprocess_data(raw_data)
    
    # Generate summary
    summary = collector.get_data_summary()
    logger.info(f"Data collection summary: {summary}")
    
    return summary

def train_models(commodity=None):
    """Train time series forecasting models."""
    from src.model_training.time_series_models import TimeSeriesModels
    import pandas as pd
    import os
    
    ts_models = TimeSeriesModels(models_dir='models')
    
    # Get list of commodities to process
    commodities = []
    if commodity:
        commodities = [commodity]
    else:
        processed_dir = 'data/processed'
        for filename in os.listdir(processed_dir):
            if filename.endswith('_processed.csv'):
                commodity_name = filename.replace('_processed.csv', '').replace('_', ' ').title()
                commodities.append(commodity_name)
    
    logger.info(f"Training models for commodities: {commodities}")
    
    results = {}
    for commodity_name in commodities:
        logger.info(f"Processing {commodity_name}")
        
        # Load data
        file_path = f"data/processed/{commodity_name.lower().replace(' ', '_')}_processed.csv"
        if not os.path.exists(file_path):
            logger.warning(f"Data file not found for {commodity_name}: {file_path}")
            continue
            
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        
        # Train ARIMA model
        arima_info = ts_models.train_arima_model(
            data=data,
            target_col='Close',
            test_size=0.2,
            auto_order=True,
            commodity_name=commodity_name
        )
        
        # Train Prophet model
        prophet_info = ts_models.train_prophet_model(
            data=data.reset_index(),
            date_col='Date',
            target_col='Close',
            test_size=0.2,
            forecast_periods=30,
            commodity_name=commodity_name
        )
        
        # Train LSTM model
        lstm_info = ts_models.train_lstm_model(
            data=data,
            target_col='Close',
            test_size=0.2,
            look_back=60,
            lstm_units=50,
            epochs=50,
            commodity_name=commodity_name
        )
        
        # Store results
        results[commodity_name] = {
            'arima': arima_info.get('metrics', {}),
            'prophet': prophet_info.get('metrics', {}),
            'lstm': lstm_info.get('metrics', {})
        }
        
        logger.info(f"Completed training models for {commodity_name}")
    
    return results

def run_dashboard():
    # Run the Streamlit dashboard.
    import subprocess
    import sys
    
    logger.info("Starting dashboard")
    
    # Run the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", "src/visualization/dashboard.py"]
    process = subprocess.Popen(cmd)
    
    logger.info(f"Dashboard started with PID {process.pid}")
    
    return process

def main():
    # Main function to parse arguments and run the application. 
    parser = argparse.ArgumentParser(description='Commodity Market Price Prediction & Analysis Platform')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup directories')
    
    # Data collection command
    collect_parser = subparsers.add_parser('collect', help='Collect commodity price data')
    collect_parser.add_argument('--symbols', nargs='+', help='List of commodity symbols to collect')
    collect_parser.add_argument('--period', default='5y', help='Time period to collect (e.g., 1y, 5y)')
    
    # Model training command
    train_parser = subparsers.add_parser('train', help='Train forecasting models')
    train_parser.add_argument('--commodity', help='Specific commodity to train models for')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run the dashboard')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Execute the appropriate command
    if args.command == 'setup':
        setup_directories()
        logger.info("Setup completed successfully")
    
    elif args.command == 'collect':
        setup_directories()  # Ensure directories exist
        summary = collect_data(symbols=args.symbols, period=args.period)
        logger.info("Data collection completed successfully")
        print("Data Collection Summary:")
        for commodity in summary.get('commodities', []):
            print(f"  {commodity['name']}: {commodity['records']} records from {commodity['start_date']} to {commodity['end_date']}")
    
    elif args.command == 'train':
        setup_directories()  # Ensure directories exist
        results = train_models(commodity=args.commodity)
        logger.info("Model training completed successfully")
        print("Model Training Results:")
        for commodity, models in results.items():
            print(f"\n{commodity}:")
            for model_type, metrics in models.items():
                if 'rmse' in metrics and 'mape' in metrics:
                    print(f"  {model_type.upper()}: RMSE = {metrics['rmse']:.4f}, MAPE = {metrics['mape']:.2f}%")
    
    elif args.command == 'dashboard':
        setup_directories()  # Ensure directories exist
        process = run_dashboard()
        print("Dashboard is running. Press Ctrl+C to stop.")
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            print("Dashboard stopped.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 