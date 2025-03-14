"""
Main script for the Commodity Market Price Prediction & Analysis Platform.

This script provides a command-line interface to run different components of the application.
"""

import os
import argparse
import logging
from datetime import datetime
import sys

# Define project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    try:
        # Create directories with absolute paths
        os.makedirs(os.path.join(PROJECT_ROOT, 'data/raw'), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, 'data/processed'), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, 'src'), exist_ok=True)
        
        logger.info("Directory setup completed")
        return True
    except Exception as e:
        logger.error(f"Error setting up directories: {str(e)}")
        return False

def collect_data(symbols=None, period='5y'):
    """Collect historical commodity price data."""
    try:
        # Add project root to path to ensure imports work
        sys.path.append(PROJECT_ROOT)
        
        from src.data_processing.data_collector import CommodityDataCollector
        
        logger.info(f"Collecting data for symbols: {symbols}, period: {period}")
        collector = CommodityDataCollector(data_dir=os.path.join(PROJECT_ROOT, 'data'))
        
        # Fetch data
        raw_data = collector.fetch_historical_data(symbols=symbols, period=period)
        
        # Preprocess data
        processed_data = collector.preprocess_data(raw_data)
        
        # Generate summary
        summary = collector.get_data_summary()
        logger.info(f"Data collection summary: {summary}")
        
        return summary
    except ImportError as e:
        logger.error(f"Import error: {str(e)}. Make sure all dependencies are installed.")
        print(f"Error: {str(e)}. Make sure all dependencies are installed.")
        return None
    except Exception as e:
        logger.error(f"Error collecting data: {str(e)}")
        print(f"Error collecting data: {str(e)}")
        return None

def train_models(commodity=None):
    """Train time series forecasting models."""
    try:
        # Add project root to path to ensure imports work
        sys.path.append(PROJECT_ROOT)
        
        from src.model_training.time_series_models import TimeSeriesModels
        import pandas as pd
        
        ts_models = TimeSeriesModels(models_dir=os.path.join(PROJECT_ROOT, 'models'))
        
        # Get list of commodities to process
        commodities = []
        if commodity:
            commodities = [commodity]
        else:
            processed_dir = os.path.join(PROJECT_ROOT, 'data/processed')
            if os.path.exists(processed_dir):
                for filename in os.listdir(processed_dir):
                    if filename.endswith('_processed.csv'):
                        commodity_name = filename.replace('_processed.csv', '').replace('_', ' ').title()
                        commodities.append(commodity_name)
        
        if not commodities:
            logger.warning("No commodity data found. Please run data collection first.")
            print("No commodity data found. Please run data collection first.")
            return {}
            
        logger.info(f"Training models for commodities: {commodities}")
        
        results = {}
        for commodity_name in commodities:
            logger.info(f"Processing {commodity_name}")
            
            # Load data
            file_path = os.path.join(PROJECT_ROOT, f"data/processed/{commodity_name.lower().replace(' ', '_')}_processed.csv")
            if not os.path.exists(file_path):
                logger.warning(f"Data file not found for {commodity_name}: {file_path}")
                continue
                
            try:
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
            except Exception as e:
                logger.error(f"Error training models for {commodity_name}: {str(e)}")
                print(f"Error training models for {commodity_name}: {str(e)}")
                results[commodity_name] = {"error": str(e)}
        
        return results
    except ImportError as e:
        logger.error(f"Import error: {str(e)}. Make sure all dependencies are installed.")
        print(f"Error: {str(e)}. Make sure all dependencies are installed.")
        return {}
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        print(f"Error training models: {str(e)}")
        return {}

def run_dashboard():
    """Run the Streamlit dashboard."""
    try:
        # Add project root to path to ensure imports work
        sys.path.append(PROJECT_ROOT)
        
        import subprocess
        
        logger.info("Starting dashboard")
        
        # Get the dashboard path
        dashboard_path = os.path.join(PROJECT_ROOT, "src/visualization/dashboard.py")
        
        if not os.path.exists(dashboard_path):
            logger.error(f"Dashboard file not found: {dashboard_path}")
            print(f"Error: Dashboard file not found: {dashboard_path}")
            return None
        
        # Run the Streamlit app with absolute path
        cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path, "--server.headless", "true"]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        
        logger.info(f"Dashboard started with PID {process.pid}")
        
        return process
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        print(f"Error running dashboard: {str(e)}")
        return None

def main():
    """Main function to parse arguments and run the application."""
    try:
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
        os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
        
        # Execute the appropriate command
        if args.command == 'setup':
            if setup_directories():
                logger.info("Setup completed successfully")
                print("Setup completed successfully")
            else:
                logger.error("Setup failed")
                print("Setup failed")
        
        elif args.command == 'collect':
            if not setup_directories():
                logger.error("Setup failed, cannot proceed with data collection")
                print("Setup failed, cannot proceed with data collection")
                return
                
            summary = collect_data(symbols=args.symbols, period=args.period)
            if summary:
                logger.info("Data collection completed successfully")
                print("Data Collection Summary:")
                for commodity in summary.get('commodities', []):
                    print(f"  {commodity['name']}: {commodity['records']} records from {commodity['start_date']} to {commodity['end_date']}")
            else:
                logger.error("Data collection failed")
                print("Data collection failed")
        
        elif args.command == 'train':
            if not setup_directories():
                logger.error("Setup failed, cannot proceed with model training")
                print("Setup failed, cannot proceed with model training")
                return
                
            results = train_models(commodity=args.commodity)
            if results:
                logger.info("Model training completed successfully")
                print("Model Training Results:")
                for commodity, models in results.items():
                    print(f"\n{commodity}:")
                    if "error" in models:
                        print(f"  Error: {models['error']}")
                    else:
                        for model_type, metrics in models.items():
                            if 'rmse' in metrics and 'mape' in metrics:
                                print(f"  {model_type.upper()}: RMSE = {metrics['rmse']:.4f}, MAPE = {metrics['mape']:.2f}%")
            else:
                logger.error("Model training failed")
                print("Model training failed")
        
        elif args.command == 'dashboard':
            if not setup_directories():
                logger.error("Setup failed, cannot proceed with dashboard")
                print("Setup failed, cannot proceed with dashboard")
                return
                
            process = run_dashboard()
            if process:
                print("Dashboard is running. Press Ctrl+C to stop.")
                try:
                    process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    print("Dashboard stopped.")
            else:
                logger.error("Failed to start dashboard")
                print("Failed to start dashboard")
        
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Unexpected error in main function: {str(e)}")
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 