"""
# Data Collector Module

# This module handles the collection of historical commodity price data from various sources.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Union
import json
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define commodity symbols
COMMODITY_SYMBOLS = {
    'Gold': 'GLD',
    'Silver': 'SLV',
    'Crude Oil': 'USO',
    'Natural Gas': 'UNG',
    'Copper': 'CPER'
}

class CommodityDataCollector:
    """Class for collecting and processing commodity price data."""
    
    def __init__(self, data_dir: str = '../data'):
        """
        Initialize the data collector.
        
        Args:
            data_dir: Directory to store the collected data
        """
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, 'raw')
        self.processed_data_dir = os.path.join(data_dir, 'processed')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        logger.info(f"Initialized CommodityDataCollector with data directory: {data_dir}")
    
    def fetch_historical_data(
        self, 
        symbols: Optional[List[str]] = None, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: Optional[str] = '5y',
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate sample historical price data for specified commodities.
        
        Args:
            symbols: List of commodity symbols to fetch. If None, fetch all defined commodities.
            start_date: Start date in 'YYYY-MM-DD' format. If None, use period instead.
            end_date: End date in 'YYYY-MM-DD' format. If None, use current date.
            period: Time period to fetch if start_date is None (e.g., '1y', '6mo', '5y').
            interval: Data interval ('1d', '1wk', '1mo', etc.).
            
        Returns:
            Dictionary mapping commodity names to their historical price DataFrames.
        """
        logger.info("Using sample data generation instead of Yahoo Finance API")
        
        if symbols is None:
            symbols = list(COMMODITY_SYMBOLS.keys())
        
        # Determine date range
        end_date_dt = datetime.now()
        if period == '5y':
            start_date_dt = end_date_dt - timedelta(days=5*365)
        elif period == '1y':
            start_date_dt = end_date_dt - timedelta(days=365)
        elif period == '6mo':
            start_date_dt = end_date_dt - timedelta(days=180)
        else:
            start_date_dt = end_date_dt - timedelta(days=365)
        
        # Override with explicit dates if provided
        if start_date:
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate date range
        if interval == '1d':
            dates = pd.date_range(start=start_date_dt, end=end_date_dt, freq='D')
        elif interval == '1wk':
            dates = pd.date_range(start=start_date_dt, end=end_date_dt, freq='W')
        elif interval == '1mo':
            dates = pd.date_range(start=start_date_dt, end=end_date_dt, freq='M')
        else:
            dates = pd.date_range(start=start_date_dt, end=end_date_dt, freq='D')
        
        data_dict = {}
        
        # Generate sample data for each commodity
        for commodity_name in symbols:
            try:
                logger.info(f"Generating sample data for {commodity_name}")
                
                # Set base price and volatility based on commodity
                if commodity_name == 'Gold':
                    base_price = 1800.0
                    volatility = 0.01
                    trend = 0.0001
                elif commodity_name == 'Silver':
                    base_price = 25.0
                    volatility = 0.015
                    trend = 0.0002
                elif commodity_name == 'Crude Oil':
                    base_price = 70.0
                    volatility = 0.02
                    trend = -0.0001
                elif commodity_name == 'Natural Gas':
                    base_price = 3.0
                    volatility = 0.025
                    trend = 0.0003
                else:  # Default/Copper
                    base_price = 4.0
                    volatility = 0.018
                    trend = 0.0002
                
                # Generate price series with random walk
                np.random.seed(42 + hash(commodity_name) % 100)  # Different seed for each commodity
                returns = np.random.normal(trend, volatility, len(dates))
                price_series = [base_price]
                
                for ret in returns:
                    price_series.append(price_series[-1] * (1 + ret))
                
                price_series = price_series[1:]  # Remove the initial base price
                
                # Create DataFrame
                df = pd.DataFrame({
                    'Date': dates[:len(price_series)],
                    'Open': price_series,
                    'High': [p * (1 + np.random.uniform(0, 0.01)) for p in price_series],
                    'Low': [p * (1 - np.random.uniform(0, 0.01)) for p in price_series],
                    'Close': [p * (1 + np.random.normal(0, 0.005)) for p in price_series],
                    'Adj Close': [p * (1 + np.random.normal(0, 0.005)) for p in price_series],
                    'Volume': [int(np.random.uniform(100000, 1000000)) for _ in price_series]
                })
                
                df = df.set_index('Date')
                
                # Store the data
                data_dict[commodity_name] = df
                
                # Save raw data to CSV
                raw_file_path = os.path.join(
                    self.raw_data_dir, 
                    f"{commodity_name.replace(' ', '_').lower()}_{interval}.csv"
                )
                df.to_csv(raw_file_path)
                logger.info(f"Saved sample data for {commodity_name} to {raw_file_path}")
                
            except Exception as e:
                logger.error(f"Error generating sample data for {commodity_name}: {str(e)}")
        
        return data_dict
    
    def preprocess_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess the raw commodity data.
        
        Args:
            data_dict: Dictionary mapping commodity names to their raw price DataFrames.
            
        Returns:
            Dictionary mapping commodity names to their processed DataFrames.
        """
        processed_data = {}
        
        for commodity_name, df in data_dict.items():
            try:
                logger.info(f"Preprocessing data for {commodity_name}")
                
                # Make a copy to avoid modifying the original
                processed_df = df.copy()
                
                # Reset index to make Date a column
                processed_df = processed_df.reset_index()
                
                # Handle missing values
                processed_df = processed_df.fillna(method='ffill')
                
                # Add technical indicators
                # 1. Moving Averages
                processed_df['MA_7'] = processed_df['Close'].rolling(window=7).mean()
                processed_df['MA_21'] = processed_df['Close'].rolling(window=21).mean()
                processed_df['MA_50'] = processed_df['Close'].rolling(window=50).mean()
                
                # 2. Volatility (standard deviation of returns)
                processed_df['Returns'] = processed_df['Close'].pct_change()
                processed_df['Volatility_21'] = processed_df['Returns'].rolling(window=21).std()
                
                # 3. Price momentum
                processed_df['Momentum_14'] = processed_df['Close'] - processed_df['Close'].shift(14)
                
                # Drop rows with NaN values after creating features
                processed_df = processed_df.dropna()
                
                # Save processed data
                processed_file_path = os.path.join(
                    self.processed_data_dir,
                    f"{commodity_name.replace(' ', '_').lower()}_processed.csv"
                )
                processed_df.to_csv(processed_file_path, index=False)
                logger.info(f"Saved processed data for {commodity_name} to {processed_file_path}")
                
                processed_data[commodity_name] = processed_df
                
            except Exception as e:
                logger.error(f"Error preprocessing data for {commodity_name}: {str(e)}")
        
        return processed_data
    
    def get_data_summary(self) -> Dict:
        """
        Generate a summary of the available data.
        
        Returns:
            Dictionary with summary information about the collected data.
        """
        summary = {
            "commodities": [],
            "total_records": 0,
            "date_range": {}
        }
        
        for filename in os.listdir(self.processed_data_dir):
            if filename.endswith('_processed.csv'):
                try:
                    commodity_name = filename.replace('_processed.csv', '').replace('_', ' ').title()
                    file_path = os.path.join(self.processed_data_dir, filename)
                    
                    df = pd.read_csv(file_path)
                    
                    # Get date range
                    df['Date'] = pd.to_datetime(df['Date'])
                    start_date = df['Date'].min().strftime('%Y-%m-%d')
                    end_date = df['Date'].max().strftime('%Y-%m-%d')
                    
                    commodity_summary = {
                        "name": commodity_name,
                        "records": len(df),
                        "start_date": start_date,
                        "end_date": end_date,
                        "features": list(df.columns)
                    }
                    
                    summary["commodities"].append(commodity_summary)
                    summary["total_records"] += len(df)
                    
                except Exception as e:
                    logger.error(f"Error generating summary for {filename}: {str(e)}")
        
        if summary["commodities"]:
            all_start_dates = [c["start_date"] for c in summary["commodities"]]
            all_end_dates = [c["end_date"] for c in summary["commodities"]]
            
            summary["date_range"] = {
                "earliest": min(all_start_dates),
                "latest": max(all_end_dates)
            }
        
        return summary

def main():
    """Main function to demonstrate the data collection process."""
    collector = CommodityDataCollector(data_dir='../../data')
    
    # Generate sample data for all commodities
    raw_data = collector.fetch_historical_data(period='5y')
    
    # Preprocess the data
    processed_data = collector.preprocess_data(raw_data)
    
    # Generate and print summary
    summary = collector.get_data_summary()
    print(json.dumps(summary, indent=2))
    
    logger.info("Data collection and preprocessing completed successfully")

if __name__ == "__main__":
    main() 