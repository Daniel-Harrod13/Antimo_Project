"""
Time Series Models Module

This module implements various time series forecasting models for commodity price prediction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
import logging
import joblib
from datetime import datetime, timedelta

# Statistical models
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from prophet import Prophet

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesModels:
    """Class for training and evaluating time series forecasting models."""
    
    def __init__(self, models_dir: str = '../models'):
        """
        Initialize the time series models.
        
        Args:
            models_dir: Directory to store trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Dictionary to store trained models
        self.trained_models = {}
        
        logger.info(f"Initialized TimeSeriesModels with models directory: {models_dir}")
    
    def train_arima_model(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'Close',
        test_size: float = 0.2,
        auto_order: bool = True,
        order: Tuple[int, int, int] = (5, 1, 0),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        commodity_name: str = 'commodity'
    ) -> Dict:
        """
        Train an ARIMA or SARIMA model for time series forecasting.
        
        Args:
            data: DataFrame containing the time series data
            target_col: Column name of the target variable
            test_size: Proportion of data to use for testing
            auto_order: Whether to automatically determine the order using pmdarima
            order: ARIMA order (p, d, q) if auto_order is False
            seasonal_order: Seasonal order (P, D, Q, s) for SARIMA
            commodity_name: Name of the commodity for model identification
            
        Returns:
            Dictionary with model information and evaluation metrics
        """
        logger.info(f"Training ARIMA model for {commodity_name}")
        
        # Prepare the data
        y = data[target_col].values
        train_size = int(len(y) * (1 - test_size))
        train, test = y[:train_size], y[train_size:]
        
        model_info = {
            'model_type': 'SARIMA' if seasonal_order else 'ARIMA',
            'commodity': commodity_name,
            'target_column': target_col,
            'train_size': train_size,
            'test_size': len(test),
            'metrics': {}
        }
        
        try:
            # Determine the best order if auto_order is True
            if auto_order:
                logger.info("Automatically determining ARIMA order")
                auto_arima = pm.auto_arima(
                    train,
                    seasonal=seasonal_order is not None,
                    m=seasonal_order[-1] if seasonal_order else None,
                    suppress_warnings=True,
                    error_action='ignore',
                    stepwise=True
                )
                order = auto_arima.order
                seasonal_order = auto_arima.seasonal_order
                
                model_info['order'] = order
                model_info['seasonal_order'] = seasonal_order
                logger.info(f"Auto-determined order: {order}, seasonal_order: {seasonal_order}")
            
            # Train the model
            if seasonal_order:
                model = SARIMAX(
                    train, 
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fitted = model.fit(disp=False)
            else:
                model = ARIMA(train, order=order)
                model_fitted = model.fit()
            
            # Make predictions
            forecast_steps = len(test)
            forecast = model_fitted.forecast(steps=forecast_steps)
            
            # Calculate metrics
            mse = np.mean((test - forecast) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((test - forecast) / test)) * 100
            
            model_info['metrics'] = {
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            }
            
            # Save the model
            model_name = f"{commodity_name.lower().replace(' ', '_')}_{model_info['model_type'].lower()}"
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            joblib.dump(model_fitted, model_path)
            
            model_info['model_path'] = model_path
            self.trained_models[model_name] = model_info
            
            logger.info(f"ARIMA model for {commodity_name} trained successfully. RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error training ARIMA model for {commodity_name}: {str(e)}")
            model_info['error'] = str(e)
            return model_info
    
    def train_prophet_model(
        self, 
        data: pd.DataFrame,
        date_col: str = 'Date',
        target_col: str = 'Close',
        test_size: float = 0.2,
        forecast_periods: int = 30,
        commodity_name: str = 'commodity',
        include_components: bool = True
    ) -> Dict:
        """
        Train a Prophet model for time series forecasting.
        
        Args:
            data: DataFrame containing the time series data
            date_col: Column name of the date variable
            target_col: Column name of the target variable
            test_size: Proportion of data to use for testing
            forecast_periods: Number of periods to forecast
            commodity_name: Name of the commodity for model identification
            include_components: Whether to include trend and seasonality components
            
        Returns:
            Dictionary with model information and evaluation metrics
        """
        logger.info(f"Training Prophet model for {commodity_name}")
        
        model_info = {
            'model_type': 'Prophet',
            'commodity': commodity_name,
            'target_column': target_col,
            'forecast_periods': forecast_periods,
            'metrics': {}
        }
        
        try:
            # Prepare the data for Prophet (needs 'ds' and 'y' columns)
            prophet_data = data[[date_col, target_col]].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Split into train and test
            train_size = int(len(prophet_data) * (1 - test_size))
            train_data = prophet_data.iloc[:train_size]
            test_data = prophet_data.iloc[train_size:]
            
            model_info['train_size'] = len(train_data)
            model_info['test_size'] = len(test_data)
            
            # Initialize and train the model
            prophet_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode='multiplicative'
            )
            
            prophet_model.fit(train_data)
            
            # Create a dataframe for future predictions
            future = prophet_model.make_future_dataframe(periods=len(test_data))
            forecast = prophet_model.predict(future)
            
            # Extract predictions for the test period
            test_forecast = forecast.iloc[-len(test_data):]
            
            # Calculate metrics
            y_true = test_data['y'].values
            y_pred = test_forecast['yhat'].values
            
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            model_info['metrics'] = {
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            }
            
            # Save the model
            model_name = f"{commodity_name.lower().replace(' ', '_')}_prophet"
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            joblib.dump(prophet_model, model_path)
            
            model_info['model_path'] = model_path
            self.trained_models[model_name] = model_info
            
            # Generate component plots if requested
            if include_components:
                components_path = os.path.join(self.models_dir, f"{model_name}_components.png")
                fig = prophet_model.plot_components(forecast)
                fig.savefig(components_path)
                model_info['components_plot'] = components_path
            
            logger.info(f"Prophet model for {commodity_name} trained successfully. RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error training Prophet model for {commodity_name}: {str(e)}")
            model_info['error'] = str(e)
            return model_info
    
    def _prepare_lstm_data(
        self, 
        data: np.ndarray, 
        look_back: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model by creating sequences.
        
        Args:
            data: Input time series data
            look_back: Number of previous time steps to use as input features
            
        Returns:
            Tuple of input sequences (X) and target values (y)
        """
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)
    
    def train_lstm_model(
        self, 
        data: pd.DataFrame,
        target_col: str = 'Close',
        test_size: float = 0.2,
        look_back: int = 60,
        lstm_units: int = 50,
        dropout_rate: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        commodity_name: str = 'commodity'
    ) -> Dict:
        """
        Train an LSTM model for time series forecasting.
        
        Args:
            data: DataFrame containing the time series data
            target_col: Column name of the target variable
            test_size: Proportion of data to use for testing
            look_back: Number of previous time steps to use as input features
            lstm_units: Number of LSTM units in the model
            dropout_rate: Dropout rate for regularization
            epochs: Number of training epochs
            batch_size: Batch size for training
            commodity_name: Name of the commodity for model identification
            
        Returns:
            Dictionary with model information and evaluation metrics
        """
        logger.info(f"Training LSTM model for {commodity_name}")
        
        model_info = {
            'model_type': 'LSTM',
            'commodity': commodity_name,
            'target_column': target_col,
            'look_back': look_back,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'metrics': {}
        }
        
        try:
            # Extract the target column and scale the data
            data_values = data[target_col].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_values)
            
            # Prepare sequences for LSTM
            X, y = self._prepare_lstm_data(scaled_data, look_back)
            
            # Split into train and test sets
            train_size = int(len(X) * (1 - test_size))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            model_info['train_size'] = train_size
            model_info['test_size'] = len(X_test)
            
            # Reshape input for LSTM [samples, time steps, features]
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(look_back, 1)))
            model.add(Dropout(dropout_rate))
            model.add(LSTM(units=lstm_units))
            model.add(Dropout(dropout_rate))
            model.add(Dense(units=1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Define callbacks
            model_name = f"{commodity_name.lower().replace(' ', '_')}_lstm"
            model_path = os.path.join(self.models_dir, f"{model_name}.h5")
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss')
            ]
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Inverse transform to original scale
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_inv = scaler.inverse_transform(y_pred)
            
            # Calculate metrics
            mse = np.mean((y_test_inv - y_pred_inv) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
            
            model_info['metrics'] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape)
            }
            
            # Save the scaler for later use
            scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            
            model_info['model_path'] = model_path
            model_info['scaler_path'] = scaler_path
            self.trained_models[model_name] = model_info
            
            logger.info(f"LSTM model for {commodity_name} trained successfully. RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error training LSTM model for {commodity_name}: {str(e)}")
            model_info['error'] = str(e)
            return model_info
    
    def forecast_future(
        self, 
        model_name: str,
        periods: int = 30,
        data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Generate future forecasts using a trained model.
        
        Args:
            model_name: Name of the trained model to use
            periods: Number of periods to forecast
            data: Recent data to use for forecasting (required for LSTM)
            
        Returns:
            Dictionary with forecast information
        """
        if model_name not in self.trained_models:
            logger.error(f"Model {model_name} not found in trained models")
            return {'error': f"Model {model_name} not found"}
        
        model_info = self.trained_models[model_name]
        model_type = model_info['model_type']
        
        forecast_result = {
            'model_name': model_name,
            'model_type': model_type,
            'periods': periods,
            'forecast': None,
            'dates': None
        }
        
        try:
            if model_type == 'ARIMA' or model_type == 'SARIMA':
                # Load the ARIMA/SARIMA model
                model = joblib.load(model_info['model_path'])
                
                # Generate forecast
                forecast = model.forecast(steps=periods)
                forecast_result['forecast'] = forecast.tolist()
                
                # Generate dates
                last_date = datetime.now()
                dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
                forecast_result['dates'] = [d.strftime('%Y-%m-%d') for d in dates]
                
            elif model_type == 'Prophet':
                # Load the Prophet model
                model = joblib.load(model_info['model_path'])
                
                # Create future dataframe
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)
                
                # Extract the forecast for the future periods
                future_forecast = forecast.iloc[-periods:]
                
                forecast_result['forecast'] = future_forecast['yhat'].tolist()
                forecast_result['lower_bound'] = future_forecast['yhat_lower'].tolist()
                forecast_result['upper_bound'] = future_forecast['yhat_upper'].tolist()
                forecast_result['dates'] = future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
                
            elif model_type == 'LSTM':
                if data is None:
                    return {'error': "Recent data is required for LSTM forecasting"}
                
                # Load the LSTM model and scaler
                model = load_model(model_info['model_path'])
                scaler = joblib.load(model_info['scaler_path'])
                
                # Prepare the most recent data
                look_back = model_info['look_back']
                target_col = model_info['target_column']
                
                recent_data = data[target_col].values[-look_back:].reshape(-1, 1)
                scaled_data = scaler.transform(recent_data)
                
                # Initialize arrays for forecasting
                input_data = scaled_data.copy()
                forecasts = []
                
                # Generate forecasts one step at a time
                for _ in range(periods):
                    # Reshape for LSTM input [1, look_back, 1]
                    x_input = input_data[-look_back:].reshape(1, look_back, 1)
                    
                    # Predict the next value
                    next_pred = model.predict(x_input)
                    
                    # Append to forecasts
                    forecasts.append(next_pred[0, 0])
                    
                    # Update input data for next prediction
                    input_data = np.append(input_data, next_pred)
                    input_data = input_data.reshape(-1, 1)
                
                # Inverse transform to original scale
                forecast_values = scaler.inverse_transform(
                    np.array(forecasts).reshape(-1, 1)
                )
                
                forecast_result['forecast'] = forecast_values.flatten().tolist()
                
                # Generate dates
                last_date = datetime.now()
                dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
                forecast_result['dates'] = [d.strftime('%Y-%m-%d') for d in dates]
            
            logger.info(f"Generated {periods} period forecast using {model_name}")
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error generating forecast with {model_name}: {str(e)}")
            forecast_result['error'] = str(e)
            return forecast_result
    
    def get_model_comparison(self) -> Dict:
        """
        Compare the performance of all trained models.
        
        Returns:
            Dictionary with model comparison information
        """
        comparison = {
            'models': [],
            'best_model_by_commodity': {}
        }
        
        # Group models by commodity
        commodity_models = {}
        for model_name, model_info in self.trained_models.items():
            commodity = model_info['commodity']
            if commodity not in commodity_models:
                commodity_models[commodity] = []
            
            if 'metrics' in model_info and 'rmse' in model_info['metrics']:
                model_summary = {
                    'model_name': model_name,
                    'model_type': model_info['model_type'],
                    'commodity': commodity,
                    'rmse': model_info['metrics']['rmse'],
                    'mape': model_info['metrics']['mape']
                }
                
                commodity_models[commodity].append(model_summary)
                comparison['models'].append(model_summary)
        
        # Find the best model for each commodity
        for commodity, models in commodity_models.items():
            if models:
                # Sort by RMSE (lower is better)
                best_model = min(models, key=lambda x: x['rmse'])
                comparison['best_model_by_commodity'][commodity] = best_model
        
        return comparison

def main():
    """Main function to demonstrate the time series modeling process."""
    # Example usage
    import pandas as pd
    
    # Load sample data
    data_path = '../../data/processed/gold_processed.csv'
    try:
        data = pd.read_csv(data_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        
        # Initialize models
        ts_models = TimeSeriesModels(models_dir='../../models')
        
        # Train ARIMA model
        arima_info = ts_models.train_arima_model(
            data=data,
            target_col='Close',
            test_size=0.2,
            auto_order=True,
            commodity_name='Gold'
        )
        
        # Train Prophet model
        prophet_info = ts_models.train_prophet_model(
            data=data.reset_index(),
            date_col='Date',
            target_col='Close',
            test_size=0.2,
            forecast_periods=30,
            commodity_name='Gold'
        )
        
        # Train LSTM model
        lstm_info = ts_models.train_lstm_model(
            data=data,
            target_col='Close',
            test_size=0.2,
            look_back=60,
            lstm_units=50,
            epochs=50,
            commodity_name='Gold'
        )
        
        # Compare models
        comparison = ts_models.get_model_comparison()
        print("Model Comparison:")
        for model in comparison['models']:
            print(f"{model['model_name']}: RMSE = {model['rmse']:.4f}, MAPE = {model['mape']:.2f}%")
        
        # Generate forecast with the best model
        best_model = comparison['best_model_by_commodity'].get('Gold', {}).get('model_name')
        if best_model:
            forecast = ts_models.forecast_future(
                model_name=best_model,
                periods=30,
                data=data
            )
            print(f"Forecast using {best_model}:")
            print(f"Next 5 days: {forecast['forecast'][:5]}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 