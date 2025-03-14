# Commodity Market Price Prediction & Analysis Platform

This project implements an end-to-end machine learning pipeline for predicting commodity price movements and analyzing market trends. It leverages various AI/ML techniques including time series forecasting, deep learning, and data visualization to provide insights for trading decisions.

## Features

- Historical price data collection and synthetic data generation for multiple commodities
- Advanced time series forecasting using statistical and deep learning models:
  - ARIMA/SARIMA for statistical time series forecasting
  - Prophet for trend and seasonality decomposition
  - LSTM networks for sequence modeling
- Interactive dashboards for visualizing predictions and market trends
- Performance evaluation of different trading strategies

## Project Structure

```
commodity_price_predictor/
├── data/                      # Data storage
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data files
├── logs/                      # Application logs
├── models/                    # Trained models
├── notebooks/                 # Jupyter notebooks
├── src/                       # Source code
│   ├── data_processing/       # Data collection and processing
│   ├── model_training/        # Model training and evaluation
│   └── visualization/         # Dashboard and visualization
├── main.py                    # Main application script
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

### Prerequisites

- Python 3.9+ (recommended)
- Conda or pip for package management

### Using Conda (Recommended)

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/commodity_price_predictor.git
   cd commodity_price_predictor
   ```

2. Create a new Conda environment:
   ```
   conda create -n commodity python=3.9
   conda activate commodity
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Using pip

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/commodity_price_predictor.git
   cd commodity_price_predictor
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

The application provides a command-line interface for running different components:

### 1. Setup

Create necessary directories:

```
python main.py setup
```

### 2. Data Collection

Generate synthetic data for commodities:

```
python main.py collect
```

To specify particular commodities:

```
python main.py collect --symbols "Gold" "Silver" "Crude Oil"
```

### 3. Model Training

Train forecasting models for all commodities:

```
python main.py train
```

Train models for a specific commodity:

```
python main.py train --commodity "Gold"
```

### 4. Dashboard

Run the interactive dashboard:

```
python main.py dashboard
```

The dashboard will be available at http://localhost:8501

## Troubleshooting

### Common Issues

1. **Missing directories**: Run `python main.py setup` to create all necessary directories.

2. **Import errors**: Make sure you're running commands from the project root directory.

3. **Dependency issues**: Try installing dependencies one by one if you encounter issues with the requirements file.

4. **Dashboard not starting**: Check if Streamlit is properly installed and try running `streamlit hello` to verify.

### Logs

Check the logs directory for detailed error messages:

```
cat logs/app_YYYYMMDD_HHMMSS.log
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 