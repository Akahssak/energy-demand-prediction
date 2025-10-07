import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking URI (local)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Energy_Demand_Prediction")

def load_and_preprocess_data():
    """
    Load energy consumption data
    For this demo, we'll create synthetic data similar to PJM hourly data
    In real scenario, download from: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
    """
    print("Loading and preprocessing data...")
    
    # Create synthetic hourly energy data for 2 years
    date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='H')
    
    # Generate realistic energy demand pattern
    np.random.seed(42)
    n = len(date_range)
    
    # Base load + daily pattern + weekly pattern + noise
    hour_of_day = date_range.hour
    day_of_week = date_range.dayofweek
    month = date_range.month
    
    # Create realistic energy demand pattern
    base_load = 15000
    daily_pattern = 3000 * np.sin(2 * np.pi * hour_of_day / 24)
    weekly_pattern = 1500 * (day_of_week < 5).astype(int)  # Higher on weekdays
    seasonal_pattern = 2000 * np.sin(2 * np.pi * month / 12)
    noise = np.random.normal(0, 500, n)
    
    energy_demand = base_load + daily_pattern + weekly_pattern + seasonal_pattern + noise
    
    df = pd.DataFrame({
        'Datetime': date_range,
        'Energy_Demand_MW': energy_demand
    })
    
    return df

def create_features(df):
    """Create time-based features for modeling"""
    df = df.copy()
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month
    df['DayOfYear'] = df['Datetime'].dt.dayofyear
    df['WeekOfYear'] = df['Datetime'].dt.isocalendar().week
    df['Quarter'] = df['Datetime'].dt.quarter
    
    # Lag features
    df['Lag_1h'] = df['Energy_Demand_MW'].shift(1)
    df['Lag_24h'] = df['Energy_Demand_MW'].shift(24)
    df['Lag_168h'] = df['Energy_Demand_MW'].shift(168)  # 1 week
    
    # Rolling statistics
    df['Rolling_Mean_24h'] = df['Energy_Demand_MW'].rolling(window=24).mean()
    df['Rolling_Std_24h'] = df['Energy_Demand_MW'].rolling(window=24).std()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model with MLflow tracking"""
    print("\nTraining Random Forest Model...")
    
    with mlflow.start_run(run_name="RandomForest_Model"):
        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Random Forest - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        return model, y_pred, mae, rmse, r2

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting model with MLflow tracking"""
    print("\nTraining Gradient Boosting Model...")
    
    with mlflow.start_run(run_name="GradientBoosting_Model"):
        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Gradient Boosting - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        return model, y_pred, mae, rmse, r2

def main():
    print("="*60)
    print("Energy Demand Prediction - MLflow Training Pipeline")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    
    # Create features
    df_features = create_features(df)
    print(f"Features created. New shape: {df_features.shape}")
    
    # Prepare data for modeling
    feature_cols = ['Hour', 'DayOfWeek', 'Month', 'DayOfYear', 'WeekOfYear', 
                    'Quarter', 'Lag_1h', 'Lag_24h', 'Lag_168h', 
                    'Rolling_Mean_24h', 'Rolling_Std_24h']
    
    X = df_features[feature_cols]
    y = df_features['Energy_Demand_MW']
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models
    rf_model, rf_pred, rf_mae, rf_rmse, rf_r2 = train_random_forest(X_train, X_test, y_train, y_test)
    gb_model, gb_pred, gb_mae, gb_rmse, gb_r2 = train_gradient_boosting(X_train, X_test, y_train, y_test)
    
    # Save the best model
    if rf_rmse < gb_rmse:
        best_model = rf_model
        best_name = "RandomForest"
        best_rmse = rf_rmse
    else:
        best_model = gb_model
        best_name = "GradientBoosting"
        best_rmse = gb_rmse
    
    print("\n" + "="*60)
    print(f"Best Model: {best_name} with RMSE: {best_rmse:.2f}")
    print("="*60)
    
    # Save test data and predictions for visualization
    test_results = df_features.iloc[-len(y_test):].copy()
    test_results['RF_Prediction'] = rf_pred
    test_results['GB_Prediction'] = gb_pred
    test_results.to_csv('test_predictions.csv', index=False)
    
    # Save the full dataset for the Streamlit app
    df_features.to_csv('energy_data.csv', index=False)
    
    print("\nTraining completed! Files saved:")
    print("- test_predictions.csv")
    print("- energy_data.csv")
    print("- MLflow runs logged in ./mlruns")

if __name__ == "__main__":
    main()