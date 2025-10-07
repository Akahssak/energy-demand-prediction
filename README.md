# ⚡ Energy Demand Prediction - MLOps Project

A complete MLOps project for time series forecasting of energy demand using MLflow for experiment tracking and Streamlit for visualization.

## 🎯 Project Overview

This project demonstrates:
- **Time series forecasting** for energy demand prediction
- **MLflow integration** for experiment tracking, model logging, and metrics comparison
- **Multiple ML models**: Random Forest and Gradient Boosting
- **Interactive dashboard** built with Streamlit
- **Production-ready deployment** on Streamlit Cloud

## 📊 Dataset

The project uses synthetic hourly energy consumption data (mimicking real-world patterns from PJM Interconnection):
- **Duration**: 2 years of hourly data
- **Features**: Time-based features, lag features, rolling statistics
- **Target**: Energy demand in Megawatts (MW)

For real-world data, you can use the [PJM Hourly Energy Consumption dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) from Kaggle.

## 🛠️ Technologies Used

- **Python 3.8+**
- **MLflow**: Experiment tracking and model registry
- **Scikit-learn**: Machine learning models
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data manipulation

## 📁 Project Structure

```
energy-demand-prediction/
├── train_model.py           # Training pipeline with MLflow
├── app.py                   # Streamlit dashboard
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore file
├── energy_data.csv         # Generated dataset (after training)
├── test_predictions.csv    # Model predictions (after training)
└── mlruns/                 # MLflow experiment logs (after training)
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd energy-demand-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Models

Run the training pipeline to generate data and train models:

```bash
python train_model.py
```

This will:
- Generate synthetic energy demand data
- Create time-based features
- Train Random Forest and Gradient Boosting models
- Log experiments, parameters, and metrics to MLflow
- Save predictions and data for visualization

### 4. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📈 Features

### Training Pipeline (`train_model.py`)
- **Data Generation**: Creates realistic synthetic energy demand data
- **Feature Engineering**: Time-based features, lag features, rolling statistics
- **Model Training**: Random Forest and Gradient Boosting
- **MLflow Tracking**: Logs parameters, metrics, and models
- **Evaluation**: MAE, RMSE, R² Score

### Dashboard (`app.py`)
1. **Overview Page**
   - Dataset statistics
   - Time series visualization
   - Hourly and weekly demand patterns

2. **Predictions Page**
   - Model predictions vs actual values
   - Detailed prediction table
   - Downloadable results

3. **Model Performance Page**
   - MLflow experiment metrics
   - Model comparison charts
   - Best model identification

4. **About Page**
   - Project documentation
   - Technical details

## 📊 Models & Metrics

### Models Implemented
1. **Random Forest Regressor**
   - n_estimators: 100
   - max_depth: 15
   - min_samples_split: 5

2. **Gradient Boosting Regressor**
   - n_estimators: 100
   - max_depth: 5
   - learning_rate: 0.1

### Evaluation Metrics
- **MAE** (Mean Absolute Error): Average magnitude of errors
- **RMSE** (Root Mean Squared Error): Penalizes larger errors more
- **R² Score**: Proportion of variance explained by the model

## 🌐 Deployment on Streamlit Cloud

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Energy Demand Prediction MLOps project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `app.py`
6. Click "Deploy"

**Important**: Before deploying, run `train_model.py` locally and commit the generated files:
- `energy_data.csv`
- `test_predictions.csv`
- `mlruns/` folder

## 📝 MLflow Experiment Tracking

### View MLflow UI Locally

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser to explore:
- All experiment runs
- Parameters and metrics comparison
- Model artifacts
- Run history

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ End-to-end MLOps pipeline
- ✅ Experiment tracking with MLflow
- ✅ Multiple model comparison
- ✅ Time series feature engineering
- ✅ Interactive dashboard creation
- ✅ Git version control
- ✅ Cloud deployment

## 📊 Expected Results

Based on synthetic data, you should see:
- **Random Forest**: RMSE ~400-600 MW, R² ~0.85-0.90
- **Gradient Boosting**: RMSE ~450-650 MW, R² ~0.83-0.88

Results will vary with random seed and data patterns.

## 🔧 Customization

### Using Real Data

Replace the `load_and_preprocess_data()` function in `train_model.py`:

```python
def load_and_preprocess_data():
    # Download from Kaggle: hourly-energy-consumption
    df = pd.read_csv('PJME_hourly.csv')
    df.columns = ['Datetime', 'Energy_Demand_MW']
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df
```

### Adding More Models

Add new model training functions following the existing pattern:

```python
def train_xgboost(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="XGBoost_Model"):
        # Your implementation
        pass
```

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

### Issue: No data found
**Solution**: Run training script first
```bash
python train_model.py
```

### Issue: MLflow UI not showing runs
**Solution**: Check that mlruns folder exists
```bash
ls mlruns/
```

## 📚 References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Time Series](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

## 👨‍💻 Author

Created as an MLOps course project demonstrating:
- Machine Learning pipeline
- Experiment tracking
- Model deployment
- Interactive visualization

## 📄 License

This project is for educational purposes.

---

**🎉 Ready to submit? Make sure to:**
1. ✅ Run `python train_model.py`
2. ✅ Test `streamlit run app.py` locally
3. ✅ Commit all files including `mlruns/`, CSV files
4. ✅ Push to GitHub
5. ✅ Deploy on Streamlit Cloud

**Happy Learning! 🚀*