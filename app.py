import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
import os

# Page configuration
st.set_page_config(
    page_title="Energy Demand Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the energy demand data"""
    try:
        df = pd.read_csv('energy_data.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run train_model.py first!")
        return None

@st.cache_data
def load_predictions():
    """Load test predictions"""
    try:
        df = pd.read_csv('test_predictions.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df
    except FileNotFoundError:
        return None

def get_mlflow_metrics():
    """Get metrics from MLflow experiments"""
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        experiment = mlflow.get_experiment_by_name("Energy_Demand_Prediction")
        
        if experiment is None:
            return None
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if len(runs) == 0:
            return None
        
        # Get metrics for each run
        metrics_data = []
        for _, run in runs.iterrows():
            metrics_data.append({
                'Model': run['tags.mlflow.runName'],
                'MAE': run['metrics.mae'],
                'RMSE': run['metrics.rmse'],
                'R2 Score': run['metrics.r2_score'],
                'Run ID': run['run_id']
            })
        
        return pd.DataFrame(metrics_data)
    except Exception as e:
        st.warning(f"Could not load MLflow metrics: {e}")
        return None

def plot_time_series(df, title="Energy Demand Over Time"):
    """Plot time series data"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Datetime'],
        y=df['Energy_Demand_MW'],
        mode='lines',
        name='Actual Demand',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Energy Demand (MW)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_predictions(df):
    """Plot actual vs predicted values"""
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=df['Datetime'],
        y=df['Energy_Demand_MW'],
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Random Forest predictions
    fig.add_trace(go.Scatter(
        x=df['Datetime'],
        y=df['RF_Prediction'],
        mode='lines',
        name='Random Forest',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Gradient Boosting predictions
    fig.add_trace(go.Scatter(
        x=df['Datetime'],
        y=df['GB_Prediction'],
        mode='lines',
        name='Gradient Boosting',
        line=dict(color='#2ca02c', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title="Model Predictions vs Actual Energy Demand",
        xaxis_title="Date",
        yaxis_title="Energy Demand (MW)",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_hourly_pattern(df):
    """Plot average energy demand by hour of day"""
    hourly_avg = df.groupby('Hour')['Energy_Demand_MW'].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hourly_avg['Hour'],
        y=hourly_avg['Energy_Demand_MW'],
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title="Average Energy Demand by Hour of Day",
        xaxis_title="Hour",
        yaxis_title="Average Energy Demand (MW)",
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_weekly_pattern(df):
    """Plot average energy demand by day of week"""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg = df.groupby('DayOfWeek')['Energy_Demand_MW'].mean().reset_index()
    weekly_avg['Day'] = weekly_avg['DayOfWeek'].apply(lambda x: days[x])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=weekly_avg['Day'],
        y=weekly_avg['Energy_Demand_MW'],
        marker_color='#2ca02c'
    ))
    
    fig.update_layout(
        title="Average Energy Demand by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Average Energy Demand (MW)",
        height=400,
        template='plotly_white'
    )
    
    return fig

def main():
    # Title and description
    st.title("⚡ Energy Demand Prediction Dashboard")
    st.markdown("**MLOps Project**: Time Series Forecasting with MLflow Tracking")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Overview", "🔮 Predictions", "📈 Model Performance", "ℹ️ About"]
    )
    
    # Load data
    df = load_data()
    predictions_df = load_predictions()
    
    if df is None:
        st.error("⚠️ Please run `train_model.py` first to generate the data and train models!")
        st.code("python train_model.py", language="bash")
        return
    
    # Page routing
    if page == "📊 Overview":
        st.header("Dataset Overview")
        
        # Key statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Avg Demand", f"{df['Energy_Demand_MW'].mean():.2f} MW")
        with col3:
            st.metric("Max Demand", f"{df['Energy_Demand_MW'].max():.2f} MW")
        with col4:
            st.metric("Min Demand", f"{df['Energy_Demand_MW'].min():.2f} MW")
        
        st.markdown("---")
        
        # Time series plot
        st.subheader("Energy Demand Time Series")
        date_range = st.slider(
            "Select Date Range",
            min_value=df['Datetime'].min().to_pydatetime(),
            max_value=df['Datetime'].max().to_pydatetime(),
            value=(df['Datetime'].min().to_pydatetime(), 
                   df['Datetime'].min().to_pydatetime() + timedelta(days=30)),
            format="YYYY-MM-DD"
        )
        
        filtered_df = df[(df['Datetime'] >= date_range[0]) & (df['Datetime'] <= date_range[1])]
        st.plotly_chart(plot_time_series(filtered_df), use_container_width=True)
        
        # Patterns
        st.markdown("---")
        st.subheader("Demand Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_hourly_pattern(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_weekly_pattern(df), use_container_width=True)
        
        # Data sample
        st.markdown("---")
        st.subheader("Data Sample")
        st.dataframe(df.head(100), use_container_width=True)
    
    elif page == "🔮 Predictions":
        st.header("Model Predictions")
        
        if predictions_df is None:
            st.warning("No predictions available. Please run the training script first.")
            return
        
        # Prediction plot
        st.plotly_chart(plot_predictions(predictions_df), use_container_width=True)
        
        # Detailed predictions table
        st.markdown("---")
        st.subheader("Prediction Details")
        
        # Calculate errors
        predictions_df['RF_Error'] = abs(predictions_df['Energy_Demand_MW'] - predictions_df['RF_Prediction'])
        predictions_df['GB_Error'] = abs(predictions_df['Energy_Demand_MW'] - predictions_df['GB_Prediction'])
        
        display_cols = ['Datetime', 'Energy_Demand_MW', 'RF_Prediction', 'RF_Error', 
                       'GB_Prediction', 'GB_Error']
        
        st.dataframe(
            predictions_df[display_cols].head(50),
            use_container_width=True
        )
        
        # Download predictions
        st.markdown("---")
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    
    elif page == "📈 Model Performance":
        st.header("Model Performance Metrics")
        
        # Get MLflow metrics
        metrics_df = get_mlflow_metrics()
        
        if metrics_df is not None and len(metrics_df) > 0:
            st.subheader("MLflow Experiment Metrics")
            
            # Display metrics table
            st.dataframe(
                metrics_df[['Model', 'MAE', 'RMSE', 'R2 Score']].style.format({
                    'MAE': '{:.2f}',
                    'RMSE': '{:.2f}',
                    'R2 Score': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # Metrics comparison chart
            st.markdown("---")
            st.subheader("Model Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_mae = px.bar(
                    metrics_df,
                    x='Model',
                    y='MAE',
                    title='Mean Absolute Error (MAE)',
                    color='Model',
                    text='MAE'
                )
                fig_mae.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_mae.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_mae, use_container_width=True)
            
            with col2:
                fig_r2 = px.bar(
                    metrics_df,
                    x='Model',
                    y='R2 Score',
                    title='R² Score',
                    color='Model',
                    text='R2 Score'
                )
                fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_r2.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_r2, use_container_width=True)
            
            # Best model
            best_model = metrics_df.loc[metrics_df['RMSE'].idxmin()]
            st.success(f"🏆 **Best Model**: {best_model['Model']} with RMSE: {best_model['RMSE']:.2f} MW")
            
        else:
            st.warning("No MLflow metrics found. Please run the training script first.")
            st.code("python train_model.py", language="bash")
    
    else:  # About page
        st.header("About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        This is an **MLOps project** for **Energy Demand Prediction** using time series forecasting.
        
        ### 🛠️ Technologies Used
        - **MLflow**: Experiment tracking and model management
        - **Scikit-learn**: Machine learning models (Random Forest, Gradient Boosting)
        - **Streamlit**: Interactive web dashboard
        - **Plotly**: Interactive visualizations
        - **Pandas & NumPy**: Data processing
        
        ### 📊 Models Implemented
        1. **Random Forest Regressor**: Ensemble method using decision trees
        2. **Gradient Boosting Regressor**: Sequential ensemble method
        
        ### 📈 Features
        - Time-based features (hour, day, month, etc.)
        - Lag features (1h, 24h, 168h)
        - Rolling statistics (mean, std)
        
        ### 🎓 Project Structure
        ```
        ├── train_model.py       # Training pipeline with MLflow
        ├── app.py              # Streamlit dashboard
        ├── requirements.txt    # Python dependencies
        ├── energy_data.csv     # Generated dataset
        ├── test_predictions.csv # Model predictions
        └── mlruns/            # MLflow experiment logs
        ```
        
        ### 🚀 How to Run
        1. Train models: `python train_model.py`
        2. Launch dashboard: `streamlit run app.py`
        
        ### 📝 Key Metrics
        - **MAE** (Mean Absolute Error): Average prediction error
        - **RMSE** (Root Mean Squared Error): Penalizes larger errors
        - **R² Score**: Proportion of variance explained
        
        ---
        **Developed for MLOps Course Project**
        """)
        
        st.info("💡 **Tip**: Run the training script first, then explore predictions and model performance!")

if __name__ == "__main__":
    main()