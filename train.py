import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Add src to path
import sys
sys.path.append('.')
from src.data_generator import generate_er_data

def create_features(df):
    """Create features for modeling"""
    df = df.copy()
    
    # Time-based features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
    
    # Cyclical encoding for hour (captures circular nature of time)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['complaint', 'arrival_mode'], prefix=['complaint', 'arrival'])
    
    return df

def train_model():
    print("="*50)
    print("🏥 ER WAIT TIME PREDICTOR - MODEL TRAINING")
    print("="*50)
    
    # Generate data
    print("\n📊 Step 1: Generating synthetic data...")
    df = generate_er_data(5000)
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/er_visits.csv', index=False)
    print(f"✅ Generated {len(df)} records")
    print(f"📁 Saved to data/raw/er_visits.csv")
    
    # Feature engineering
    print("\n🔧 Step 2: Engineering features...")
    df = create_features(df)
    print(f"✅ Created {len(df.columns)} features")
    
    # Prepare for modeling
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'wait_time_minutes']]
    X = df[feature_cols]
    y = df['wait_time_minutes']
    
    print(f"\n📋 Dataset shape: {X.shape}")
    print(f"📋 Features: {len(feature_cols)}")
    
    # Split data
    print("\n📂 Step 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Training set: {len(X_train)} samples")
    print(f"✅ Test set: {len(X_test)} samples")
    
    # Scale features
    print("\n📏 Step 4: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled")
    
    # Train model
    print("\n🤖 Step 5: Training Random Forest model...")
    print("⏳ This may take a minute...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    print("✅ Model trained successfully!")
    
    # Evaluate
    print("\n📊 Step 6: Evaluating model...")
    
    # Training performance
    train_pred = model.predict(X_train_scaled)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    # Test performance
    test_pred = model.predict(X_test_scaled)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\n🎯 TRAINING PERFORMANCE:")
    print(f"   MAE: {train_mae:.1f} minutes")
    print(f"   R² Score: {train_r2:.3f}")
    
    print(f"\n🎯 TEST PERFORMANCE:")
    print(f"   MAE: {test_mae:.1f} minutes")
    print(f"   R² Score: {test_r2:.3f}")
    
    # Feature importance
    print("\n📊 Top 5 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Save model and artifacts
    print("\n💾 Step 7: Saving model artifacts...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    
    print("✅ Saved: models/model.pkl")
    print("✅ Saved: models/scaler.pkl")
    print("✅ Saved: models/feature_cols.pkl")
    
    print("\n" + "="*50)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("="*50)
    print("\n📌 Next step: Run 'streamlit run app/streamlit_app.py'")
    
    return model, scaler, feature_cols

if __name__ == "__main__":
    train_model()