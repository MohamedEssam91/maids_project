import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_data(file_path):
    """Load data from Excel file."""
    return pd.read_excel(file_path)

def preprocess_data(df):
    """Preprocess data by handling missing values and scaling features."""
    
    X = df[['battery_power', 'px_height', 'px_width', 'ram']]
    
    for column in X.columns:
        X[column].fillna(X[column].mean(), inplace=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if 'price_range' in df.columns:        
        y = df['price_range']
        return X_scaled, y
      
    return X_scaled, None

def train_model(X_train, y_train):
    """Train a K-Nearest Neighbor Classifier model."""
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    return model
