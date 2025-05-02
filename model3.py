import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle
import os

# Database connection
DB_CONNECTION = "postgresql://postgres:abcde@localhost:5432/gyk1northwind"

def load_and_preprocess_data():
    """Load and preprocess data from the database"""
    engine = create_engine(DB_CONNECTION)
    
    # Load data from database
    products_df = pd.read_sql("SELECT product_id, category_id, unit_price FROM products", engine)
    categories_df = pd.read_sql("SELECT category_id, category_name FROM categories", engine)
    order_details_df = pd.read_sql("SELECT order_id, product_id, unit_price, quantity, discount FROM order_details", engine)
    orders_df = pd.read_sql("SELECT order_id, customer_id FROM orders", engine)
    
    # Merge data
    merged_df = order_details_df.merge(products_df, on='product_id', suffixes=('_order', '_product'))
    merged_df = merged_df.merge(categories_df, on='category_id')
    merged_df = merged_df.merge(orders_df, on='order_id')
    
    # Calculate total spent per category per customer
    customer_category_spend = merged_df.groupby(['customer_id', 'category_id'])['unit_price_order'].sum().reset_index()
    
    # Create pivot table for customer-category spending
    customer_category_matrix = customer_category_spend.pivot(
        index='customer_id',
        columns='category_id',
        values='unit_price_order'
    ).fillna(0)
    
    # Save category mapping
    category_mapping = categories_df.set_index('category_id')['category_name'].to_dict()
    with open('category_mapping.pkl', 'wb') as f:
        pickle.dump(category_mapping, f)
    
    return customer_category_matrix, category_mapping

def create_model(input_dim, output_dim):
    """Create and compile the neural network model"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_save_model():
    """Train the model and save it along with necessary files"""
    # Load and preprocess data
    customer_category_matrix, category_mapping = load_and_preprocess_data()
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_category_matrix)
    
    # Save scaler and columns
    with open('product_recommendation_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('product_recommendation_columns.pkl', 'wb') as f:
        pickle.dump(customer_category_matrix.columns.tolist(), f)
    
    # Create and train model
    model = create_model(
        input_dim=X_scaled.shape[1],
        output_dim=X_scaled.shape[1]
    )
    
    # Train the model (using the same data as both input and target for autoencoder-like behavior)
    model.fit(
        X_scaled, X_scaled,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the model
    model.save('product_recommendation_model.h5')

if __name__ == "__main__":
    train_and_save_model()
