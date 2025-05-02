from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import shap
from sqlalchemy import create_engine

app = FastAPI(
    title="İade Riski Tahmini API",
    description="Siparişlerin iade riskini tahmin eden ve açıklayan API",
    version="1.0.0"
)

# Veritabanı bağlantısı
user = "postgres"
password = "abcde"
host = "localhost"
port = "5432"
database = "gyk1northwind"
engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')

# Model ve scaler yükleme
model = tf.keras.models.load_model("return_risk_model.h5")
scaler = joblib.load("return_risk_scaler.pkl")
columns = joblib.load("return_risk_columns.pkl")

# SHAP değerlerini yükle
background = np.load("return_risk_background.npy")
explainer = shap.DeepExplainer(model, background)

class OrderID(BaseModel):
    order_id: int

@app.post("/predict_risk")
def predict_risk(data: OrderID):
    # Sipariş verilerini çek
    query = f"""
    WITH order_metrics AS (
        SELECT 
            od.order_id,
            od.product_id,
            od.quantity,
            od.unit_price,
            od.discount,
            (od.unit_price * od.quantity) as total_amount,
            (od.unit_price * od.quantity * (1 - od.discount)) as final_amount
        FROM order_details od
    )
    SELECT 
        om.order_id,
        AVG(om.discount) as avg_discount,
        SUM(om.quantity) as total_quantity,
        SUM(om.final_amount) as order_total
    FROM order_metrics om
    WHERE om.order_id = {data.order_id}
    GROUP BY om.order_id;
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Veriyi hazırla
    input_data = df[['avg_discount', 'total_quantity', 'order_total']]
    input_scaled = scaler.transform(input_data)
    
    # Tahmin yap
    prediction = model.predict(input_scaled)[0][0]
    
    return {
        "order_id": data.order_id,
        "risk_probability": float(prediction),
        "is_risky": int(prediction > 0.5),
        "features": {
            "avg_discount": float(df['avg_discount'].iloc[0]),
            "total_quantity": int(df['total_quantity'].iloc[0]),
            "order_total": float(df['order_total'].iloc[0])
        }
    }

@app.post("/explain_risk")
def explain_risk(data: OrderID):
    # Sipariş verilerini çek
    query = f"""
    WITH order_metrics AS (
        SELECT 
            od.order_id,
            od.product_id,
            od.quantity,
            od.unit_price,
            od.discount,
            (od.unit_price * od.quantity) as total_amount,
            (od.unit_price * od.quantity * (1 - od.discount)) as final_amount
        FROM order_details od
    )
    SELECT 
        om.order_id,
        AVG(om.discount) as avg_discount,
        SUM(om.quantity) as total_quantity,
        SUM(om.final_amount) as order_total
    FROM order_metrics om
    WHERE om.order_id = {data.order_id}
    GROUP BY om.order_id;
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Veriyi hazırla
    input_data = df[['avg_discount', 'total_quantity', 'order_total']]
    input_scaled = scaler.transform(input_data)
    
    # SHAP değerlerini hesapla
    shap_values = explainer.shap_values(input_scaled)
    
    # Feature importance'ı hesapla
    feature_importance = dict(zip(columns, np.abs(shap_values[0])))
    
    return {
        "order_id": data.order_id,
        "feature_importance": feature_importance,
        "feature_values": {
            "avg_discount": float(df['avg_discount'].iloc[0]),
            "total_quantity": int(df['total_quantity'].iloc[0]),
            "order_total": float(df['order_total'].iloc[0])
        }
    }
