from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sqlalchemy import create_engine

app = FastAPI(
    title="Yeniden Sipariş Tahmini API",
    description="Müşterilerin tekrar sipariş verme olasılığını tahmin eden API",
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
model = tf.keras.models.load_model("reorder_model.h5")
scaler = joblib.load("reorder_scaler.pkl")
columns = joblib.load("reorder_columns.pkl")

class CustomerID(BaseModel):
    customer_id: str

@app.post("/predict")
def predict(data: CustomerID):
    query = f"""
    WITH aylik_data AS (
        SELECT *,
            TO_DATE(ay_yil || '-01', 'YYYY-MM-DD') AS ay_tarih
        FROM aylik_siparis_ozeti
    ),
    son_siparis AS (
        SELECT customer_id, MAX(ay_tarih) AS son_siparis_tarihi
        FROM aylik_data
        GROUP BY customer_id
    )
    SELECT 
        a.customer_id,
        a.siparis_sayisi,
        a.toplam_tutar,
        a.ortalama_siparis_buyuklugu,
        CASE 
            WHEN a.ay IN (12, 1, 2) THEN 'Q1'
            WHEN a.ay IN (3, 4, 5) THEN 'Q2'
            WHEN a.ay IN (6, 7, 8) THEN 'Q3'
            WHEN a.ay IN (9, 10, 11) THEN 'Q4'
            ELSE NULL
        END AS ceyreklik,
        DATE_PART('month', AGE(CURRENT_DATE, MAX(a.ay_tarih) OVER (PARTITION BY a.customer_id))) +
        12 * DATE_PART('year', AGE(CURRENT_DATE, MAX(a.ay_tarih) OVER (PARTITION BY a.customer_id))) AS ay_farki
    FROM aylik_data a
    WHERE a.customer_id = '{data.customer_id}'
    ORDER BY a.ay_tarih DESC
    LIMIT 1;
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    row = df.iloc[0]
    ceyreklik = row['ceyreklik']
    row_dict = {
        'siparis_sayisi': row['siparis_sayisi'],
        'toplam_tutar': row['toplam_tutar'],
        'ortalama_siparis_buyuklugu': row['ortalama_siparis_buyuklugu'],
        'ay_farki': row['ay_farki'],
        'ceyreklik_Q2': 0,
        'ceyreklik_Q3': 0,
        'ceyreklik_Q4': 0
    }
    if ceyreklik in ['Q2', 'Q3', 'Q4']:
        row_dict[f'ceyreklik_{ceyreklik}'] = 1
    input_row = np.array([[row_dict.get(col, 0) for col in columns]])
    input_row_scaled = scaler.transform(input_row)
    prediction = model.predict(input_row_scaled)[0][0]
    return {
        "customer_id": data.customer_id,
        "prediction_probability": float(prediction),
        "will_reorder": int(prediction > 0.5)
    } 