from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sqlalchemy import create_engine, text
import logging
import traceback
import os

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model dosyalarının varlığını kontrol et
required_files = [
    "reorder_model.h5",
    "reorder_scaler.pkl",
    "reorder_columns.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        logger.error(f"Gerekli dosya bulunamadı: {file}")
        raise FileNotFoundError(f"Gerekli dosya bulunamadı: {file}")

app = FastAPI(
    title="Yeniden Sipariş Tahmini API",
    description="Müşterilerin tekrar sipariş verme olasılığını tahmin eden API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    servers=[
        {"url": "http://127.0.0.1:8000", "description": "Local development server"},
        {"url": "http://localhost:8000", "description": "Local development server"}
    ]
)

# CORS ayarları
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

# Trusted Host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Veritabanı bağlantısı
try:
    user = "postgres"
    password = "abcde"
    host = "localhost"
    port = "5432"
    database = "gyk1northwind"
    connection_string = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
    logger.info(f"Veritabanına bağlanılıyor: {host}:{port}/{database}")
    engine = create_engine(connection_string)
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Veritabanı bağlantısı başarılı")
except Exception as e:
    logger.error(f"Veritabanı bağlantı hatası: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Model ve scaler yükleme
try:
    logger.info("Model yükleniyor...")
    model = tf.keras.models.load_model("reorder_model.h5")
    logger.info("Model başarıyla yüklendi")
    
    logger.info("Scaler yükleniyor...")
    scaler = joblib.load("reorder_scaler.pkl")
    logger.info("Scaler başarıyla yüklendi")
    
    logger.info("Sütun isimleri yükleniyor...")
    columns = joblib.load("reorder_columns.pkl")
    logger.info("Sütun isimleri başarıyla yüklendi")
    
    logger.info("Tüm modeller başarıyla yüklendi")
except Exception as e:
    logger.error(f"Model yükleme hatası: {str(e)}")
    logger.error(traceback.format_exc())
    raise

class CustomerID(BaseModel):
    customer_id: str

@app.get("/")
async def root():
    return {
        "message": "Yeniden Sipariş Tahmini API'sine Hoş Geldiniz",
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict - Müşteri yeniden sipariş tahmini"
        }
    }

@app.post("/predict")
def predict(data: CustomerID):
    """
    Müşterinin gelecekte tekrar sipariş verip vermeyeceğini tahmin eder.
    
    - **customer_id**: Müşteri ID'si (örn: "ALFKI")
    
    Returns:
    - **prediction_probability**: Tahmin olasılığı (0-1 arası)
    - **will_reorder**: Tekrar sipariş verecek mi? (0 veya 1)
    """
    try:
        logger.info(f"Tahmin isteği alındı - Customer ID: {data.customer_id}")
        
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
        logger.info("SQL sorgusu çalıştırılıyor...")
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logger.warning(f"Müşteri bulunamadı - Customer ID: {data.customer_id}")
            raise HTTPException(status_code=404, detail="Customer not found")
        
        logger.info("Veri hazırlanıyor...")
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
        
        logger.info("Tahmin yapılıyor...")
        input_row = np.array([[row_dict.get(col, 0) for col in columns]])
        input_row_scaled = scaler.transform(input_row)
        prediction = model.predict(input_row_scaled)[0][0]
        
        result = {
            "customer_id": data.customer_id,
            "prediction_probability": float(prediction),
            "will_reorder": int(prediction > 0.5)
        }
        logger.info(f"Tahmin başarılı - Customer ID: {data.customer_id}")
        return result
        
    except Exception as e:
        logger.error(f"Tahmin hatası - Customer ID: {data.customer_id}")
        logger.error(f"Hata detayı: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin hatası: {str(e)}"
        ) 