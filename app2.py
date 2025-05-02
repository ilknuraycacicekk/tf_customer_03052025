from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import shap
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
    "return_risk_model.h5",
    "return_risk_scaler.pkl",
    "return_risk_columns.pkl",
    "return_risk_background.npy"
]

for file in required_files:
    if not os.path.exists(file):
        logger.error(f"Gerekli dosya bulunamadı: {file}")
        raise FileNotFoundError(f"Gerekli dosya bulunamadı: {file}")

app = FastAPI(
    title="İade Riski Tahmini API",
    description="Siparişlerin iade riskini tahmin eden ve açıklayan API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
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
    model = tf.keras.models.load_model("return_risk_model.h5")
    logger.info("Model başarıyla yüklendi")
    
    logger.info("Scaler yükleniyor...")
    scaler = joblib.load("return_risk_scaler.pkl")
    logger.info("Scaler başarıyla yüklendi")
    
    logger.info("Sütun isimleri yükleniyor...")
    columns = joblib.load("return_risk_columns.pkl")
    logger.info("Sütun isimleri başarıyla yüklendi")
    
    logger.info("SHAP background verisi yükleniyor...")
    background = np.load("return_risk_background.npy")
    logger.info("SHAP background verisi başarıyla yüklendi")
    
    logger.info("SHAP explainer oluşturuluyor...")
    explainer = shap.DeepExplainer(model, background)
    logger.info("SHAP explainer başarıyla oluşturuldu")
    
    logger.info("Tüm modeller başarıyla yüklendi")
except Exception as e:
    logger.error(f"Model yükleme hatası: {str(e)}")
    logger.error(traceback.format_exc())
    raise

class OrderID(BaseModel):
    order_id: int

@app.get("/")
async def root():
    return {
        "message": "İade Riski Tahmini API'sine Hoş Geldiniz",
        "docs": "/docs",
        "endpoints": {
            "predict_risk": "/predict_risk - Sipariş iade riski tahmini",
            "explain_risk": "/explain_risk - Risk tahmininin açıklaması"
        }
    }

@app.post("/predict_risk")
def predict_risk(data: OrderID):
    """
    Siparişin iade riskini tahmin eder.
    
    - **order_id**: Sipariş ID'si (örn: 10248)
    
    Returns:
    - **risk_probability**: Risk olasılığı (0-1 arası)
    - **is_risky**: Riskli mi? (0 veya 1)
    - **features**: Sipariş özellikleri
    """
    try:
        logger.info(f"Tahmin isteği alındı - Order ID: {data.order_id}")
        
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
        logger.info("SQL sorgusu çalıştırılıyor...")
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logger.warning(f"Sipariş bulunamadı - Order ID: {data.order_id}")
            raise HTTPException(status_code=404, detail="Order not found")
        
        logger.info("Veri hazırlanıyor...")
        input_data = df[['avg_discount', 'total_quantity', 'order_total']]
        input_scaled = scaler.transform(input_data)
        
        logger.info("Tahmin yapılıyor...")
        prediction = model.predict(input_scaled)[0][0]
        
        result = {
            "order_id": data.order_id,
            "risk_probability": float(prediction),
            "is_risky": int(prediction > 0.5),
            "features": {
                "avg_discount": float(df['avg_discount'].iloc[0]),
                "total_quantity": int(df['total_quantity'].iloc[0]),
                "order_total": float(df['order_total'].iloc[0])
            }
        }
        logger.info(f"Tahmin başarılı - Order ID: {data.order_id}")
        return result
        
    except Exception as e:
        logger.error(f"Tahmin hatası - Order ID: {data.order_id}")
        logger.error(f"Hata detayı: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain_risk")
def explain_risk(data: OrderID):
    """
    Siparişin iade riski tahmininin açıklamasını verir.
    
    - **order_id**: Sipariş ID'si (örn: 10248)
    
    Returns:
    - **feature_importance**: Özelliklerin önem dereceleri
    - **feature_values**: Özellik değerleri
    """
    try:
        logger.info(f"SHAP açıklama isteği alındı - Order ID: {data.order_id}")
        
        # Veritabanı sorgusu
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
        logger.info("SQL sorgusu çalıştırılıyor...")
        try:
            df = pd.read_sql(query, engine)
            logger.info(f"SQL sorgusu başarılı. Veri şekli: {df.shape}")
        except Exception as e:
            logger.error(f"SQL sorgusu hatası: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Veritabanı sorgusu hatası: {str(e)}")
        
        if df.empty:
            logger.warning(f"Sipariş bulunamadı - Order ID: {data.order_id}")
            raise HTTPException(status_code=404, detail="Order not found")
        
        # Veri hazırlama
        logger.info("Veri hazırlanıyor...")
        logger.info(f"Gelen veri şekli: {df.shape}")
        logger.info(f"Gelen veri sütunları: {df.columns.tolist()}")
        
        try:
            input_data = df[['avg_discount', 'total_quantity', 'order_total']]
            logger.info(f"Seçilen özellikler: {input_data.columns.tolist()}")
            logger.info(f"Veri örneği:\n{input_data.head()}")
        except Exception as e:
            logger.error(f"Veri hazırlama hatası: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Veri hazırlama hatası: {str(e)}")
        
        # Veri normalizasyonu
        logger.info("Veri normalizasyonu yapılıyor...")
        try:
            input_scaled = scaler.transform(input_data)
            logger.info(f"Normalize edilmiş veri şekli: {input_scaled.shape}")
        except Exception as e:
            logger.error(f"Normalizasyon hatası: {str(e)}")
            logger.error(f"Scaler bilgisi: {scaler}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Normalizasyon hatası: {str(e)}")
        
        # SHAP değerleri hesaplama
        logger.info("SHAP değerleri hesaplanıyor...")
        try:
            logger.info("Model tahmin yapıyor...")
            model_prediction = model.predict(input_scaled)
            logger.info(f"Model tahmin şekli: {model_prediction.shape}")
            
            logger.info("SHAP değerleri hesaplanıyor...")
            shap_values = explainer.shap_values(input_scaled)
            logger.info(f"SHAP değerleri şekli: {np.array(shap_values).shape}")
            
            # SHAP değerlerini JSON'a uygun formata dönüştür
            shap_values_array = np.array(shap_values[0])
            feature_importance = {
                col: float(val) for col, val in zip(columns, np.abs(shap_values_array))
            }
            logger.info(f"Hesaplanan özellik önemleri: {feature_importance}")
        except Exception as e:
            logger.error(f"SHAP hesaplama hatası: {str(e)}")
            logger.error(f"Model bilgisi: {model.summary()}")
            logger.error(f"Explainer bilgisi: {explainer}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"SHAP hesaplama hatası: {str(e)}")
        
        # Sonuçları JSON'a uygun formata dönüştür
        result = {
            "order_id": data.order_id,
            "feature_importance": feature_importance,
            "feature_values": {
                "avg_discount": float(df['avg_discount'].iloc[0]),
                "total_quantity": int(df['total_quantity'].iloc[0]),
                "order_total": float(df['order_total'].iloc[0])
            }
        }
        logger.info(f"SHAP açıklaması başarılı - Order ID: {data.order_id}")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Beklenmeyen hata - Order ID: {data.order_id}")
        logger.error(f"Hata detayı: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Beklenmeyen hata: {str(e)}"
        )
