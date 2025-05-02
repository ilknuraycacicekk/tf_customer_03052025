from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import tensorflow as tf
import pickle
from typing import List, Dict

app = FastAPI(
    title="Ürün Kategori Öneri Sistemi API",
    description="""
    Bu API, müşterilerin geçmiş satın alma davranışlarına dayanarak, 
    yeni ürün kategorilerinde alışveriş yapma olasılıklarını tahmin eder.
    
    ## Özellikler
    * Müşteri bazlı kategori önerileri
    * Geçmiş alışveriş verilerine dayalı tahminler
    * Derin öğrenme tabanlı öneri sistemi
    * Kategori bazlı olasılık skorları
    
    ## Nasıl Çalışır?
    1. Müşterinin geçmiş alışveriş verileri analiz edilir
    2. Her kategori için harcama miktarları hesaplanır
    3. Derin öğrenme modeli ile yeni kategori tercihleri tahmin edilir
    4. Sonuçlar olasılık skorlarına göre sıralanır
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Database connection
DB_CONNECTION = "postgresql://postgres:abcde@localhost:5432/gyk1northwind"

# Load necessary files
try:
    model = tf.keras.models.load_model('product_recommendation_model.h5')
    with open('product_recommendation_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('product_recommendation_columns.pkl', 'rb') as f:
        category_columns = pickle.load(f)
    with open('category_mapping.pkl', 'rb') as f:
        category_mapping = pickle.load(f)
except Exception as e:
    raise Exception(f"Error loading model files: {str(e)}")

class CustomerInput(BaseModel):
    customer_id: str = Field(
        ...,
        description="Müşteri ID'si (örn: 'ALFKI')",
        example="ALFKI"
    )

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "ALFKI"
            }
        }

class CategoryRecommendation(BaseModel):
    category_id: int = Field(..., description="Kategori ID'si")
    category_name: str = Field(..., description="Kategori adı")
    probability: float = Field(..., description="Bu kategoride alışveriş yapma olasılığı (0-1 arası)")

    class Config:
        schema_extra = {
            "example": {
                "category_id": 1,
                "category_name": "Beverages",
                "probability": 0.85
            }
        }

@app.get("/", tags=["Genel"])
def read_root():
    """
    API'nin ana sayfası. API hakkında genel bilgi döndürür.
    """
    return {
        "message": "Ürün Kategori Öneri Sistemi API'sine Hoş Geldiniz",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.post("/predict", 
    response_model=List[CategoryRecommendation],
    tags=["Tahmin"],
    summary="Kategori Önerileri",
    description="""
    Verilen müşteri ID'si için kategori bazlı alışveriş olasılıklarını tahmin eder.
    
    ## Girdi
    * customer_id: Müşteri ID'si
    
    ## Çıktı
    * Kategori bazlı öneriler listesi
        * category_id: Kategori ID'si
        * category_name: Kategori adı
        * probability: Alışveriş yapma olasılığı
    
    ## Örnek Yanıt
    ```json
    [
        {
            "category_id": 1,
            "category_name": "Beverages",
            "probability": 0.85
        },
        {
            "category_id": 2,
            "category_name": "Condiments",
            "probability": 0.72
        }
    ]
    ```
    """
)
async def predict_categories(customer_input: CustomerInput):
    try:
        # Get customer's historical data
        engine = create_engine(DB_CONNECTION)
        query = f"""
        WITH customer_categories AS (
            SELECT 
                o.customer_id,
                p.category_id,
                SUM(od.unit_price * od.quantity * (1 - od.discount)) as total_spent
            FROM orders o
            JOIN order_details od ON o.order_id = od.order_id
            JOIN products p ON od.product_id = p.product_id
            WHERE o.customer_id = '{customer_input.customer_id}'
            GROUP BY o.customer_id, p.category_id
        )
        SELECT 
            category_id,
            COALESCE(total_spent, 0) as total_spent
        FROM customer_categories
        """
        
        customer_data = pd.read_sql(query, engine)
        
        # Create feature vector
        feature_vector = np.zeros(len(category_columns))
        for _, row in customer_data.iterrows():
            if row['category_id'] in category_columns:
                idx = category_columns.index(row['category_id'])
                feature_vector[idx] = row['total_spent']
        
        # Scale the input
        feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))
        
        # Get predictions
        predictions = model.predict(feature_vector_scaled)[0]
        
        # Create recommendations
        recommendations = []
        for idx, prob in enumerate(predictions):
            category_id = category_columns[idx]
            recommendations.append(
                CategoryRecommendation(
                    category_id=category_id,
                    category_name=category_mapping[category_id],
                    probability=float(prob)
                )
            )
        
        # Sort by probability
        recommendations.sort(key=lambda x: x.probability, reverse=True)
        
        return recommendations
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin sırasında bir hata oluştu: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
