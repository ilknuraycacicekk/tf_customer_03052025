# Müşteri Davranış Analizi ve Ürün Öneri Sistemi

Bu proje, Northwind veritabanı üzerinde üç farklı derin öğrenme modeli kullanarak müşteri davranışlarını analiz eden ve tahminler yapan API'ler içerir.

## API'ler

### 1. Yeniden Sipariş Tahmini API (Port: 8000)
Müşterilerin gelecekte tekrar sipariş verip vermeyeceğini tahmin eder.

**Endpoint:** http://127.0.0.1:8000
- Dokümantasyon: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

#### Endpoints:
- `POST /predict`: Müşteri yeniden sipariş tahmini
  ```json
  {
    "customer_id": "ALFKI"
  }
  ```

### 2. İade Riski Tahmini API (Port: 8001)
Siparişlerin iade riskini tahmin eder ve risk faktörlerini açıklar.

**Endpoint:** http://127.0.0.1:8001
- Dokümantasyon: http://127.0.0.1:8001/docs
- ReDoc: http://127.0.0.1:8001/redoc

#### Endpoints:
- `POST /predict_risk`: Sipariş iade riski tahmini
  ```json
  {
    "order_id": 10248
  }
  ```
- `POST /explain_risk`: Risk tahmininin açıklaması
  ```json
  {
    "order_id": 10248
  }
  ```

### 3. Ürün Kategori Öneri Sistemi API (Port: 8002)
Müşterilerin geçmiş alışveriş davranışlarına dayanarak, yeni ürün kategorilerinde alışveriş yapma olasılıklarını tahmin eder.

**Endpoint:** http://127.0.0.1:8002
- Dokümantasyon: http://127.0.0.1:8002/docs
- ReDoc: http://127.0.0.1:8002/redoc

#### Endpoints:
- `POST /predict`: Kategori bazlı öneriler
  ```json
  {
    "customer_id": "ALFKI"
  }
  ```

#### Örnek Yanıt:
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

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Veritabanı bağlantısını yapılandırın:
- PostgreSQL veritabanı kurulu olmalı
- Veritabanı adı: gyk1northwind
- Kullanıcı adı: postgres
- Şifre: abcde
- Host: localhost
- Port: 5432

3. Model dosyalarının varlığını kontrol edin:
- Yeniden Sipariş Tahmini için:
  - reorder_model.h5
  - reorder_scaler.pkl
  - reorder_columns.pkl
- İade Riski Tahmini için:
  - return_risk_model.h5
  - return_risk_scaler.pkl
  - return_risk_columns.pkl
  - return_risk_background.npy
- Ürün Kategori Öneri Sistemi için:
  - product_recommendation_model.h5
  - product_recommendation_scaler.pkl
  - product_recommendation_columns.pkl
  - category_mapping.pkl

## Çalıştırma

1. Yeniden Sipariş Tahmini API'sini başlatın:
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

2. İade Riski Tahmini API'sini başlatın:
```bash
uvicorn app2:app --reload --host 127.0.0.1 --port 8001
```

3. Ürün Kategori Öneri Sistemi API'sini başlatın:
```bash
uvicorn app3:app --reload --host 127.0.0.1 --port 8002
```

## API Kullanımı

### Yeniden Sipariş Tahmini
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "customer_id": "ALFKI"
}'
```

### İade Riski Tahmini
```bash
curl -X 'POST' \
  'http://127.0.0.1:8001/predict_risk' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "order_id": 10248
}'
```

### Ürün Kategori Önerileri
```bash
curl -X 'POST' \
  'http://127.0.0.1:8002/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "customer_id": "ALFKI"
}'
```

## Özellikler

- FastAPI ile modern ve hızlı API'ler
- Detaylı hata yakalama ve loglama
- CORS desteği
- Swagger ve ReDoc dokümantasyonu
- SHAP ile model açıklanabilirliği
- Veritabanı bağlantı kontrolü
- Model dosyası varlık kontrolü
- Derin öğrenme tabanlı tahmin modelleri
- Müşteri davranış analizi
- Kategori bazlı ürün önerileri

## Geliştirme

- Python 3.8+
- FastAPI
- TensorFlow
- SQLAlchemy
- SHAP
- Pandas
- NumPy

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 