# Müşteri Davranış Analizi ve İade Riski Tahmini API'leri

Bu proje, Northwind veritabanı üzerinde iki farklı derin öğrenme modeli kullanarak müşteri davranışlarını analiz eden ve iade riskini tahmin eden iki ayrı API içerir.

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

## Çalıştırma

1. Yeniden Sipariş Tahmini API'sini başlatın:
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

2. İade Riski Tahmini API'sini başlatın:
```bash
uvicorn app2:app --reload --host 127.0.0.1 --port 8001
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

### Risk Açıklaması
```bash
curl -X 'POST' \
  'http://127.0.0.1:8001/explain_risk' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "order_id": 10248
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