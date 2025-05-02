# Northwind Müşteri Analizi ve Tahmin Sistemi

Bu proje, Northwind veritabanı üzerinde iki farklı derin öğrenme modeli kullanarak müşteri davranışlarını analiz eder ve tahminler yapar:

1. **Yeniden Sipariş Tahmini**: Müşterilerin gelecekte tekrar sipariş verip vermeyeceğini tahmin eder.
2. **İade Riski Tahmini**: Siparişlerin iade riskini tahmin eder.

## Özellikler

- Derin öğrenme modelleri (TensorFlow/Keras)
- SMOTE ile sınıf dengesizliği yönetimi
- SHAP ile model açıklanabilirliği
- FastAPI ile REST API
- PostgreSQL veritabanı entegrasyonu

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. PostgreSQL veritabanını kurun ve Northwind veritabanını içe aktarın.

3. Veritabanı bağlantı bilgilerini `model.py`, `model2.py`, `app.py` ve `app2.py` dosyalarında güncelleyin:
```python
user = "postgres"
password = "your_password"
host = "localhost"
port = "5432"
database = "gyk1northwind"
```

## Kullanım

### Model Eğitimi

1. Yeniden sipariş tahmini modelini eğitin:
```bash
python model.py
```

2. İade riski tahmini modelini eğitin:
```bash
python model2.py
```

### API Kullanımı

1. API'yi başlatın:
```bash
uvicorn app:app --reload
```

2. İkinci API'yi başlatın (farklı bir port kullanarak):
```bash
uvicorn app2:app --reload --port 8001
```

### API Endpoint'leri

#### Yeniden Sipariş Tahmini API (Port 8000)

- **POST /predict**
  ```json
  {
    "customer_id": "ALFKI"
  }
  ```

#### İade Riski Tahmini API (Port 8001)

- **POST /predict_risk**
  ```json
  {
    "order_id": 10248
  }
  ```

- **POST /explain_risk**
  ```json
  {
    "order_id": 10248
  }
  ```

## Model Detayları

### Yeniden Sipariş Tahmini Modeli
- Girdi özellikleri: Sipariş sayısı, toplam tutar, ortalama sipariş büyüklüğü, çeyreklik
- Çıktı: Müşterinin tekrar sipariş verme olasılığı

### İade Riski Tahmini Modeli
- Girdi özellikleri: Ortalama indirim, toplam miktar, sipariş toplamı
- Çıktı: Siparişin iade riski olasılığı
- SHAP değerleri ile özellik önemlilikleri

## Dosya Yapısı

```
.
├── README.md
├── requirements.txt
├── model.py              # Yeniden sipariş tahmini modeli
├── model2.py            # İade riski tahmini modeli
├── app.py               # Yeniden sipariş API'si
├── app2.py              # İade riski API'si
├── reorder_model.h5     # Eğitilmiş yeniden sipariş modeli
├── reorder_scaler.pkl   # Yeniden sipariş scaler'ı
├── reorder_columns.pkl  # Yeniden sipariş özellik isimleri
├── return_risk_model.h5 # Eğitilmiş iade riski modeli
├── return_risk_scaler.pkl # İade riski scaler'ı
├── return_risk_columns.pkl # İade riski özellik isimleri
├── return_risk_shap_values.npy # SHAP değerleri
└── return_risk_background.npy  # SHAP background verisi
```

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 