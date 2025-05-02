import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import joblib

# Veritabanı bağlantısı
user = "postgres"
password = "abcde"
host = "localhost"
port = "5432"
database = "gyk1northwind"
engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')

# SQL ile veri çekme
sql_query = """
WITH aylik_data AS (
    SELECT *,
        TO_DATE(ay_yil || '-01', 'YYYY-MM-DD') AS ay_tarih
    FROM aylik_siparis_ozeti
),
son_siparis AS (
    SELECT customer_id, MAX(ay_tarih) AS son_siparis_tarihi
    FROM aylik_data
    GROUP BY customer_id
),
tekrar_siparis AS (
    SELECT
        a.customer_id,
        CASE
            WHEN COUNT(*) FILTER (
                WHERE ay_tarih > s.son_siparis_tarihi
                  AND ay_tarih <= s.son_siparis_tarihi + INTERVAL '6 month'
            ) > 0 THEN 1
            ELSE 0
        END AS tekrar_siparis_var_mi
    FROM aylik_data a
    JOIN son_siparis s ON a.customer_id = s.customer_id
    GROUP BY a.customer_id, s.son_siparis_tarihi
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
    t.tekrar_siparis_var_mi
FROM aylik_data a
JOIN tekrar_siparis t ON a.customer_id = t.customer_id
ORDER BY a.customer_id, a.ay_tarih DESC;
"""
df = pd.read_sql(sql_query, engine)

# Feature engineering
X = df[['siparis_sayisi', 'toplam_tutar', 'ortalama_siparis_buyuklugu', 'ceyreklik']]
X = pd.get_dummies(X, columns=['ceyreklik'], drop_first=True)
y = df['tekrar_siparis_var_mi']

# Sınıf dağılımını kontrol et ve yazdır
print("Sınıf dağılımı:")
print(y.value_counts())

# Data augmentation (SMOTE) - sadece birden fazla sınıf varsa uygula
if len(y.unique()) > 1:
    print("\nSMOTE uygulanıyor...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print("SMOTE sonrası sınıf dağılımı:")
    print(pd.Series(y_res).value_counts())
else:
    print("\nSMOTE uygulanamıyor: Veri setinde tek sınıf var!")
    X_res, y_res = X, y

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, stratify=y_res, random_state=42)

# Normalizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Model ve scaler kaydet
model.save("reorder_model.h5")
joblib.dump(scaler, "reorder_scaler.pkl")
joblib.dump(X.columns.tolist(), "reorder_columns.pkl")

# Test sonucu
from sklearn.metrics import classification_report
print(classification_report(y_test, (model.predict(X_test_scaled) > 0.5).astype(int))) 