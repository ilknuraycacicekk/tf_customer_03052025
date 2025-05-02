import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
import shap

# Veritabanı bağlantısı
user = "postgres"
password = "abcde"
host = "localhost"
port = "5432"
database = "gyk1northwind"
engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')

# SQL ile veri çekme
sql_query = """
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
    om.product_id,
    om.quantity,
    om.unit_price,
    om.discount,
    om.total_amount,
    om.final_amount,
    -- Risk skoru için özellikler
    AVG(om.discount) OVER (PARTITION BY om.order_id) as avg_discount,
    SUM(om.quantity) OVER (PARTITION BY om.order_id) as total_quantity,
    SUM(om.final_amount) OVER (PARTITION BY om.order_id) as order_total
FROM order_metrics om
ORDER BY om.order_id, om.product_id;
"""
df = pd.read_sql(sql_query, engine)

# Risk etiketleme (yüksek indirim + düşük harcama = riskli)
df['is_risky'] = ((df['avg_discount'] > df['avg_discount'].quantile(0.75)) & 
                  (df['order_total'] < df['order_total'].quantile(0.25))).astype(int)

# Feature engineering
X = df[['avg_discount', 'total_quantity', 'order_total']]
y = df['is_risky']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cost-sensitive learning için class weights
class_weights = {
    0: 1.0,
    1: 2.0  # Riskli siparişlere daha fazla ağırlık
}

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Model eğitimi
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)

# Model ve scaler kaydet
model.save("return_risk_model.h5")
joblib.dump(scaler, "return_risk_scaler.pkl")
joblib.dump(X.columns.tolist(), "return_risk_columns.pkl")

# SHAP değerlerini hesapla
background = X_train_scaled[:100]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_test_scaled[:100])

# SHAP değerlerini numpy array olarak kaydet
np.save("return_risk_shap_values.npy", shap_values)
np.save("return_risk_background.npy", background)

# Test sonuçları
from sklearn.metrics import classification_report, confusion_matrix
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
print("\nTest Sonuçları:")
print(classification_report(y_test, y_pred))

# SHAP değerlerini görselleştir
shap.summary_plot(shap_values, X_test[:100], feature_names=X.columns)
