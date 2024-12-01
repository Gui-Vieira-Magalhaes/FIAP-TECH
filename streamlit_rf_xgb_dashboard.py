
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Função para calcular RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Título do aplicativo
st.title("Análise e Previsão de Preços do Petróleo com Múltiplos Modelos")

# Eventos históricos importantes
events = {
    '2014-11-25': 'OPEP não corta produção, preço cai',
    '2016-02-01': 'Acordo de Corte da OPEP',
    '2018-05-08': 'EUA saem do acordo nuclear com Irã',
    '2020-03-03': 'Pandemia de COVID-19',
    '2022-02-24': 'Invasão da Ucrânia pela Rússia',
    '2023-02-01': 'Tensões no Oriente Médio'
}

# Carregar os dados
@st.cache
def load_data():
    data = pd.read_csv('brent_oil.csv')
    data = data[data['Price'] != 'Ticker']
    data = data[['Price', 'Adj Close']]
    data.columns = ['Date', 'Price']
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
    data = data.dropna()
    return data

data = load_data()

# Visualizar os dados históricos
st.subheader("Dados Históricos")
st.write(data.head())

# Gráfico de preços com eventos históricos
st.subheader("Variação do Preço com Eventos Históricos")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'], data['Price'], label='Preço do Petróleo', color='blue', alpha=0.7)
for event_date, event_desc in events.items():
    event_date = pd.to_datetime(event_date)
    ax.axvline(x=event_date, color='red', linestyle='--', alpha=0.7)
    ax.text(event_date, data['Price'].max() * 0.8, event_desc, rotation=90, verticalalignment='center', fontsize=10, alpha=0.7)
ax.set_title('Preço do Petróleo com Eventos Históricos')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD)')
ax.legend()
st.pyplot(fig)

# Gráficos interativos para médias móveis
st.subheader("Análise Interativa com Média Móvel")
window_size = st.slider("Escolha o intervalo de média móvel (em dias):", 1, 365, 30)
data['Rolling Mean'] = data['Price'].rolling(window=window_size).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'], data['Price'], label='Preço Real', color='blue', alpha=0.5)
ax.plot(data['Date'], data['Rolling Mean'], label=f'Média Móvel ({window_size} dias)', color='orange')
ax.set_title('Preço do Petróleo com Média Móvel')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD)')
ax.legend()
st.pyplot(fig)

# Modelo de previsão com Prophet
st.subheader("Modelo de Previsão")
forecast_period = st.slider("Escolha o período de previsão (em dias):", 30, 365, 90)

# Preparar os dados para Prophet
prophet_data = data.rename(columns={"Date": "ds", "Price": "y"})
prophet_model = Prophet()
prophet_model.fit(prophet_data)

# Fazer a previsão com Prophet
future = prophet_model.make_future_dataframe(periods=forecast_period)
forecast = prophet_model.predict(future)

# Preparação de dados para Random Forest e XGBoost
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
train_data = data[:-forecast_period]
test_data = data[-forecast_period:]

X_train = train_data[['Day', 'Month', 'Year']]
y_train = train_data['Price']
X_test = test_data[['Day', 'Month', 'Year']]
y_test = test_data['Price']

# Modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Modelo XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# Métricas de Desempenho
st.subheader("Comparação de Modelos")
mae_rf = mean_absolute_error(y_test, rf_predictions)
rmse_rf = calculate_rmse(y_test, rf_predictions)
mae_xgb = mean_absolute_error(y_test, xgb_predictions)
rmse_xgb = calculate_rmse(y_test, xgb_predictions)

st.write(f"**Random Forest - MAE:** {mae_rf:.2f}, **RMSE:** {rmse_rf:.2f}")
st.write(f"**XGBoost - MAE:** {mae_xgb:.2f}, **RMSE:** {rmse_xgb:.2f}")

# Gráfico comparativo
st.subheader("Previsões Comparativas")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_data['Date'], y_test, label='Valores Reais', color='blue', alpha=0.6)
ax.plot(test_data['Date'], rf_predictions, label='Random Forest', color='green', alpha=0.6)
ax.plot(test_data['Date'], xgb_predictions, label='XGBoost', color='orange', alpha=0.6)
ax.set_title('Comparação de Modelos de Previsão')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD)')
ax.legend()
st.pyplot(fig)
