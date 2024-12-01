
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# Função para calcular RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Função para preparar os dados para LSTM
def prepare_lstm_data(data, n_lags):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(n_lags, len(data_scaled)):
        X.append(data_scaled[i-n_lags:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(X), np.array(y), scaler

# Função para prever datas futuras com base no modelo LSTM
def predict_future_lstm(model, data, n_lags, days, scaler):
    future_values = data[-n_lags:].reshape(1, n_lags, 1)
    predictions = []
    for _ in range(days):
        pred = model.predict(future_values)
        predictions.append(pred[0, 0])
        future_values = np.append(future_values[:, 1:, :], [[pred[0, 0]]], axis=1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Título do aplicativo
st.title("Análise e Previsão de Preços do Petróleo")

# Integrantes do grupo
st.markdown("**Integrantes do Grupo:** Fábio Cervantes Lima, Guilherme Vieira Magalhães")

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

# Modelo LSTM
st.subheader("Previsão com LSTM")
forecast_period = st.slider("Escolha o período de previsão (em dias):", 30, 1000, 90)
n_lags = st.slider("Escolha o número de lags para o LSTM:", 30, 365, 60)

# Preparar dados para LSTM
price_data = data['Price'].values
X_lstm, y_lstm, scaler = prepare_lstm_data(price_data, n_lags)

# Dividir em treino e teste
train_size = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

# Construir modelo LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(n_lags, 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50, activation='relu'))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Treinar modelo LSTM
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=0)

# Fazer previsões com LSTM
y_pred_lstm_scaled = model_lstm.predict(X_test_lstm)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1))

# Avaliar desempenho LSTM
mae_lstm = mean_absolute_error(y_test_lstm, scaler.inverse_transform(y_test_lstm.reshape(-1, 1)))
rmse_lstm = calculate_rmse(y_test_lstm, scaler.inverse_transform(y_test_lstm.reshape(-1, 1)))

st.write(f"**LSTM - MAE:** {mae_lstm:.2f}, **RMSE:** {rmse_lstm:.2f}")

# Gráfico comparativo LSTM com data real
st.subheader("Previsão LSTM Comparativa ao Longo do Tempo")
test_dates = data['Date'].iloc[-len(y_test_lstm):].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_dates, scaler.inverse_transform(y_test_lstm.reshape(-1, 1)), label='Valores Reais', color='blue', alpha=0.6)
ax.plot(test_dates, y_pred_lstm, label='Previsões LSTM', color='orange', alpha=0.6)
ax.set_title('Previsão com LSTM')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD)')
ax.legend()
st.pyplot(fig)

# Interação para previsão de datas futuras
st.subheader("Previsão para Datas Futuras")
future_date = st.date_input("Escolha uma data futura:", datetime.now() + timedelta(days=30))
days_into_future = (future_date - data['Date'].iloc[-1].date()).days

if days_into_future > 0:
    future_prediction = predict_future_lstm(model_lstm, price_data, n_lags, days_into_future, scaler)
    st.write(f"**O valor previsto, com base no modelo, do petróleo é {future_prediction[-1, 0]:.2f} para o dia {future_date}.**")
else:
    st.warning("Por favor, escolha uma data futura válida.")
