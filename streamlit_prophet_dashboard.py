
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

# Título do aplicativo
st.title("Análise e Previsão de Preços do Petróleo")

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

# Modelo de previsão com Prophet
st.subheader("Modelo de Previsão")
forecast_period = st.slider("Escolha o período de previsão (em dias):", 30, 365, 90)

# Preparar os dados para o Prophet
prophet_data = data.rename(columns={"Date": "ds", "Price": "y"})
model = Prophet()
model.fit(prophet_data)

# Fazer a previsão
future = model.make_future_dataframe(periods=forecast_period)
forecast = model.predict(future)

# Visualizar a previsão
st.subheader("Previsão de Preços do Petróleo")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Mostrar componentes de tendência e sazonalidade
st.subheader("Componentes da Previsão")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# Avaliar o desempenho do modelo (opcional)
st.subheader("Avaliação do Modelo")
forecast_train = forecast.iloc[:-forecast_period]
true_values = prophet_data['y'][-forecast_period:]
predicted_values = forecast_train['yhat'][-forecast_period:]
mae = (true_values - predicted_values).abs().mean()
rmse = ((true_values - predicted_values) ** 2).mean() ** 0.5

st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
