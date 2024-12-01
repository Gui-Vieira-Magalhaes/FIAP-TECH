
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Testar a importação correta do Prophet
try:
    from prophet import Prophet
except ImportError:
    st.error("O pacote 'prophet' não está instalado. Por favor, instale-o com 'pip install prophet'.")
    st.stop()

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
