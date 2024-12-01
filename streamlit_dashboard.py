
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título do aplicativo
st.title("Dashboard Interativo - Análise de Preços do Petróleo")

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

# Exibir os dados
st.subheader("Dados Históricos")
st.write(data.head())

# Visualização do preço ao longo do tempo
st.subheader("Variação do Preço ao Longo do Tempo")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Date'], data['Price'], label='Preço do Petróleo', color='blue')
ax.set_title('Preço do Petróleo ao Longo do Tempo')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD)')
ax.legend()
st.pyplot(fig)
