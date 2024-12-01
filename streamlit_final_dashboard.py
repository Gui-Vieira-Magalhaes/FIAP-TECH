
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título do aplicativo
st.title("Análise e Previsão de Preços do Petróleo")

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

# Insights com storytelling
st.subheader("Insights sobre Variações do Preço")
st.markdown(
    '''
    1. **Picos de Preço:** Investigaremos eventos como guerras ou crises econômicas que coincidem com os picos.
    2. **Sazonalidade:** Identificação de padrões anuais ou mensais.
    3. **Recuperação após Crises:** Como o preço do petróleo se comporta após choques econômicos ou políticos.
    4. **Tendências de Longo Prazo:** Mudanças estruturais no mercado de energia.
    '''
)

# Gráficos interativos para exploração
st.subheader("Análise Interativa")
window_size = st.slider("Escolha o intervalo de média móvel (em dias):", 1, 365, 30)
data['Rolling Mean'] = data['Price'].rolling(window=window_size).mean()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Date'], data['Price'], label='Preço Real', color='blue', alpha=0.5)
ax.plot(data['Date'], data['Rolling Mean'], label=f'Média Móvel ({window_size} dias)', color='orange')
ax.set_title('Preço do Petróleo com Média Móvel')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD)')
ax.legend()
st.pyplot(fig)
