
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

# Exibir os dados
st.subheader("Dados Históricos")
st.write(data.head())

# Visualização do preço ao longo do tempo com eventos históricos
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

# Insights com storytelling
st.subheader("Insights sobre Variações do Preço")
st.markdown(
    '''
    1. **Impacto de Decisões da OPEP:** Decisões como cortes ou manutenção de produção influenciam diretamente o preço.
    2. **Crises Geopolíticas:** Saída dos EUA do acordo com o Irã, invasão da Ucrânia e tensões no Oriente Médio geraram alta volatilidade.
    3. **Pandemia de COVID-19:** Redução global na demanda de energia provocou quedas históricas.
    4. **Tendências Longo Prazo:** Eventos críticos alteram tendências de longo prazo no mercado.
    '''
)

# Gráficos interativos para explorar médias móveis
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

# Preparação para o modelo de previsão
st.subheader("Modelo de Previsão")
st.markdown("**Próximo passo:** Implementar um modelo de Machine Learning para prever os preços futuros.")
