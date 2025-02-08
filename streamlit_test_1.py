import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuração da página
st.set_page_config(page_title="POSTECH - DTAT - Datathon - Fase 5", layout="wide")

# Título do app
st.markdown("# 🏆 POSTECH - DTAT - Datathon - Fase 5")
st.markdown("### 📌 Integrantes do Grupo: Fábio Cervantes Lima, Guilherme Vieira Magalhães")

# Gerar dados fictícios (simulando um dataset similar ao real)
def gerar_dados():
    np.random.seed(42)
    tamanho = 1000
    dados = {
        "INDE_2022": np.random.uniform(5, 10, tamanho),
        "IEG_2022": np.random.uniform(5, 10, tamanho),
        "IPV_2022": np.random.uniform(5, 10, tamanho),
        "IDA_2022": np.random.uniform(3, 9, tamanho),
        "NOTA_MAT_2022": np.random.uniform(3, 10, tamanho),
        "NOTA_PORT_2022": np.random.uniform(3, 10, tamanho),
        "CG_2022": np.random.uniform(0, 10, tamanho),
        "CT_2022": np.random.uniform(0, 10, tamanho),
        "QTD_AVAL_2022": np.random.randint(2, 5, tamanho),
        "FASE_2022": np.random.randint(1, 7, tamanho),
        "IAA_2022": np.random.uniform(5, 10, tamanho),
    }
    return pd.DataFrame(dados)

# Gerar e treinar o modelo automaticamente
df = gerar_dados()
X = df.drop(columns=["IAA_2022"])
y = df["IAA_2022"]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinar modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# Exibir métricas
st.write("### 📊 Avaliação do Modelo")
st.write(f"- **Erro Absoluto Médio (MAE):** {mae:.2f}")
st.write(f"- **Raiz do Erro Quadrático Médio (RMSE):** {rmse:.2f}")
st.write(f"- **Coeficiente de Determinação (R²):** {r2:.2f}")

# Sidebar para entrada de dados
st.sidebar.header("📊 Insira os dados do aluno para prever o IAA_2022")
dados_usuario = []
for feature in X.columns:
    valor = st.sidebar.number_input(f"{feature}", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    dados_usuario.append(valor)

# Botão para realizar previsão
if st.sidebar.button("🔍 Prever IAA_2022"):
    dados_usuario_np = np.array(dados_usuario).reshape(1, -1)
    dados_usuario_scaled = scaler.transform(dados_usuario_np)
    previsao = model.predict(dados_usuario_scaled)[0]
    st.success(f"🎯 Previsão do IAA_2022: {previsao:.2f}")

# Rodapé
st.markdown("---")
st.markdown("📌 *Datathon desenvolvido para a Fase 5 do POSTECH DTAT*")
