import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Configuração da página
st.set_page_config(page_title="POSTECH - DTAT - Datathon - Fase 5", layout="wide")

# Título do app
st.markdown("# 🏆 POSTECH - DTAT - Datathon - Fase 5")
st.markdown("### 📌 Integrantes do Grupo: Fábio Cervantes Lima, Guilherme Vieira Magalhães")

# Carregar o modelo treinado
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Sidebar para entrada de dados
st.sidebar.header("📊 Insira os dados do aluno para prever o IAA_2022")

# Definir campos para entrada de variáveis
features = [
    "INDE_2022", "IEG_2022", "IPV_2022", "IDA_2022", "NOTA_MAT_2022", "NOTA_PORT_2022",
    "CG_2022", "CT_2022", "QTD_AVAL_2022", "FASE_2022"
]

dados_usuario = []
for feature in features:
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
