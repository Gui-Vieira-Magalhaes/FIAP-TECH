import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuração da página
st.set_page_config(page_title="POSTECH - DTAT - Datathon - Fase 5", layout="wide")

# Criar páginas
menu = st.sidebar.radio("Navegação", ["🏆 Conceitos", "📊 Predição do IAA"])

# Gerar dados fictícios (simulando um dataset similar ao real)
def gerar_dados():
    np.random.seed(42)
    tamanho = 1000
    dados = {
        "INDE": np.random.uniform(5, 10, tamanho),
        "IEG": np.random.uniform(5, 10, tamanho),
        "IPV": np.random.uniform(5, 10, tamanho),
        "IDA": np.random.uniform(3, 9, tamanho),
        "NOTA_MAT": np.random.uniform(3, 10, tamanho),
        "NOTA_PORT": np.random.uniform(3, 10, tamanho),
        "CG": np.random.uniform(0, 10, tamanho),
        "CT": np.random.uniform(0, 10, tamanho),
        "QTD_AVAL": np.random.randint(2, 5, tamanho),
        "FASE": np.random.randint(1, 7, tamanho),
        "IAA": np.random.uniform(5, 10, tamanho),
    }
    return pd.DataFrame(dados)

# Gerar e treinar os modelos automaticamente
df = gerar_dados()
X = df.drop(columns=["IAA"])
y = df["IAA"]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinar Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Treinar Rede Neural
nn_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)

if menu == "🏆 Conceitos":
    st.markdown("# 🏆 POSTECH - DTAT - Datathon - Fase 5")
    st.markdown("### 📌 Integrantes do Grupo: Fábio Cervantes Lima, Guilherme Vieira Magalhães")
    st.write("### ℹ️ O que é o IAA?")
    st.write("O Índice de Aproveitamento Acadêmico (IAA) é uma métrica que avalia o desempenho do aluno com base em diversos fatores como notas, engajamento e participação.")
    
    st.write("### 📊 Escolha do Modelo Preditivo")
    st.write("O **Random Forest** foi escolhido devido à sua robustez em prever valores numéricos, resistência a overfitting e boa interpretabilidade. Além disso, treinamos um modelo de **Redes Neurais** para avaliar qual dos dois apresenta melhor aderência ao nosso conjunto de dados.")
    
    # Avaliação dos modelos
    rf_pred = rf_model.predict(X_test)
    nn_pred = nn_model.predict(X_test)
    
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
    rf_r2 = r2_score(y_test, rf_pred)
    
    nn_mae = mean_absolute_error(y_test, nn_pred)
    nn_rmse = mean_squared_error(y_test, nn_pred) ** 0.5
    nn_r2 = r2_score(y_test, nn_pred)
    
    st.write("### 📊 Comparação de Modelos")
    st.write(f"#### Random Forest")
    st.write(f"- **Erro Absoluto Médio (MAE):** {rf_mae:.2f}")
    st.write(f"- **Raiz do Erro Quadrático Médio (RMSE):** {rf_rmse:.2f}")
    st.write(f"- **Coeficiente de Determinação (R²):** {rf_r2:.2f}")
    
    st.write(f"#### Redes Neurais")
    st.write(f"- **Erro Absoluto Médio (MAE):** {nn_mae:.2f}")
    st.write(f"- **Raiz do Erro Quadrático Médio (RMSE):** {nn_rmse:.2f}")
    st.write(f"- **Coeficiente de Determinação (R²):** {nn_r2:.2f}")

elif menu == "📊 Predição do IAA":
    st.markdown("# 🏆 POSTECH - DTAT - Datathon - Fase 5")
    st.markdown("### 📌 Integrantes do Grupo: Fábio Cervantes Lima, Guilherme Vieira Magalhães")
    st.write("## 🎯 Insira os dados do aluno para prever o IAA")
    
    dados_usuario = []
    for feature in X.columns:
        valor = st.number_input(f"{feature}", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
        dados_usuario.append(valor)
    
    if st.button("🔍 Prever IAA"):
        dados_usuario_np = np.array(dados_usuario).reshape(1, -1)
        dados_usuario_scaled = scaler.transform(dados_usuario_np)
        rf_previsao = rf_model.predict(dados_usuario_scaled)[0]
        nn_previsao = nn_model.predict(dados_usuario_scaled)[0]
        
        st.success(f"🎯 Previsão do IAA com Random Forest: {rf_previsao:.2f}")
        st.success(f"🎯 Previsão do IAA com Redes Neurais: {nn_previsao:.2f}")
