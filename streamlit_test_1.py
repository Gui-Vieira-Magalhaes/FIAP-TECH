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

# Carregar os dados do CSV
def carregar_dados():
    file_path = "PEDE_PASSOS_DATASET_FIAP.csv"  # Atualizar conforme necessário
    df = pd.read_csv(file_path, delimiter=";")
    return df

df = carregar_dados()

# Selecionar as colunas relevantes para a modelagem
features = [
    "INDE_2022", "IEG_2022", "IPV_2022", "IDA_2022", "NOTA_MAT_2022", "NOTA_PORT_2022",
    "CG_2022", "CT_2022", "QTD_AVAL_2022", "FASE_2022"
]
target = "IAA_2022"

# Renomear colunas para remover o sufixo "_2022"
feature_names = {col: col.replace("_2022", "") for col in features}
df = df.rename(columns=feature_names)

# Remover valores nulos
df = df[list(feature_names.values()) + [target]].dropna()
X = df[list(feature_names.values())]
y = df[target]

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

if st.sidebar.radio("Navegação", ["🏆 Conceitos", "📊 Predição do IAA"]) == "🏆 Conceitos":
    st.markdown("# 🏆 POSTECH - DTAT - Datathon - Fase 5")
    st.markdown("### 📌 Integrantes do Grupo: Fábio Cervantes Lima, Guilherme Vieira Magalhães")
    st.write("### ℹ️ O que é o IAA?")
    st.write("O Índice de Aproveitamento Acadêmico (IAA) é uma métrica fundamental para medir o desempenho acadêmico dos alunos, levando em consideração uma variedade de fatores, incluindo notas escolares, participação em atividades extracurriculares e nível de engajamento no processo educacional. O IAA não apenas fornece um retrato atual do progresso acadêmico dos estudantes, mas também permite a identificação de padrões e tendências ao longo do tempo. Isso possibilita que gestores educacionais, professores e pesquisadores compreendam melhor os desafios enfrentados pelos alunos e desenvolvam estratégias eficazes para melhorar o aprendizado. Além disso, o IAA é uma ferramenta essencial para prever o desempenho futuro dos estudantes, o que facilita a implementação de intervenções pedagógicas direcionadas, reduzindo índices de evasão escolar e aumentando a taxa de sucesso acadêmico. Seu uso possibilita que instituições educacionais tomem decisões embasadas em dados, contribuindo para um ensino mais personalizado e eficiente.")
    

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
    
    st.write("### 📊 Escolha do Modelo Preditivo")
    st.write("O **Random Forest** foi escolhido devido à sua robustez na previsão de valores numéricos e sua capacidade de lidar com grandes quantidades de dados sem a necessidade de uma normalização complexa. Este modelo funciona bem com conjuntos de dados de alta dimensionalidade e é resistente ao overfitting devido à sua estrutura baseada em múltiplas árvores de decisão. Ele também permite interpretabilidade por meio da análise da importância das variáveis, fornecendo insights valiosos sobre os fatores que mais influenciam o IAA. Por outro lado, **Redes Neurais** são conhecidas por sua capacidade de capturar padrões não lineares e relações complexas entre variáveis, tornando-as poderosas para diversas tarefas preditivas. No entanto, seu treinamento exige maior poder computacional, além de maior esforço para ajuste de hiperparâmetros, como taxa de aprendizado, número de camadas e neurônios. Além disso, redes neurais tendem a ser menos interpretáveis, dificultando a explicação dos fatores que impactam diretamente o resultado. Observamos que, para nosso conjunto de dados, o **Random Forest teve melhor aderência**, apresentando um melhor equilíbrio entre desempenho e interpretabilidade, além de menor necessidade de ajuste fino para alcançar bons resultados. Este modelo nos permite extrair conhecimento relevante e embasar a tomada de decisões no contexto educacional de forma eficaz.")
    
    
    # Adicionar gráficos relevantes
    st.write("### 📊 Visualização de Dados")
    st.write("#### 🔍 Matriz de Correlação")
    st.write("Este gráfico exibe a correlação entre todas as variáveis do dataset, ajudando a entender quais variáveis possuem maior influência sobre o IAA.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    
    st.write("#### 📊 Distribuição do IAA")
    st.write("Este gráfico mostra a distribuição do IAA no dataset, ajudando a identificar tendências e padrões de desempenho acadêmico.")
    fig, ax = plt.subplots()
    sns.histplot(df[target], bins=20, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.write("#### 📊 Boxplot das Notas")
    st.write("Este boxplot compara as distribuições das notas de Matemática e Português para identificar a variabilidade e possíveis outliers.")
    fig, ax = plt.subplots()
    sns.boxplot(data=df.melt(value_vars=["NOTA_MAT", "NOTA_PORT"], var_name='Disciplina', value_name='Nota'), x='Disciplina', y='Nota', ax=ax)
    st.pyplot(fig)
    
    st.write("#### 📊 Relação entre IAA e INDE")
    st.write("Este gráfico mostra a relação entre o Índice de Desenvolvimento Educacional (INDE) e o IAA, ajudando a entender como a qualidade da educação impacta o desempenho acadêmico.")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="INDE", y=target, ax=ax)
    st.pyplot(fig)
    
    st.write("#### 📊 Comparação de Engajamento (IEG) por Fase Acadêmica")
    st.write("Este gráfico compara os níveis de engajamento dos alunos em diferentes fases acadêmicas, permitindo observar padrões de envolvimento ao longo da jornada escolar.")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="FASE", y="IEG", ax=ax)
    st.pyplot(fig)


else:
    st.markdown("# 🏆 POSTECH - DTAT - Datathon - Fase 5")
    st.markdown("### 📌 Integrantes do Grupo: Fábio Cervantes Lima, Guilherme Vieira Magalhães")
    
    st.write("## 🎯 Explicação das Variáveis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("### Variáveis Acadêmicas")
        st.write("- **INDE**: Índice de Desenvolvimento Educacional")
        st.write("- **IEG**: Índice de Engajamento Geral")
        st.write("- **IPV**: Índice de Ponto de Virada")
    with col2:
        st.write("### Notas e Avaliações")
        st.write("- **IDA**: Índice de Desenvolvimento Acadêmico")
        st.write("- **NOTA_MAT**: Nota de Matemática")
        st.write("- **NOTA_PORT**: Nota de Português")
    with col3:
        st.write("### Outras Variáveis")
        st.write("- **CG**: Carga de Grau")
        st.write("- **CT**: Carga Total")
        st.write("- **QTD_AVAL**: Quantidade de Avaliações")
        st.write("- **FASE**: Fase Acadêmica")
    
    st.write("## 🎯 Insira os dados do aluno para prever o IAA")
    st.write("Os dados foram previamente preenchidos com os dados médios do dataset")
    col1, col2, col3 = st.columns(3)
    dados_usuario = []
    for i, feature in enumerate(feature_names.values()):
        with [col1, col2, col3][i % 3]:
            valor = st.number_input(f"{feature}", min_value=0.0, max_value=10.0, step=0.1, value=min(10.0, max(0.0, df[feature].mean())))
            dados_usuario.append(valor)
    
    if st.button("🔍 Prever IAA"):
        dados_usuario_np = np.array(dados_usuario).reshape(1, -1)
        dados_usuario_scaled = scaler.transform(dados_usuario_np)
        rf_previsao = rf_model.predict(dados_usuario_scaled)[0]
        nn_previsao = nn_model.predict(dados_usuario_scaled)[0]
        
        st.success(f"🎯 Previsão do IAA com Random Forest: {rf_previsao:.2f}")
        st.success(f"🎯 Previsão do IAA com Redes Neurais: {nn_previsao:.2f}")
        st.write("📌 **Baseado na avaliação dos modelos, a predição do Random Forest é a mais aderente ao conjunto de dados.**")


