import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def main():
    print("LDR_ball.xls")

    # Tratamento de erros para garantir que o arquivo seja carregado corretamente e fornecer feedback útil em caso de falha na leitura do arquivo
    try:
        # Carrega os dados do arquivo Excel, assumindo que os dados estão organizados em colunas sem um cabeçalho específico e que a primeira linha contém os nomes das features
        df = pd.read_excel('LDR_ball.xlsx', header=None)
    except FileNotFoundError:
        print("Erro")
        return
    # Transpõe a matriz para que as colunas representem as features e as linhas representem as amostras, facilitando a manipulação dos dados para treinamento do modelo
    df = df.T 
    df.columns = df.iloc[0] 
    df = df[1:].reset_index(drop=True) 
    df = df.apply(pd.to_numeric, errors='coerce')
    # Separação das features (X) e do alvo (y) para o treinamento do modelo de regressão:
    # X recebe as duas primeiras colunas (posições x e y da bola) como variáveis independentes para prever a posição real da bola, que é a terceira coluna (y) e representa a variável dependente que o modelo tentará estimar com base nas features fornecidas.
    X = df.iloc[:, 0:2].values
    # y recebe a terceira coluna (posição real da bola) como a variável alvo que o modelo de regressão tentará prever com base nas features fornecidas em X.
    y = df.iloc[:, 2].values 
    
    # Holdout: divisão dos dados em conjuntos de treino e teste para avaliar a performance do modelo em dados não vistos durante o treinamento, garantindo uma avaliação justa da capacidade de generalização do modelo. O parâmetro test_size=0.2 indica que 20% dos dados serão reservados para teste, enquanto os 80% restantes serão usados para treinar o modelo. O parâmetro random_state=42 é utilizado para garantir a reprodutibilidade da divisão dos dados, permitindo que os mesmos conjuntos de treino e teste sejam gerados em execuções subsequentes do código.
    # random_state é uma semente para o gerador de números aleatórios, garantindo que a divisão dos dados seja a mesma em cada execução do código, o que é crucial para a reprodutibilidade dos resultados e para comparações consistentes entre diferentes modelos ou configurações.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Padronização (Z-score normalization): essencial para otimizar a convergência dos pesos da Rede Neural
    scaler = StandardScaler()
    # Extrai média e desvio padrão do treino e já aplica a transformação
    X_train_scaled = scaler.fit_transform(X_train)
    # Aplica a transformação no teste usando os parâmetros do treino para evitar vazamento de dados (data leakage)
    X_test_scaled = scaler.transform(X_test)
    
    # Instanciação da Rede Neural Multilayer Perceptron (MLPRegressor) com hiperparâmetros otimizados para regressão:
    mlp = MLPRegressor(
        hidden_layer_sizes=(10, 10), # Arquitetura: 2 camadas ocultas com 10 neurônios cada
        activation='relu',           # Função de ativação ReLU para introduzir não linearidade e evitar o problema do gradiente desaparecido
        solver='adam',               # Otimizador baseado em gradiente estocástico adaptativo para acelerar a convergência
        max_iter=2000,               # Limite máximo de épocas para garantir que o treinamento seja concluído mesmo em casos de convergência lenta
        random_state=42              # Semente para garantir a reprodutibilidade dos resultados
    )
    
    # Executa o algoritmo de treinamento da Rede Neural com os dados padronizados do conjunto de treino
    mlp.fit(X_train_scaled, y_train)
    
    print("Avaliando o modelo")
    
    # Realiza a predição no conjunto de teste para comparar com os valores reais
    y_pred = mlp.predict(X_test_scaled)
    
    # Cálculo das métricas de avaliação para quantificar a performance do modelo:
    # MSE: Média dos erros quadrados para avaliar a precisão das predições, penalizando mais fortemente os erros maiores, o que é útil para identificar a qualidade do ajuste do modelo às amostras de teste.
    mse = mean_squared_error(y_test, y_pred)
    # R2: Proporção da variância explicada pelo modelo para avaliar a qualidade do ajuste
    r2 = r2_score(y_test, y_pred)
    print("RESULTADO:")
    print(f"Erro Médio Quadrático(MSE): {mse:.4f}")
    print(f"Coeficiente de Determinação(R^2): {r2:.4f}")
    # Gráfico de dispersão para visualizar os resíduos
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue', label='Predições')
    
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal (Erro Zero)')
    plt.xlabel('Posição Real (cm)')
    plt.ylabel('Posição Estimada (cm)')
    plt.title('(PMC)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()