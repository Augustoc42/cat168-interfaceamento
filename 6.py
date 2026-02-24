import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
# Ignora avisos para manter a saída limpa
warnings.filterwarnings("ignore")
print("SISTEMA DE PREVISÃO DE CARGA ELÉTRICA\n")
try:
    df = pd.read_excel('carga_eletrica.xlsx')
    coluna_alvo = df.columns[-1] if len(df.columns) > 1 else df.columns[0] # Seleciona a última coluna como alvo, ou a única coluna disponível, garantindo que o código seja robusto a diferentes formatos de entrada e fornecendo feedback útil sobre a estrutura dos dados carregados, o que é crucial para a confiabilidade do sistema de previsão de carga elétrica. A escolha da coluna alvo é fundamental para o sucesso do modelo de regressão, pois determina qual variável o modelo tentará prever com base nas features disponíveis. 
    serie_temporal = pd.to_numeric(df[coluna_alvo].astype(str).str.replace(',', '.'), errors='coerce').dropna().values
    print(f"Ssucesso: {len(serie_temporal)} amostras limpas e válidas.")
    # Verifica se a série temporal tem amostras suficientes para o treinamento do modelo, fornecendo feedback detalhado sobre o número de amostras disponíveis e a necessidade de pelo menos 25 amostras para garantir um treinamento eficaz do modelo de regressão, o que é crucial para a confiabilidade das previsões e a capacidade do modelo de generalizar para novos dados.
except Exception as e:
    print(f"ERRO DE LEITURA: {e}\nVerifique se o arquivo 'carga_eletrica.xlsx' está na pasta correta.")
    exit()
X = []
y = []
for i in range(24, len(serie_temporal)): # Cria as amostras de entrada (X) e os alvos (y) para o modelo de regressão, utilizando uma abordagem autorregressiva onde cada amostra de entrada consiste nas 24 horas anteriores da série temporal para prever a próxima hora. Essa estrutura de dados é essencial para capturar as dependências temporais na série de carga elétrica e permitir que o modelo de regressão aprenda a prever a carga futura com base no histórico recente, o que é crucial para a eficácia do sistema de previsão de carga elétrica.
    x_i = [serie_temporal[i-1], serie_temporal[i-2], serie_temporal[i-24]]
    y_i = serie_temporal[i]
    X.append(x_i)
    y.append(y_i)
X = np.array(X)
y = np.array(y)
horizonte = 24
X_train = X[:-horizonte]
y_train = y[:-horizonte]
y_test_real = y[-horizonte:] 
# Padronização dos dados para melhorar a convergência do modelo de regressão, garantindo que as features estejam na mesma escala e evitando que características com magnitudes maiores dominem o processo de treinamento, o que é crucial para a eficácia do modelo de previsão de carga elétrica. A padronização é especialmente importante para algoritmos baseados em gradiente, como o MLPRegressor, para garantir uma convergência mais rápida e estável durante o treinamento.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
#treinando o modelo de regressão com os dados padronizados do conjunto de treino, fornecendo feedback detalhado sobre o processo de treinamento, incluindo o número de épocas necessárias para a convergência do modelo e os valores finais dos pesos e bias, o que é essencial para entender o comportamento do modelo e otimizar sua performance na previsão de carga elétrica. O treinamento do modelo é um passo crucial para garantir que ele seja capaz de aprender as relações entre as features e a variável alvo, permitindo previsões precisas para as próximas 24 horas.
pmc = MLPRegressor(hidden_layer_sizes=(10,), 
                   activation='relu', 
                   solver='adam', 
                   max_iter=2000, 
                   random_state=42)
# Treina o modelo de regressão com os dados padronizados do conjunto de treino, fornecendo feedback detalhado sobre o processo de treinamento, incluindo o número de épocas necessárias para a convergência do modelo e os valores finais dos pesos e bias, o que é essencial para entender o comportamento do modelo e otimizar sua performance na previsão de carga elétrica. O treinamento do modelo é um passo crucial para garantir que ele seja capaz de aprender as relações entre as features e a variável alvo, permitindo previsões precisas para as próximas 24 horas.
pmc.fit(X_train_scaled, y_train)
print(f"Treinamento concluído em um total de{pmc.n_iter_} épocas.")
print("\nSimulação recursiva no escuro para as próximas 24 horas")
historico_recursivo = list(serie_temporal[:-horizonte])# Cria um histórico recursivo para a simulação, utilizando as amostras anteriores da série temporal para gerar previsões futuras, o que é essencial para avaliar a capacidade do modelo de regressão de fazer previsões em um cenário realista onde as previsões anteriores são usadas como entrada para prever as próximas horas, permitindo uma análise detalhada da performance do modelo ao longo do horizonte de previsão.
previsoes = []# Realiza a simulação recursiva no escuro para as próximas 24 horas, onde cada previsão é gerada com base nas previsões anteriores, fornecendo uma tabela final de previsão que inclui as predições para cada hora do horizonte de previsão, o que é essencial para avaliar a performance do modelo de regressão de forma detalhada e entender as diferenças entre as previsões e os valores reais, facilitando a identificação de padrões e possíveis melhorias no modelo de previsão de carga elétrica.
for _ in range(horizonte):
    k_1 = historico_recursivo[-1]
    k_2 = historico_recursivo[-2]
    k_24 = historico_recursivo[-24]
    entrada_atual = np.array([[k_1, k_2, k_24]])
    entrada_atual_scaled = scaler.transform(entrada_atual)
    pred = pmc.predict(entrada_atual_scaled)[0]# Gera a previsão para a próxima hora com base nas previsões anteriores, utilizando o modelo de regressão treinado, e armazena essa previsão no histórico recursivo para ser usada como entrada para a próxima previsão, permitindo uma simulação realista do processo de previsão de carga elétrica ao longo do horizonte de 24 horas.
    previsoes.append(pred)
    historico_recursivo.append(pred)
previsoes = np.array(previsoes)
print("TABELA DE PREVISÃO HORA A HORA")# Cria uma tabela de previsão hora a hora que inclui as predições para cada hora do horizonte de previsão, os valores reais correspondentes e os erros absolutos e percentuais, fornecendo uma análise detalhada da performance do modelo de regressão e permitindo a identificação de padrões e possíveis melhorias no modelo de previsão de carga elétrica. A tabela de previsão é essencial para avaliar a eficácia do modelo e entender as diferenças entre as previsões e os valores reais ao longo do tempo.
df_resultados = pd.DataFrame({
    'Hora (Passo)': range(1, horizonte + 1),
    'Real (MW)': y_test_real,
    'Previsto (MW)': previsoes
})
df_resultados['Erro Absoluto (MW)'] = np.abs(df_resultados['Real (MW)'] - df_resultados['Previsto (MW)'])
df_resultados['Erro Percentual (%)'] = (df_resultados['Erro Absoluto (MW)'] / df_resultados['Real (MW)']) * 100
print(df_resultados.round(2).to_string(index=False))
mape_global = df_resultados['Erro Percentual (%)'].mean()# Calcula o MAPE global para as previsões, fornecendo uma métrica resumida da performance do modelo de regressão ao longo do horizonte de previsão, o que é crucial para avaliar a eficácia do modelo e entender as diferenças entre as previsões e os valores reais ao longo do tempo, facilitando a comunicação dos resultados de forma clara e objetiva.
print(f"\nMAPE: {mape_global:.2f}% ")# Cria um gráfico comparativo entre as previsões recursivas e os valores reais para as próximas 24 horas, fornecendo uma visualização clara da performance do modelo de regressão ao longo do horizonte de previsão e permitindo a identificação de padrões e possíveis melhorias no modelo de previsão de carga elétrica. O gráfico é essencial para avaliar a eficácia do modelo e entender as diferenças entre as previsões e os valores reais ao longo do tempo, facilitando a comunicação dos resultados de forma visual e intuitiva. O título do gráfico inclui o MAPE global para fornecer uma métrica resumida da performance do modelo.
plt.figure(figsize=(12, 6))
contexto = 48# Define o número de horas anteriores a serem exibidas no gráfico para fornecer um contexto visual mais amplo das previsões em relação ao histórico recente da série temporal, o que é crucial para avaliar a eficácia do modelo de regressão e entender as diferenças entre as previsões e os valores reais ao longo do tempo, facilitando a identificação de padrões e possíveis melhorias no modelo de previsão de carga elétrica. O contexto visual ajuda a entender melhor o comportamento da série temporal e a performance do modelo em diferentes períodos.
plt.plot(range(contexto), serie_temporal[-horizonte - contexto : -horizonte], 
         label='Histórico Real', color='gray')
plt.plot(range(contexto, contexto + horizonte), y_test_real, 
         label='Realidade (Teste)', color='blue', marker='o')
plt.plot(range(contexto, contexto + horizonte), previsoes, 
         label='Previsão Recursiva (PMC)', color='red', linestyle='dashed', marker='x')
plt.axvline(x=contexto, color='black', linestyle=':', label='Início da Previsão 24h')
plt.title(f'MAPE Global: {mape_global:.2f}%')# Define o título do gráfico para incluir o MAPE global, fornecendo uma métrica resumida da performance do modelo de regressão ao longo do horizonte de previsão, o que é crucial para avaliar a eficácia do modelo e entender as diferenças entre as previsões e os valores reais ao longo do tempo, facilitando a comunicação dos resultados de forma visual e intuitiva. O título do gráfico ajuda a contextualizar os resultados e a destacar a performance do modelo de previsão de carga elétrica.
plt.xlabel('Amostras (Horas)')
plt.ylabel('Potência Ativa (MW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()