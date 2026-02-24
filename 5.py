import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings

# Ignora avisos para manter a saída limpa
warnings.filterwarnings("ignore")
print("SISTEMA DE CLASSIFICAÇÃO DE ÓLEO DE TRANSFORMADOR\n")
# Função para carregar e limpar os dados, adaptada para lidar com diferentes formatos de entrada, seja em formato tabular ou em linhas de texto, garantindo a flexibilidade na leitura dos dados de treinamento e teste.
def carregar_e_limpar(nome_arquivo, aba, is_teste=False):
    df = pd.read_excel(nome_arquivo, sheet_name=aba) 
    primeira_coluna = df.columns[0]
    amostra = str(df.iloc[0, 0])
    if len(amostra.split()) > 1:
        linhas_texto = df[primeira_coluna].astype(str).tolist()
        dados_processados = []
        for linha in linhas_texto:
            valores = linha.replace(',', '.').split()
            tamanho_esperado = 3 if is_teste else 4
            if len(valores) == tamanho_esperado:
                try:
                    dados_processados.append([float(v) for v in valores]) # Converte os valores para float, substituindo vírgulas por pontos e garantindo que apenas linhas com o número correto de valores sejam processadas, o que é crucial para evitar erros de formatação e garantir a integridade dos dados para o treinamento e teste do modelo de classificação. Linhas com formato incorreto são ignoradas para manter a qualidade do conjunto de dados.
                except ValueError:
                    continue
        colunas = ['x1', 'x2', 'x3'] if is_teste else ['x1', 'x2', 'x3', 'classe']
        return pd.DataFrame(dados_processados, columns=colunas)
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        return df
try:
    # Carrega e limpa os dados de treinamento e teste, garantindo que o processo seja robusto a diferentes formatos de entrada e fornecendo feedback útil em caso de falha na leitura do arquivo, o que é essencial para a confiabilidade do sistema de classificação de óleo de transformador.
    df_treino = carregar_e_limpar('dadosTrafos.xlsx', 'tab_treinamento', is_teste=False)
    df_teste = carregar_e_limpar('dadosTrafos.xlsx', 'tab_teste', is_teste=True)
    
    # Verifica se os dados foram carregados corretamente e se as colunas esperadas estão presentes, fornecendo feedback detalhado sobre o número de amostras e a estrutura dos dados, o que é crucial para garantir que o modelo de classificação seja treinado e testado com conjuntos de dados adequados e bem formatados.
    if df_treino.empty:
        print("ERRO: O conjunto de treinamento está vazio.")
        exit()
except Exception as e:
    print(f"ERRO DE LEITURA: {e}\nVerifique se o arquivo está na mesma pasta.")
    exit()
X_train = df_treino[['x1', 'x2', 'x3']].values
y_train = df_treino['classe'].values
X_test = df_teste[['x1', 'x2', 'x3']].values
scaler = StandardScaler() # Padronização dos dados para melhorar a convergência do modelo de classificação, garantindo que as features estejam na mesma escala e evitando que características com magnitudes maiores dominem o processo de treinamento, o que é crucial para a eficácia do modelo de classificação de óleo de transformador. A padronização é especialmente importante para algoritmos baseados em gradiente, como o MLPClassifier, para garantir uma convergência mais rápida e estável durante o treinamento.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def treinar_com_pesos_customizados(seed, nome): # Função para treinar o modelo de classificação com pesos e bias personalizados, permitindo a experimentação com diferentes inicializações e fornecendo feedback detalhado sobre o processo de treinamento, incluindo a convergência do modelo e os valores finais dos pesos e bias, o que é essencial para entender o comportamento do modelo e otimizar sua performance na classificação de óleo de transformador.
    print(f"{nome}")
    np.random.seed(seed)
    pmc = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh', solver='adam', random_state=seed) # Instanciação do modelo de classificação com uma arquitetura simples (1 camada oculta com 4 neurônios) e hiperparâmetros otimizados para a tarefa de classificação, utilizando a função de ativação tangente hiperbólica para introduzir não linearidade e o otimizador Adam para acelerar a convergência, garantindo uma base sólida para o treinamento do modelo de classificação de óleo de transformador. A escolha desses hiperparâmetros é crucial para a eficácia do modelo e sua capacidade de generalização.
    pmc.partial_fit(X_train_scaled, y_train, classes=np.unique(y_train))
    pmc.coefs_[0] = np.random.uniform(0, 1, pmc.coefs_[0].shape)
    pmc.coefs_[1] = np.random.uniform(0, 1, pmc.coefs_[1].shape)
    pmc.intercepts_[0] = np.random.uniform(0, 1, pmc.intercepts_[0].shape)
    pmc.intercepts_[1] = np.random.uniform(0, 1, pmc.intercepts_[1].shape)
    loss_anterior = float('inf')
    epocas = 0
    for i in range(2000):
        pmc.partial_fit(X_train_scaled, y_train)
        epocas += 1
        if abs(loss_anterior - pmc.loss_) < 1e-4:# Critério de convergência: se a mudança na função de perda for menor que um limiar, considera-se que o modelo convergiu, o que é crucial para evitar overfitting e garantir que o modelo de classificação de óleo de transformador tenha uma performance estável e generalizável. O critério de convergência ajuda a determinar quando o modelo atingiu um ponto ótimo durante o treinamento, evitando iterações desnecessárias e garantindo uma eficiência computacional adequada.
            break
        loss_anterior = pmc.loss_
    print(f" Convergiu em {epocas} épocas.")
    print(f"\n[W1] Matriz de Pesos:\n{np.round(pmc.coefs_[0], 4)}")
    print(f"\n[b1] Vetor de Bias:{np.round(pmc.intercepts_[0], 4)}")
    print(f"\n[W2] Matriz de Pesos:\n{np.round(pmc.coefs_[1], 4)}")
    print(f"\n[b2] Bias:{np.round(pmc.intercepts_[1], 4)}")
    return pmc
pmc_T1 = treinar_com_pesos_customizados(42, "TREINAMENTO T1")
pmc_T2 = treinar_com_pesos_customizados(99, "TREINAMENTO T2")
# Realiza a predição no conjunto de teste para comparar com os valores reais, fornecendo uma tabela final de teste que inclui as predições dos dois modelos treinados (T1 e T2) para cada amostra do conjunto de teste, o que é essencial para avaliar a performance dos modelos de classificação de óleo de transformador e entender as diferenças entre as duas inicializações de pesos personalizadas. A tabela final permite uma análise detalhada das predições em relação aos dados reais, facilitando a identificação de padrões e possíveis melhorias no modelo.
print("TABELA FINAL DE TESTE")
y_pred_T1 = pmc_T1.predict(X_test_scaled)
y_pred_T2 = pmc_T2.predict(X_test_scaled)
df_final = df_teste.copy()
df_final['y (T1)'] = y_pred_T1
df_final['y (T2)'] = y_pred_T2
print(df_final.to_string(index=False))
