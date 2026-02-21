import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
print("SISTEMA DE CLASSIFICAÇÃO DE ÓLEO DE TRANSFORMADOR\n")
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
                    dados_processados.append([float(v) for v in valores])
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
    df_treino = carregar_e_limpar('dadosTrafos.xlsx', 'tab_treinamento', is_teste=False)
    df_teste = carregar_e_limpar('dadosTrafos.xlsx', 'tab_teste', is_teste=True)
except Exception as e:
    print(f"ERRO DE LEITURA: {e}\nVerifique se o arquivo está na mesma pasta.")
    exit()
X_train = df_treino[['x1', 'x2', 'x3']].values
y_train = df_treino['classe'].values
X_test = df_teste[['x1', 'x2', 'x3']].values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def treinar_com_pesos_customizados(seed, nome):
    print(f"{nome}")
    np.random.seed(seed)
    pmc = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh', solver='adam', random_state=seed)
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
        if abs(loss_anterior - pmc.loss_) < 1e-4:
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
print("TABELA FINAL DE TESTE")
y_pred_T1 = pmc_T1.predict(X_test_scaled)
y_pred_T2 = pmc_T2.predict(X_test_scaled)
df_final = df_teste.copy()
df_final['y (T1)'] = y_pred_T1
df_final['y (T2)'] = y_pred_T2
print(df_final.to_string(index=False))
