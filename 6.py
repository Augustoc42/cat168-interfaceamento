import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
print("SISTEMA DE PREVISÃO DE CARGA ELÉTRICA\n")
try:
    df = pd.read_excel('carga_eletrica.xlsx')
    coluna_alvo = df.columns[-1] if len(df.columns) > 1 else df.columns[0] 
    serie_temporal = pd.to_numeric(df[coluna_alvo].astype(str).str.replace(',', '.'), errors='coerce').dropna().values
    print(f"Série temporal carregada com sucesso: {len(serie_temporal)} amostras limpas e válidas.")
except Exception as e:
    print(f"ERRO DE LEITURA: {e}\nVerifique se o arquivo 'carga_eletrica.xlsx' está na pasta correta.")
    exit()
X = []
y = []
for i in range(24, len(serie_temporal)):
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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("\nTreinando o modelo Autorregressivo (MLPRegressor)...")
pmc = MLPRegressor(hidden_layer_sizes=(10,), 
                   activation='relu', 
                   solver='adam', 
                   max_iter=2000, 
                   random_state=42)

pmc.fit(X_train_scaled, y_train)
print(f"Treinamento concluído em {pmc.n_iter_} épocas.")
print("\nExecutando simulação recursiva no escuro para as próximas 24 horas...")
historico_recursivo = list(serie_temporal[:-horizonte])
previsoes = []
for _ in range(horizonte):
    k_1 = historico_recursivo[-1]
    k_2 = historico_recursivo[-2]
    k_24 = historico_recursivo[-24]
    entrada_atual = np.array([[k_1, k_2, k_24]])
    entrada_atual_scaled = scaler.transform(entrada_atual)
    pred = pmc.predict(entrada_atual_scaled)[0]
    previsoes.append(pred)
    historico_recursivo.append(pred)
previsoes = np.array(previsoes)
print("TABELA DE PREVISÃO HORA A HORA")
df_resultados = pd.DataFrame({
    'Hora (Passo)': range(1, horizonte + 1),
    'Real (MW)': y_test_real,
    'Previsto (MW)': previsoes
})
df_resultados['Erro Absoluto (MW)'] = np.abs(df_resultados['Real (MW)'] - df_resultados['Previsto (MW)'])
df_resultados['Erro Percentual (%)'] = (df_resultados['Erro Absoluto (MW)'] / df_resultados['Real (MW)']) * 100
print(df_resultados.round(2).to_string(index=False))
mape_global = df_resultados['Erro Percentual (%)'].mean()
print(f"\nMAPE: {mape_global:.2f}% ")
plt.figure(figsize=(12, 6))
contexto = 48
plt.plot(range(contexto), serie_temporal[-horizonte - contexto : -horizonte], 
         label='Histórico Real', color='gray')
plt.plot(range(contexto, contexto + horizonte), y_test_real, 
         label='Realidade (Teste)', color='blue', marker='o')
plt.plot(range(contexto, contexto + horizonte), previsoes, 
         label='Previsão Recursiva (PMC)', color='red', linestyle='dashed', marker='x')
plt.axvline(x=contexto, color='black', linestyle=':', label='Início da Previsão 24h')
plt.title(f'MAPE Global: {mape_global:.2f}%')
plt.xlabel('Amostras (Horas)')
plt.ylabel('Potência Ativa (MW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()