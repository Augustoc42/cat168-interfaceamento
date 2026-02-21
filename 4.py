import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def main():
    print("LDR_ball.xlsx...")
    try:
        df = pd.read_excel('LDR_ball.xlsx', header=None)
    except FileNotFoundError:
        print("Erro: O arquivo 'LDR_ball.xlsx' não foi encontrado.")
        return
    df = df.T 
    df.columns = df.iloc[0] 
    df = df[1:].reset_index(drop=True) 
    df = df.apply(pd.to_numeric, errors='coerce')
    X = df.iloc[:, 0:2].values
    y = df.iloc[:, 2].values 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Treinando o modelo MLPRegressor...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(10, 10), 
        activation='relu',           
        solver='adam',               
        max_iter=2000,              
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    print("Avaliando o modelo")
    y_pred = mlp.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("RESULTADOS DO SENSOR VIRTUAL:")
    print(f"Erro Médio Quadrático (MSE): {mse:.4f}")
    print(f"Coeficiente de Determinação (R²): {r2:.4f}")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='royalblue', label='Predições')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal (Erro Zero)')
    plt.xlabel('Posição Real (cm)')
    plt.ylabel('Posição Estimada pelo Sensor Virtual (cm)')
    plt.title('Desempenho do Sensor Virtual (PMC)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
if __name__ == "__main__":
    main()