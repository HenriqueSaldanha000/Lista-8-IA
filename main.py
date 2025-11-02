# main.py

import numpy as np
from rede_neural import RedeNeural 

# XOR
print("XOR")

X_xor = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
])
Y_xor = np.array([
    [0, 1, 1, 0]
])

arquitetura_xor = [2, 4, 1] 

# Usando tanh como ativação
nn_xor = RedeNeural(dimensoes_camadas=arquitetura_xor, ativacao='tanh')

taxa_xor = 0.1
epochs_xor = 20000
nn_xor.treinar(X_xor, Y_xor, epochs=epochs_xor, taxa_aprendizado=taxa_xor)

previsoes_xor = nn_xor.prever(X_xor)
print("\nResultados do XOR:")
print(f"Previsão da Rede: \n{previsoes_xor}")
print("Previsões arredondadas:")
print(np.round(previsoes_xor))

print("\n" + "-"*50 + "\n")

# Dígitos 7-Segmentos

print("7-Segmentos")

arquitetura_7seg = [7, 5, 10] 
# Ativação sigmoid
nn_7seg = RedeNeural(dimensoes_camadas=arquitetura_7seg, ativacao='sigmoid') 

X_7seg = np.array([
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
])

Y_7seg = np.identity(10).T

nn_7seg.treinar(X_7seg, Y_7seg, epochs=30000, taxa_aprendizado=0.15) 

previsoes_7seg = nn_7seg.prever(X_7seg)

print("\nResultados 7-Segmentos")
previsoes_digitos = np.argmax(previsoes_7seg, axis=0)
esperado_digitos = np.argmax(Y_7seg, axis=0)

print(f"Dígitos Esperados: {esperado_digitos}")
print(f"Dígitos Previstos: {previsoes_digitos}")
acertos = np.sum(previsoes_digitos == esperado_digitos)
print(f"Acurácia: {acertos} / 10")

print("\nTeste com Ruído")

digito_8_ruido = np.array([[1, 1, 1, 1, 1, 1, 0]]).T 
previsao_8 = nn_7seg.prever(digito_8_ruido)
print(f"Entrada: 8 com segmento g desligado. Previsão: {np.argmax(previsao_8, axis=0)[0]}")