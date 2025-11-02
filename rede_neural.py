import numpy as np

# Funções de Ativação e Suas Derivadas

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1 - np.tanh(x)**2

def mse(y_verdadeiro, y_previsto):
    return np.mean((y_verdadeiro - y_previsto)**2)

def mse_derivada(y_verdadeiro, y_previsto):
    # Corrigido (sem divisão)
    return 2 * (y_previsto - y_verdadeiro)


# Rede Neural

class RedeNeural:
    
    def __init__(self, dimensoes_camadas, ativacao='sigmoid'):
        self.parametros = {}
        self.num_camadas = len(dimensoes_camadas) - 1
        self.dimensoes = dimensoes_camadas
        
        if ativacao == 'relu':
            self.ativacao = relu
            self.ativacao_derivada = relu_derivada
        elif ativacao == 'tanh':
            self.ativacao = tanh
            self.ativacao_derivada = tanh_derivada
        else:
            self.ativacao = sigmoid
            self.ativacao_derivada = sigmoid_derivada

        for i in range(1, self.num_camadas + 1):
            fan_in = dimensoes_camadas[i-1]
            fan_out = dimensoes_camadas[i]
            
            limite = np.sqrt(6 / (fan_in + fan_out))
            
            self.parametros[f'W{i}'] = np.random.uniform(
                -limite, 
                limite, 
                (fan_out, fan_in)
            )

            self.parametros[f'b{i}'] = np.zeros((fan_out, 1))

    def forward(self, X):
        cache = {}
        A = X
        cache['A0'] = X
        
        for i in range(1, self.num_camadas):
            W = self.parametros[f'W{i}']
            b = self.parametros[f'b{i}']
            Z = np.dot(W, A) + b
            A = self.ativacao(Z)
            cache[f'Z{i}'] = Z
            cache[f'A{i}'] = A
            
        W_final = self.parametros[f'W{self.num_camadas}']
        b_final = self.parametros[f'b{self.num_camadas}']
        Z_final = np.dot(W_final, A) + b_final
        A_final = sigmoid(Z_final)
        
        cache[f'Z{self.num_camadas}'] = Z_final
        cache[f'A{self.num_camadas}'] = A_final

        return A_final, cache

    def backward(self, y_verdadeiro, cache):
        gradientes = {}
        m = y_verdadeiro.shape[1] 
        L = self.num_camadas
        y_previsto = cache[f'A{L}']
        Z_final = cache[f'Z{L}']
        
        dA_final = mse_derivada(y_verdadeiro, y_previsto) 
        dZ_final = dA_final * sigmoid_derivada(Z_final)
        
        A_anterior = cache[f'A{L-1}']
        gradientes[f'dW{L}'] = np.dot(dZ_final, A_anterior.T) / m
        gradientes[f'db{L}'] = np.sum(dZ_final, axis=1, keepdims=True) / m
        dA_anterior = np.dot(self.parametros[f'W{L}'].T, dZ_final)

        for i in reversed(range(1, L)):
            Z = cache[f'Z{i}']
            A_anterior = cache[f'A{i-1}']
            
            dZ = dA_anterior * self.ativacao_derivada(Z) 
            
            gradientes[f'dW{i}'] = np.dot(dZ, A_anterior.T) / m
            gradientes[f'db{i}'] = np.sum(dZ, axis=1, keepdims=True) / m
            
            if i > 1:
                dA_anterior = np.dot(self.parametros[f'W{i}'].T, dZ)
            
        return gradientes

    def atualizar_pesos(self, gradientes, taxa_aprendizado):
        for i in range(1, self.num_camadas + 1):
            self.parametros[f'W{i}'] -= taxa_aprendizado * gradientes[f'dW{i}']
            self.parametros[f'b{i}'] -= taxa_aprendizado * gradientes[f'db{i}']

    def treinar(self, X_treino, Y_treino, epochs, taxa_aprendizado, print_loss=True):
        historico_loss = []
        
        for i in range(epochs + 1):
            Y_previsto, cache = self.forward(X_treino)
            loss = mse(Y_treino, Y_previsto)
            
            if i % (epochs // 10) == 0:
                historico_loss.append(loss)
                if print_loss:
                    print(f"Epoch {i} / {epochs} - Erro: {loss:.6f}")
            
            if loss < 1e-6:
                print(f"Convergência atingida na Epoch {i}.")
                break
                
            gradientes = self.backward(Y_treino, cache)
            self.atualizar_pesos(gradientes, taxa_aprendizado)
            
        return historico_loss

    def prever(self, X):
        Y_previsto, _ = self.forward(X)
        return Y_previsto