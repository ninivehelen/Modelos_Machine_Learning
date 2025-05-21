import pandas as pd 
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

base_rico_credito = pd.read_csv('naive_bayes/risco_credito.csv', sep = ',')
# print(base_rico_credito.head())

# Separar as variaveis previsores e classe 

# Previsores variavel X historia,divida,garantias,renda
X_risco_credito = base_rico_credito.iloc[:, 0:4].values
print('valores previsores', X_risco_credito)

# Transformar valores númericos em numero
label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantias = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:,0] = label_encoder_historia.fit_transform(X_risco_credito[:,0])
X_risco_credito[:,1] = label_encoder_divida.fit_transform(X_risco_credito[:,1])
X_risco_credito[:,2] = label_encoder_garantias.fit_transform(X_risco_credito[:,2])
X_risco_credito[:,3] = label_encoder_renda.fit_transform(X_risco_credito[:,3])

print(X_risco_credito[:,0])
print(X_risco_credito[:,1])
print(X_risco_credito[:,2])
print(X_risco_credito[:,3])


# Variavel classe Y risco
y_risco_credito = base_rico_credito.iloc[:, 4].values
print('variavel classe', y_risco_credito)
#   

# Salvando o que foi realizado acima 
with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito, y_risco_credito], f)


# Algoritmo Naive Bayes  GaussianNB usado para problemas mais genericos

naive_risco_credito = GaussianNB()
print(naive_risco_credito.fit(X_risco_credito, y_risco_credito))

previsao = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])

# Resultado da previsão 
print('valores de precisão', previsao)

