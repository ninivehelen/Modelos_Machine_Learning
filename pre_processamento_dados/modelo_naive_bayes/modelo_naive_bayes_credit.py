import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Abrindo o arquivo que tem a base treinada
with open('pre_processamento_dados/credito/credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_test, y_credit_test = pickle.load(f)


print(X_credit_treinamento.shape)
print(y_credit_treinamento.shape)

print(X_credit_test.shape)
print(y_credit_test.shape)

# Treinando o modelo 
naive_credit_data = GaussianNB()
print(naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento))

# Agora passando o de teste que o modelo não conhece
previsores = naive_credit_data.predict(X_credit_test)
print("previsoes", previsores)

# Vericar com as resposta que temos 
print("respostas que temos", y_credit_test)

# Comparar registro por registo para saber se o algoritmo acertou
print("Resultado de acerto ", accuracy_score(y_credit_test, previsores))

print(confusion_matrix(y_credit_test, previsores))




