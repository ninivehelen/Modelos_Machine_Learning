import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Abrindo o arquivo que tem a base treinada
with open('pre_processamento_dados/censo/census.pkl', 'rb') as f:
    X_censu_treinamento, y_censu_treinamento, X_censu_test, y_censu_test = pickle.load(f)

print(X_censu_treinamento.shape)
print(y_censu_treinamento.shape)

print(X_censu_test.shape)
print(y_censu_test.shape)

naive_census = GaussianNB()
naive_census.fit(X_censu_treinamento, y_censu_treinamento)
previsoes = naive_census.predict(X_censu_test)

print("Previsões")
print(previsoes)

print("Conferindo")
print(y_censu_test)

# Canculando a acuracy 
# Taxa de acerto baixo 
print("Taxa de acerto", accuracy_score(y_censu_test, previsoes))

#
print(classification_report(y_censu_test, previsoes))


# cm = confusion_matrix(naive_census)
# cm.fit(X_censu_treinamento, y_censu_treinamento)
# print(cm.score(X_censu_test, y_censu_test))