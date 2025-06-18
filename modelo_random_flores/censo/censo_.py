import pickle
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree
import matplotlib as plt 

with open("pre_processamento_dados/censo/census.pkl", "rb") as f:
    X_census_treinamento, y_census_treinamento, X_census_test, y_census_test = pickle.load(f)

# Abrindo para visualizar as variaveis.
print(" census x treinamento", X_census_treinamento.shape)
print(" census y treinamento", y_census_treinamento.shape)

print(" census x treinamento", X_census_test.shape)
print(" census y treinamento", X_census_test.shape)

# Realizando o treinamento.
arvore_census = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
arvore_census.fit(X_census_treinamento, y_census_treinamento)

# Realizando as previsões no dados de teste, após o treinamento com os dados de treinamento.
previsoes = arvore_census.predict(X_census_test)

# conferindo as previsoes com o dados de teste e coma resposta real
print(" previsões com os dados de teste", previsoes)
print(" previsões reais", y_census_test)

# Vendo a acurracy do modelo 
cm = ConfusionMatrix(arvore_census)
cm.fit(X_census_treinamento, y_census_treinamento)
print(cm.score(X_census_test, y_census_test))

# Recall é quanto que identifica,  presicion é a precisão quando o encontra. 
print(classification_report(y_census_test, previsoes))
