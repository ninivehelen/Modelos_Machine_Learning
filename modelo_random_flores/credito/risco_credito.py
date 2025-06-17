import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree
# Abrindo o arquivo que tem a base separada 
with open('pre_processamento_dados/credito/credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_test, y_credit_test = pickle.load(f)

print(X_credit_treinamento.shape, y_credit_treinamento.shape)
print(X_credit_test.shape, y_credit_test.shape)

# Preparando o modelo para treinar
arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state = 0)
# treinando com previsoes e classe 
arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)

# Prevendo sobre o previsoes de treinamento X 
# 1 não pagou, 0 pagou  o emprestimo
previsoes = arvore_credit.predict(X_credit_test)
print("Previsoes", previsoes)
print("Dados real", y_credit_test)

# Gerando as metricas
# verificando acuracy, total de acerto para essa base.
print(accuracy_score(y_credit_test, previsoes))

# Matriz de confusão 
cm = ConfusionMatrix(arvore_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
arvore = cm.score(X_credit_test, y_credit_test)

print(confusion_matrix(y_credit_test, previsoes))

previsoes = ['income', 'age', 'loan']
fig, axes = plt.subplots(nrows= 1, ncols = 1, figsize = (20, 20))
tree.plot_tree(arvore_credit, feature_names=previsoes, class_names=['0', '1'], filled= True);
fig.savefig('modelo_random_flores/credito/modearvore_credit.png')

