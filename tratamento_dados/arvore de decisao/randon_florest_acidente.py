import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
diretorio = 'dados_acidentes_prf'

# Aplicando random florest 

#Abrindo arquivo
with open(diretorio+'/acidente.pkl', 'rb') as f:
    X_treinamento_acidente, Y_treinamento_acidente, X_teste_acidente, Y_teste_acidente = pickle.load(f)

print("Dados de treinamento")
print(X_treinamento_acidente.shape, Y_treinamento_acidente.shape)

print("Dados de teste")
print(X_teste_acidente.shape, Y_teste_acidente.shape)

# Aplicando random flores 
print("Aplicando Random Florest")
random_forest_acidente = RandomForestClassifier(n_estimators= 80, criterion= 'entropy', random_state = 0)
random_forest_acidente.fit(X_treinamento_acidente, Y_treinamento_acidente)

previsoes = random_forest_acidente.predict(X_teste_acidente)
print("Previsoes")
print(previsoes)
print(" Dados com previsões corretas")
print(Y_teste_acidente)

# Aplicando metricas para saber o score do algoritmo 
print("Vendo accuracy do modelo")
accuracy = accuracy_score(Y_teste_acidente, previsoes)
print("Accuracy do modelo", accuracy)

# Visualizar a matriz de confusão
cm = confusion_matrix(Y_teste_acidente, previsoes)
print("Matriz de confusão")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()