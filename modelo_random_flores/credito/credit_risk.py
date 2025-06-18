import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib as plt 
with open('pre_processamento_dados/credito/risco_credito.pkl', 'rb') as f:
    X_risco_credit, y_risco_credit =  pickle.load(f)

print(X_risco_credit)
print(y_risco_credit)

# Criando a arvore de decisão para prever risco de credito neste modelo

# Fazendo o treinamento utilizando calculo entropy
arvore_risco_credito = DecisionTreeClassifier(criterion = 'entropy')
# Treinamnto
arvore_risco_credito.fit(X_risco_credit, y_risco_credit)

# Mostrando resultado do calculo para saber qual mais impotante, no resultado a renda é o mais importante para a previsão
print(arvore_risco_credito.feature_importances_)
# Para visualizar a arvore 
previsores = ['historia_credito', 'dívida', 'garantias', 'renda']
#figura, eixos = plt.subplots(nrows= 1, ncols=1, figsize = (10,10))
print(tree.plot_tree(arvore_risco_credito, feature_names= previsores, class_names= arvore_risco_credito.classes_, filled= True));
#print(tree.plot_tree(arvore_risco_credito))

#historia boa, dívida alta, garantias nehuma, renda > 35
#historia ruim, dívida alta, garantias adequadas, renda < 15

previsoes_novos_dados = arvore_risco_credito.predict([[0,0,1,2],[2,0,0,0]])
print(previsoes_novos_dados)