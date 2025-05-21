import pandas as pd 
import matplotlib as plt
import seaborn as sns
import numpy as np
import plotly.express as px

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Etapa de exploração de dados
# Dicionário de dados client id = id do cliente, income = renda da pessoa, age = idade da pessoa, load = divída que a pessoa possuí, default = se pagou 0 caso contrario 1. 

#abrindo a base de dados 
base_credit = pd.read_csv("pre_processamento_dados/credito/credit_data.csv", encoding = 'utf8', sep=',')
#print(base_credit)

#visualizar os primeiros 10 registros
#print(base_credit.head(10))

# # descrição dos dados
# print(base_credit.describe())

# #Pessoa com a maior renda.
# print(base_credit[base_credit['income'] >= 69995.685578])

# #Pessoa com menor renda
# print(base_credit[base_credit['income'] >= 20014.489470])

# #Pessoa com menor divida
# print(base_credit[base_credit['loan'] <= 1.377630])

# contar quantos registro existe na base default 
print(np.unique((base_credit['default']), return_counts = True))

# #Gráfico total se pagou ou não
# grafico_se_pagou = sns.countplot(x= base_credit['default'])
# grafico_se_pagou_quant = grafico_se_pagou.get_figure()
# grafico_se_pagou_quant.savefig("grafico_pagou_nao_pagou.png")

# # Gráfico histograma idade
# grafico_hist_age = sns.histplot(x= base_credit['age'])
# grafico_hist_age_quant = grafico_hist_age.get_figure()
# grafico_hist_age_quant .savefig("histograma_idade.png")

# Gráfico histograma idade
# grafico_hist_income = sns.histplot(x= base_credit['income'])
# grafico_hist_income_quant = grafico_hist_income.get_figure()
# grafico_hist_income_quant .savefig("histograma_income.png")

# grafico_hist_loan= sns.histplot(x= base_credit['loan'])
# grafico_hist_loan_quant = grafico_hist_loan.get_figure()
# grafico_hist_loan_quant .savefig("histograma_loan.png")

#para saber quem paga e quem não paga emprestimo.  0 paga 1 não paga 
# pessoas_pagam_nao_pagam_imprestimo = np.unique(base_credit['default'], return_counts = True)

# total_pagam_nao_pagam = sns.countplot(x = base_credit['default']);
# grafico_quant_total = total_pagam_nao_pagam.get_figure()
# grafico_quant_total.savefig("quant_total_paga_nao.png")

# grafico = px.scatter_matrix(base_credit, dimensions = ['age', 'income', 'loan'], color = 'default')
# grafico.show()

# Tratamento de valores negativos na idade 
# print(base_credit.loc[base_credit['age'] < 0])
# print(base_credit[base_credit['age'] < 0])

# #primeira opção apagar a coluna idade
# base_creddit_2 = base_credit.drop('age', axis = 1)
# print(base_creddit_2)

# # apagar somente os registros que tem idade negativa
# base_creddit_3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
# print(base_creddit_3) # ou pedir os dados faltantes para os clientes.

# ou preencher os valores faltantes com a média das idades.

#Tratamento da idade com valores negativos adicionando a média. 
print(base_credit['age'].mean())

print(base_credit['age'][base_credit['age']> 0].mean())

base_credit.loc[base_credit['age'] < 0 , 'age' ] = 40.92

print(base_credit.loc[base_credit['age']< 0])

print(base_credit.head(27))

# Tratamento de valores faltantes
print(base_credit.isnull().sum())

#Apenas a idade tem dados faltantes. 
print(base_credit.loc[pd.isnull(base_credit['age'])])
# Passa a média para substituir valores vazios.
base_credit['age'].fillna(base_credit['age'].mean(), inplace= True)
# Tratamento de valores faltantes
print(base_credit.isnull().sum())
print(base_credit.loc[(base_credit['clientid']== 29)])
print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])

# selecionando previsores e classe 
X_credit = base_credit.iloc[:, 1:4].values
Y_credit = base_credit.iloc[:, 4].values

# Escalonamento de dados, pois a idade e renda e divida são bem diferente entre eles
print(X_credit[:, 0].min())
print(X_credit[:, 0].max())

# padronizando os dados 

#Colocando na mesma escala 
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

#imprimindo o escalonamento realizado 
print(X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min())
print(X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max())

# Divisão da base de treinamento e teste 

X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit, Y_credit, test_size = 0.25, random_state = 0)

print(X_credit_treinamento.shape)
print(y_credit_treinamento.shape)

#Salvando as alterações para nao precisar rodar novamente

with open('pre_processamento_dados/credito/credit.pkl', mode = 'wb') as f:
     pickle.dump([X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste], f)

