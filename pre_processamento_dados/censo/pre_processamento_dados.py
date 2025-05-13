import pandas as pd 
import numpy as np

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# LabelEncoder
LabelEncoder_teste = LabelEncoder()

base_censo = pd.read_csv('pre_processamento_dados/censo/census.csv' , sep = ",")
print(base_censo.head(10))

# Visualização de estatistica 
print(base_censo.describe())

# Verificando se existe valores vazios
print(base_censo.isnull().sum())

#Verificando quantidade de income unicos 
print(np.unique(base_censo['income'], return_counts = True))

# grafico_quant_income = sns.countplot(x = base_censo['income'])
# grafico_quant_total = grafico_quant_income.get_figure()
# grafico_quant_total.savefig("quant_total_income.png")

# grafic_censo_age_hist = plt.hist(x = base_censo['age'])
# grafico_censo_age_ = grafic_censo_age_hist.get_figure()
# grafico_censo_age_.savefig("grafic_censo_age_hist.png")

# Gráfico interativo
# grafico = px.treemap(base_censo, path= ['workclass','age'])
# # print(grafico.show())

# grafico = px.treemap(base_censo, path= ['occupation','relationship', 'age'])
# print(grafico.show())

# grafico = px.parallel_categories(base_censo, dimensions= ['occupation','relationship','sex'])
# print(grafico.show())

# Divisão de divisores e previsores e classe

# Classe 

X_census = base_censo.iloc[:,0:14].values
# print(X_census)

Y_census = base_censo.iloc[:, 14].values
# print(Y_census)

# Tratamento de atributos categóricos
# print(X_census[:,1])

# Dados categoricos precisa esta em númericos
# Isso foi apenas um teste, esse teste foi na coluna 1
# teste = LabelEncoder_teste.fit_transform(X_census[:,1])
# print(teste)

# Primeiro aplica o lader enconder e depois o One Hoter coder, esse é mais usado na literatura

# Transformando todas as colunas categoricas em numero, pois o modelo trabalha melhor com númerico
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

# Aplicando na base 
X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_education.fit_transform(X_census[:,3])
X_census[:,5] = label_encoder_marital.fit_transform(X_census[:,5])
X_census[:,6] = label_encoder_occupation.fit_transform(X_census[:,6])
X_census[:,7] = label_encoder_relationship.fit_transform(X_census[:,7])
X_census[:,8] = label_encoder_race.fit_transform(X_census[:,8])
X_census[:,9] = label_encoder_sex.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_country.fit_transform(X_census[:,13])

# Agora todas as variaveis categoricas estão como númericas
# print(X_census[0])
# 

# Maneira mais utilizada na literatura 
# Complemento do oneHotEncoder para que o indice não valer mais nos calculos de machine learning ( Passando as colunas que os atributos são categorias)
onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
X_census = onehotencoder_census.fit_transform(X_census).toarray()
print(X_census[0])









