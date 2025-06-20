import pandas as pd 
import numpy as np 
import random
diretorio = 'dados_acidentes'
base_dados_negativos = []

base_dados = pd.read_csv(diretorio+'/detran_acidentes_unido.csv')

import random
import pandas as pd

# Função para criar os dados negativos de acidentes.
def criar_dados_negativos():
    # marca como 1 os que tem acidentes
    base_dados["acidente"] = 1
    
    # mudando o tipo pois facilita para criar novos
    base_dados["latitude"] = base_dados["latitude"].str.replace(',', '.').astype(float)
    base_dados["longitude"] = base_dados["longitude"].str.replace(',', '.').astype(float)
    base_dados["km"] = base_dados["km"].str.replace(',', '.').astype(float)

    base_dados_negativos = []  # lista para os dados negativos
    print("Criando dados negativos com base nos positivos de acidentes")
    # Pecorrendo as linhas do data frame positivos para criar os negativos com bases nas linhas
    # Porém alterando km longitude e latitude, outras colunas para criar os dados negativos
    # Esses dados vão ser importantes para criar o modelo de machine learning. 
    for _, linha in base_dados.iterrows():
        linha_diferente = linha.copy()

        linha_diferente["km"] = max(0, linha_diferente["km"] + random.randint(-30, 30))
        linha_diferente["latitude"] = linha_diferente["latitude"] + random.uniform(-0.01, 0.01)
        linha_diferente["longitude"] = linha_diferente["longitude"] + random.uniform(-0.01, 0.01)

        linha_diferente["mortos"] = 0
        linha_diferente["feridos_leves"] = 0
        linha_diferente["feridos_graves"] = 0
        linha_diferente["feridos"] = 0
        linha_diferente["ilesos"] = 0
        linha_diferente["pessoas"] = random.randint(1, 5)
        linha_diferente["veiculos"] = random.randint(1, 3)

        linha_diferente["acidente"] = 0  # marca como dado sem acidente
       
        # Adcionando a lista 
        base_dados_negativos.append(linha_diferente)
    print("Dados negativos criados")
    # Salvando o csv com os dados negativos, é bem grande a base.
    df_negativos = pd.DataFrame(base_dados_negativos)
    df_negativos.to_csv(diretorio+'/detran_acidentes_negativos.csv', index=False)
    print(df_negativos['acidente'])
    print(df_negativos)

def unir_dados_negativos_dados_positivos():
    base_dados_negativos = pd.read_csv(diretorio+'/detran_acidentes_negativos.csv')
    base_dados_positivos_negativos  = pd.concat([base_dados, base_dados_negativos ], ignore_index=True)
    base_dados_positivos_negativos.to_csv(diretorio+'/detran_acidentes_positivos_negativos_unido.csv', index=False)
    print("Unidos e salvo", base_dados_positivos_negativos)

    
if __name__ == "__main__":
    criar_dados_negativos()
    unir_dados_negativos_dados_positivos()
    #tratamentodados
    

