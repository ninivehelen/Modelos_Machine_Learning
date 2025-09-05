import pandas as pd 
import random

# Função para criar os dados negativos de acidentes.
def criar_dados_negativos(diretorio, base_dados):
    base_dados_negativos = []
    base_dados["acidente"] = 1
    # Garantir que 'acidente' está como 1 nos dados positivos

    # Corrigir e limpar latitude
    base_dados["latitude"] = base_dados["latitude"].astype(str).str.replace(',', '.')
    base_dados["latitude"] = pd.to_numeric(base_dados["latitude"], errors='coerce')
    
    # Corrigir e limpar longitude
    base_dados["longitude"] = base_dados["longitude"].astype(str).str.replace(',', '.')
    base_dados["longitude"] = pd.to_numeric(base_dados["longitude"], errors='coerce')
    
    # Corrigir e limpar km
    base_dados["km"] = base_dados["km"].astype(str).str.replace(',', '.')
    base_dados["km"] = pd.to_numeric(base_dados["km"], errors='coerce')
    
    # Remover linhas com latitude, longitude ou km inválidos (NaN)
    base_dados = base_dados.dropna(subset=["latitude", "longitude", "km"])
  
    print("Criando dados negativos com base nos positivos de acidentes")
    
    for _, linha in base_dados.iterrows():
        linha_diferente = linha.copy()
        
        # Deslocamento decimal no km (float)
        linha_diferente["km"] = max(0, linha_diferente["km"] + random.uniform(-30, 30))
        
        # Pequeno deslocamento nas coordenadas geográficas
        linha_diferente["latitude"] = linha_diferente["latitude"] + random.uniform(-0.01, 0.01)
        linha_diferente["longitude"] = linha_diferente["longitude"] + random.uniform(-0.01, 0.01)
        
        # Zerar vítimas (não existe acidente nesses dados negativos)
        linha_diferente["mortos"] = 0
        linha_diferente["feridos_leves"] = 0
        linha_diferente["feridos_graves"] = 0
        linha_diferente["feridos"] = 0
        linha_diferente["ilesos"] = 0
        
        # Gerar pessoas e veículos aleatoriamente para esses casos
        linha_diferente["pessoas"] = random.randint(1, 5)
        linha_diferente["veiculos"] = random.randint(1, 3)
        
        # Marcar como dado negativo (sem acidente)
        linha_diferente["acidente"] = 0
        
        base_dados_negativos.append(linha_diferente)
    print("Dados negativos criados")
    df_negativos = pd.DataFrame(base_dados_negativos)
    base_dados = base_dados[
        (base_dados["latitude"].between(-34, 5)) &
        (base_dados["longitude"].between(-74, -34)) &
        (base_dados["km"].between(0, 2000))
    ].reset_index(drop=True)
    
    # Salvar CSV dos acidentes onde não aconteceu
    df_negativos.to_csv(diretorio + '/detran_acidentes_negativos.csv', index=False)
    return(df_negativos)

def unir_dados_negativos_dados_positivos(base_dados, base_dados_negativos):
    print("Unindo dados de acidentes com dados criados negativos de acidentes")
    base_dados_positivos_negativos  = pd.concat([base_dados, base_dados_negativos ], ignore_index=True)
    base_dados_positivos_negativos.to_csv(diretorio+'/detran_acidentes_positivos_negativos_unido.csv', index=False)
    print("Unido e salvo")
    print(base_dados_positivos_negativos)

if __name__ == "__main__":
    diretorio = 'dados_acidentes'
    base_dados = pd.read_csv(diretorio+'/detran_acidentes_unido.csv')
    base_dados_negativos = criar_dados_negativos(diretorio, base_dados)
    unir_dados_negativos_dados_positivos(base_dados, base_dados_negativos)
    