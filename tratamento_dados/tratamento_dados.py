import pandas as pd 
diretorio = 'dados_acidentes'

base_dados_positivos_negativos = pd.read_csv(diretorio+'/detran_acidentes_positivos_negativos_unido.csv')

# Função para tratar os dados    
def tratamento_dados():
    print("Quantidade de colunas antes de remover as desnecessárias")
    base_dados_positivos_negativos ["acidente"] = 1
    print(base_dados_positivos_negativos .shape[1])
    # Existe colunas que não serão necessárias para realizar o treinamento do machine learning de previsão de risco de acidente
    #Lista de colunas não necessárias = id, classificação_acidente, uso_solo, 
    # pessoas, mortos, feridos_leves, feridos_graves, ilesos, ignorados
    # feridos, veiculo, latitude, longitude, delegacia, uop, veiculos.
    base_dados_positivos_negativos .drop(columns=['classificacao_acidente', 'uso_solo','pessoas', 'mortos', 'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos', 'latitude', 'longitude', 'regional', 'delegacia', 'uop'], inplace= True)
    print("Quantidade de colunas após remover as desnecessárias")
    print(base_dados_positivos_negativos .shape[1])
    # Ao final ficaram 14 colunas, antes havia 30 colunas.
# Etapa para analisar os dados e ver se existe descrepanças, dados faltantes, pois essa etapa é importate.
# --------------------------------------------------------------------------------------------#
# Analisando dados faltantes
    quant_faltantes = base_dados_positivos_negativos .isnull().sum()
    print(" Quantidade de dados faltantes", quant_faltantes)
    # Apenas a coluna tipo_acidente possui um valor null e é apenas 1 dado, como apenas 1 está vazio, pode ser removido.
    base_dados_positivos_negativos .dropna(subset=['tipo_acidente'], inplace=True)
    #Removido contabilizar novamente 
    quant_faltantes = base_dados_positivos_negativos .isnull().sum()
    print("quantidade de dados faltantes após a remoção", quant_faltantes)
    # Agora nenhuma coluna possúi dados faltantes.
# Analisando se existe dados duplicados pelo id, pois essa coluna deve ser única.
    duplicados_id = base_dados_positivos_negativos ['id'].duplicated().sum()
    print("verificando duplicadas na coluna ID:\n", duplicados_id)
    duplicados_id_colunas = base_dados_positivos_negativos .duplicated().sum()
    print("verificando duplicadas nos dados:\n", duplicados_id_colunas)
    # os dados não possui duplicados

if __name__ == "__main__":
    tratamento_dados()