import pandas as pd 
diretorio = 'dados_acidentes'

# Função para tratar os dados    
def tratamento_dados(base_dados_positivos_negativos):
    print(base_dados_positivos_negativos)
    print("Quantidade de colunas antes de remover as desnecessárias")
    print(base_dados_positivos_negativos.shape[1])
    print("Quantidade de linhas antes da limpeza")
    print(base_dados_positivos_negativos.shape[0])
    print("---------------------------------------")

    # Existe colunas que não serão necessárias para realizar o treinamento do machine learning de previsão de risco de acidente
    #Lista de colunas não necessárias = id, classificação_acidente, uso_solo, 
    # pessoas, mortos, feridos_leves, feridos_graves, ilesos, ignorados
    # feridos, veiculo, latitude, longitude, delegacia, uop, veiculos.

    base_dados_positivos_negativos.drop(columns=["data_inversa", "causa_acidente", "tipo_acidente", "classificacao_acidente",  "pessoas", "mortos", "feridos_leves", "feridos_graves", "ilesos", "ignorados", "feridos", "veiculos", "regional", "delegacia", "uop"], inplace= True)
    print("Quantidade de colunas após remover as desnecessárias")
    print(base_dados_positivos_negativos.shape[1])
    print("Quantidade de linhas depois da limpeza")
    print(base_dados_positivos_negativos.shape[0])
    print("---------------------------------------")

    # Ao final ficaram 14 colunas, antes havia 30 colunas.
# Etapa para analisar os dados e ver se existe descrepanças, dados faltantes, pois essa etapa é importate.
# --------------------------------------------------------------------------------------------#
# Analisando dados faltantes
    quant_faltantes = base_dados_positivos_negativos.isnull().sum()
    print("Quantidade de dados faltantes") 
    print(quant_faltantes)
# Analisando dados duplicados
    duplicados_id = base_dados_positivos_negativos['id'].duplicated().sum()
    print("verificando duplicadas na coluna ID")
    # Como foi criado dados "negativos onde não teve acidentes" utilizando dados de acidentes, normal ID está remetido, pois não foi criado de forma aleatória o ID
    print(duplicados_id)
    print("---------------------------------------")
# Analisando um pouco os dados 
    print("vendo a descrição dos dados")
    print(base_dados_positivos_negativos.describe())
   
if __name__ == "__main__":
    base_dados_positivos_negativos = pd.read_csv(diretorio+'/detran_acidentes_positivos_negativos_unido.csv')
    tratamento_dados(base_dados_positivos_negativos)