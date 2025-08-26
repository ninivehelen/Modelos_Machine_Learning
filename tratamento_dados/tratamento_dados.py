import pandas as pd 
import pickle 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

diretorio = 'dados_acidentes_prf'

# Função para tratar os dados    
def analisa_dados(base_dados_positivos_negativos):
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
    tratamento_dados(base_dados_positivos_negativos)


def tratamento_dados(df_acidente):
    df_filtra_acidente = df_acidente.sample(n=10000, random_state=42)
    # Divisão da base de dados, previsores e a classe 
    print('colunas', df_filtra_acidente.columns)
    X_acidente = df_filtra_acidente.iloc[:, [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]].values
    Y_acidente = df_filtra_acidente.iloc[:, 15].values
    print("Visualizando variáveis previsoras")
    print(X_acidente[0])
    print(X_acidente.shape)
    # Transformar variáveis categóricas em númericas
    Label_encoder_dia_semana = LabelEncoder()
    Label_encoder_horario = LabelEncoder()
    # Label_encoder_uf = LabelEncoder()
    # Label_encoder_municipio = LabelEncoder()
    Label_encoder_fase_dia = LabelEncoder()	
    Label_encoder_sentido_via = LabelEncoder()	
    Label_encoder_condicao_metereologica = LabelEncoder()
    Label_encoder_tipo_pista = LabelEncoder()
    Label_encoder_tracado_via = LabelEncoder()
    Label_encoder_uso_solo = LabelEncoder()
    # convertendo variavel categorica para númerica
    X_acidente[:, 0] = Label_encoder_dia_semana.fit_transform(X_acidente[:,0])
    X_acidente[:, 1] = Label_encoder_horario.fit_transform(X_acidente[:,1])
    # X_acidente[:, 2] = Label_encoder_uf.fit_transform(X_acidente[:,2])
    # X_acidente[:, 5] = Label_encoder_municipio.fit_transform(X_acidente[:,5])
    X_acidente[:, 4] = Label_encoder_fase_dia.fit_transform(X_acidente[:, 4])
    X_acidente[:, 5] = Label_encoder_sentido_via.fit_transform(X_acidente[:, 5])
    X_acidente[:, 6] = Label_encoder_condicao_metereologica.fit_transform(X_acidente[:, 6])
    X_acidente[:, 7] = Label_encoder_tipo_pista.fit_transform(X_acidente[:, 7])
    X_acidente[:, 8] = Label_encoder_tracado_via.fit_transform(X_acidente[:, 8])
    X_acidente[:, 9] = Label_encoder_uso_solo.fit_transform(X_acidente[:, 9])
    
    print("Visualizando após conventer categórica para númerica")
    print(X_acidente[0])

    # É necessário utilizar um que complementa o label coder. OneHotEncoder
# Por que é importante usar um complemento ?  por que o indices da categorias influênciam, nos calculos. o menor indice por ser considerado menos importantes, mas são somente categórias diferentes.
    onehotencoder_acidente = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 1, 2, 5, 6, 7, 8, 9, 10, 11 ])] , remainder = 'passthrough') 
    X_acidente = onehotencoder_acidente.fit_transform(X_acidente).toarray()
    print("Aplicando OneHot")
    print(X_acidente.shape)
    print(X_acidente)

# Realizando o escalonamento dos dados, pois longitude e latitude é bem mais alto, apesar de usar random flores, preferir escalonar

    scaler_acidente  = StandardScaler()
    X_acidente = scaler_acidente.fit_transform(X_acidente)
    print("Após escalonamento")
    print(X_acidente)

# Divisão de treinamento e divisão de teste, pois é necessário fazer a divisão para aplicar o machine learning 
    
    X_treinamento_acidente, X_teste_acidente , Y_treinamento_acidente, Y_teste_acidente = train_test_split(X_acidente, Y_acidente, test_size = 0.25, random_state = 0)
    # Salvar a variavel, para que não precise ser executado novamente 
    with open(diretorio+'/acidente.pkl', mode = 'wb') as f:
         pickle.dump([X_treinamento_acidente, Y_treinamento_acidente, X_teste_acidente, Y_teste_acidente], f)
    print("acidente.pkl salvo")

if __name__ == "__main__":
    base_dados_positivos_negativos = pd.read_csv(diretorio+'/detran_acidentes_positivos_negativos_unido.csv')
    analisa_dados(base_dados_positivos_negativos)
    