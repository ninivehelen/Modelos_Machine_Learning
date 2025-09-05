import glob
import os 
import pandas as pd 
diretorio = 'dados_acidentes_prf'

def unir_arquivos():
    arquivos_csv = glob.glob(os.path.join(diretorio+ "/datatran*.csv")) 
    print(arquivos_csv)
    base_dados = pd.DataFrame()
    for arquivo in arquivos_csv:
        df_temp = pd.read_csv(arquivo, sep=';' ,encoding='latin-1')
        base_dados  = pd.concat([df_temp,base_dados])
    base_dados.to_csv(diretorio+'/detran_acidentes_unido.csv', index=False)
    print("CSV com todas os csv´s de acidentes unidos")

if __name__ == "__main__":
    unir_arquivos()





