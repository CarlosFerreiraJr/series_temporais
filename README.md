## Séries Temporárias em Python

```
from scipy import stats
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf 
import requests
import datetime as dt
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
plt.style.use('bmh')

url = "https://investnews.com.br/financas/veja-a-lista-completa-dos-bdrs-disponiveis-para-pessoas-fisicas-na-b3/"
r = requests.get(url)
html = r.text
df_nomes_tickers = pd.read_html(html, header=0)[0]

codigos_ativos = df_nomes_tickers['CÓDIGO']


for i, item in enumerate(codigos_ativos):
  if item == 'MSFT34':
    print("Posição do ticker msft34", i)
  
cod_microsoft = codigos_ativos[405]
cod_microsoft = [str(cod_microsoft+ '.SA')]


for i, item in enumerate(codigos_ativos):
  if item == 'TWTR34':
    print("Posição do ticker twtr34", i)

cod_tw = codigos_ativos[592]
cod_tw = [str(cod_tw+ '.SA')]

df_cod_microsoft = yf.download(cod_microsoft[0],
                 start = '2022-04-20',
                 end = '2022-10-20'
                 )

df_cod_tw = yf.download(cod_tw[0],
                 start = '2020-10-20',
                 end = '2022-10-20'
                 )
mpf.plot(df_cod_tw, type="candle", volume=True, style="yahoo")

df_cod_tw[['Open','High','Low','Close']].plot(title = str(cod_tw[0])+ " - Valor de fechamento")

df_cod_microsoft[['Open','High','Low','Close']].plot(title = str(cod_microsoft[0])+ " - Valor de fechamento")

mpf.plot(df_cod_microsoft, type="candle", volume=True, style="yahoo")

def analise_exploratoria(df):
  media = df.mean()
  desvio_padrao=df.std()
  variancia=df.var()
  minimo = df.min()
  maximo = df.max()
  moda = stats.mode(df)
  dados = [[media, desvio_padrao, variancia, minimo, maximo, moda.mode[0]]]
  df_dados = pd.DataFrame(dados, columns=['media', 'desvio_padrao', 'variancia', 'minimo', 'maximo', 'moda'])
  return df_dados
  
df_analise_ex_tw = analise_exploratoria(df_cod_tw['Close'])
df_analise_ex_tw.to_excel('analise_exploratoria_Twiter.xlsx')
df_analise_ex_tw

df_analise_ex_ms = analise_exploratoria(df_cod_microsoft['Close'])
df_analise_ex_ms.to_excel('analise_exploratoria_Microsoft.xlsx')
df_analise_ex_ms

df_cod_microsoft.isnull()
df_cod_microsoft.isnull().sum()

sujos_df_cod_microsoft = df_cod_microsoft['Volume'] == 0
sujos_df_cod_microsoft.sum()
df_cod_microsoft[sujos_df_cod_microsoft]
df_cod_microsoft = df_cod_microsoft[~sujos_df_cod_microsoft]

df_cod_microsoft.isnull()
df_cod_microsoft.isnull().sum()

sujos_df_cod_tw = df_cod_tw['Volume'] == 0
sujos_df_cod_tw.sum()
df_cod_tw[sujos_df_cod_tw]
df_cod_tw = df_cod_tw[~sujos_df_cod_tw]

def treino_teste(dados):

  treino = 0.75

  dia1 = min(dados.index)
  diafinal = max(dados.index)
  diastotais = (diafinal - dia1).days

  dia_treino = np.ceil(diastotais*treino)
  dia_teste = np.floor(diastotais*(1-treino))

  dia_da_parada = dia1 + timedelta(days=dia_treino)

  dados_para_treino = dados[:dia_da_parada]
  dados_para_teste = dados[dia_da_parada:]

  return dados_para_treino,dados_para_teste
  
  ### PREDIÇÕES PARA O ATIVO DO TWITTER 
  
  def avarage_forecasting(x,y):

      y_hat_AF = []
      for i in range(len(y)):
        
        y_hat_AF.append(np.mean(x))
            
      y_hat_AF = pd.Series(y_hat_AF, index=y.index)
      return y_hat_AF
      
x_df_cod_tw, y_df_cod_tw = treino_teste(df_cod_tw['Close'])

y_hat_AF_df_cod_tw = avarage_forecasting(x_df_cod_tw, y_df_cod_tw)

y_hat_AF_df_cod_tw = avarage_forecasting(x_df_cod_tw, y_df_cod_tw)

with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Average Forecast")
    

    plt.plot(x_df_cod_tw, label='Valores de treino')
    plt.plot(y_df_cod_tw, label='Valores reais')
    
   
    plt.plot(y_hat_AF_df_cod_tw, label='Average forecasting', color='red')
    
    plt.legend()
    plt.show()
    
    def DF(x, y):
    y_t = x[-1]
    m = (y_t - x[0]) / len(x)
    h = np.linspace(0,len(y.index)-1, len(y.index))
    
  
    y_hat_DF = []
    for i in range(len(y.index)):
        y_hat_DF.append(y_t + m * h[i])
    
    
    y_hat_DF = pd.Series(y_hat_DF, index=y.index)
    return y_hat_DF
    
    y_hat_DF_df_cod_tw = DF(x_df_cod_tw, y_df_cod_tw)
    
    with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Average Forecast e Drift Forecast")
    
    # Dados reais
    plt.plot(x_df_cod_tw, label='Valores de treino')
    plt.plot(y_df_cod_tw, label='Valores reais')
    
    # Predições
    plt.plot(y_hat_AF_df_cod_tw, label='Average Forecast', color='red')
    plt.plot(y_hat_DF_df_cod_tw, label='Drift Forecast', color='Yellow')
    
    plt.legend()
    plt.show()
    
    def SMA(dados, day):
    y_hat_SMA = dados['Close'].rolling(window=day).mean()
    return y_hat_SMA
    
    days = [75, 150, 300]
colors = ['green', 'blue', 'pink', 'purple']

with plt.style.context('dark_background'):
  
    plt.figure(figsize=(20, 5.5))
    plt.title("Combinação de técnicas")
    

    plt.plot(x_df_cod_tw, label='Valores de treino', color='white')
    plt.plot(y_df_cod_tw, label='Valores reais')
    

    for i, day in enumerate(days):
        y_hat_SMA_df_cod_tw = SMA(df_cod_tw, day)
        plt.plot(y_hat_SMA_df_cod_tw, label='Média móvel simples '+str(day), color=colors[i])
        
       
    plt.plot(y_hat_AF_df_cod_tw, label='Average Forecast', color='red')
    plt.plot(y_hat_DF_df_cod_tw, label='Drift Forecast', color='Yellow')
    
 
    plt.legend()
    plt.show()
    
    def CMA(dados, day):
    y_hat_CMA = dados['Close'].expanding(min_periods=day).mean()
    return y_hat_CMA
    
    y_hat_CMA_df_cod_tw = CMA(df_cod_tw, 5)
    
    days = [5, 10, 25, 50]
colors = ['green', 'blue', 'pink', 'purple']

     
with plt.style.context('dark_background'):
    
    plt.figure(figsize=(20, 5.5))
    plt.title("Cumulative Moving Average")
    
   
    plt.plot(x_df_cod_tw, label='Valores de treino', color='white')
    plt.plot(y_df_cod_tw, label='Valores reais')
    
    # Moving average
    plt.plot(y_hat_CMA_df_cod_tw, label='CMA', color='green')
    
    # Previsões
    plt.plot(y_hat_AF_df_cod_tw, label='Average Forecast', color='red')
    plt.plot(y_hat_DF_df_cod_tw, label='Drift Forecast', color='Yellow')

    # Configurações básicas
    plt.legend()
    plt.show()
    
    def avarage_forecasting(x,y):

      y_hat_AF = []
      for i in range(len(y)):
        
        y_hat_AF.append(np.mean(x))
            
      y_hat_AF = pd.Series(y_hat_AF, index=y.index)
      return y_hat_AF



x_df_cod_microsoft, y_df_cod_microsoft = treino_teste(df_cod_microsoft['Close'])

y_hat_AF_df_cod_microsoft = avarage_forecasting(x_df_cod_microsoft, y_df_cod_microsoft)

y_hat_AF_df_cod_microsoft = avarage_forecasting(x_df_cod_microsoft, y_df_cod_microsoft)

with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Average Forecast")
    

    plt.plot(x_df_cod_microsoft, label='Valores de treino')
    plt.plot(y_df_cod_microsoft, label='Valores reais')
    
   
    plt.plot(y_hat_AF_df_cod_microsoft, label='Average forecasting', color='red')
    
    plt.legend()
    plt.show()
    
    def DF(x, y):
    y_t = x[-1]
    m = (y_t - x[0]) / len(x)
    h = np.linspace(0,len(y.index)-1, len(y.index))
    
  
    y_hat_DF = []
    for i in range(len(y.index)):
        y_hat_DF.append(y_t + m * h[i])
    
    
    y_hat_DF = pd.Series(y_hat_DF, index=y.index)
    return y_hat_DF

y_hat_DF_df_cod_microsoft = DF(x_df_cod_microsoft, y_df_cod_microsoft)

with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Average Forecast e Drift Forecast")
    
    # Dados reais
    plt.plot(x_df_cod_microsoft, label='Valores de treino')
    plt.plot(y_df_cod_microsoft, label='Valores reais')
    
    # Predições
    plt.plot(y_hat_AF_df_cod_microsoft, label='Average Forecast', color='red')
    plt.plot(y_hat_DF_df_cod_microsoft, label='Drift Forecast', color='Yellow')
    
    plt.legend()
    plt.show()
    
    def SMA(dados, day):
    y_hat_SMA = dados['Close'].rolling(window=day).mean()
    return y_hat_SMA


days = [30, 45, 90]
colors = ['green', 'blue', 'pink', 'purple']

with plt.style.context('dark_background'):
  
    plt.figure(figsize=(20, 5.5))
    plt.title("Simple Moving Average")
    

    plt.plot(x_df_cod_microsoft, label='Valores de treino', color='white')
    plt.plot(y_df_cod_microsoft, label='Valores reais')
    

    for i, day in enumerate(days):
        y_hat_SMA_df_cod_microsoft = SMA(df_cod_microsoft, day)
        plt.plot(y_hat_SMA_df_cod_microsoft, label='Média móvel simples '+str(day), color=colors[i])
        
       
    plt.plot(y_hat_AF_df_cod_microsoft, label='Average Forecast', color='red')
    plt.plot(y_hat_DF_df_cod_microsoft, label='Drift Forecast', color='Yellow')
      plt.legend()
    plt.show()    

```
