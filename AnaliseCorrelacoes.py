import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import pstdev
import statsmodels.api as sm
import seaborn as sns


#Trazendo as bases.

Sectors = pd.read_csv("Setores.csv",delimiter = ";",index_col = "Codigo")
Fech = pd.read_csv("Fechamento.csv",delimiter = ";",index_col="Data",decimal = ".")

#Convertendo vírgula em ponto e tranformando em float

for i in Fech.columns:
  Fech[i]=Fech[i].str.replace(',','.')
  Fech[i].replace("-", np.nan, inplace=True)
  Fech[i] = Fech[i].astype(float)

Fech.drop(columns= ["BIDI3","BIDI4"],inplace=True)
# Criando o dataframe com os log retornos diários das ações
df_logretorno = np.log(Fech/Fech.shift(1))
#df_logretorno

df_logretorno.replace( np.nan,0, inplace=True)


df_logretorno.index = pd.to_datetime(df_logretorno.index,infer_datetime_format=True)


RetSetorial = pd.DataFrame(index = df_logretorno.index, columns = list(set(Sectors['Setor'])))
for setor in RetSetorial.columns:
    lista = Sectors[Sectors['Setor']==setor].index #lista das ações por setor
    dfTemp = pd.DataFrame(index = df_logretorno.index,columns = lista)    #retornos por ação
    for asset in dfTemp.columns:
        try:
            dfTemp[asset] = df_logretorno[asset]  # Achar a coluna de retornos parae esse ativo
        except:
            continue
    dfTemp['Total'] = dfTemp.mean(axis=1)  # coluna total é a média de cada um dos ativos.
    RetSetorial[setor] = dfTemp['Total']



"""-----------Construindo um dataframe de retornos diários por setor---------------------"""
correlacao = RetSetorial.corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(correlacao, annot=False, linewidths=.5, ax=ax)

print(['Papel e Celulose'])
RetSetorial['Papel e Celulose'].cumsum().plot()
RetSetorial['Máquinas Indust'].cumsum().plot()

plt.legend(['Papel e Celulose','Máquinas Indust'])

ts_corr = RetSetorial.rolling(window = 36).corr()
ts_corr.index.names=['Date','Industry']
print(ts_corr[2520:4000])



#######Calculando as médias anuais de correlação de um setor com todos os outros##################3


periodos = [["2011","2012"],["2012","2013"],["2013","2014"],
            ["2014","2015"],["2015","2016"],["2016","2017"],["2017","2018"],
            ["2018","2019"],["2019","2020"],["2020","2021"],["2021",""]]


dfCorr = pd.DataFrame(index = [periodos[i][0] for i in range(0,11)], columns = ts_corr.columns)

for sector in dfCorr.columns:
  for period in periodos:
    if period[0] != "2021":
      dfCorr[sector][[period][0]]=ts_corr[period[0]:period[1]].groupby(level="Industry").apply(lambda cormat: cormat.values.mean())[sector]
    else:
      dfCorr[sector]["2021"]=ts_corr["2021":].groupby(level="Industry").apply(lambda cormat: cormat.values.mean())[sector]

print(dfCorr)

dfCorr.plot(legend = True,figsize =(14,10))

plt.bar(dfCorr.index,dfCorr['Construção'],color='red')

################Correlação de um setor em específico com outro##################333

setor1 = 'Comércio'
setor2 = 'Papel e Celulose'

CorBet = pd.DataFrame(index = [periodos[i][0] for i in range(1,11)], columns =['relação'])
for date in periodos[1:]:
  CorBet['relação'][date[0]] = ts_corr[setor1][date[0]][setor2].mean()

plt.bar(CorBet.index,CorBet['relação'])

#------------------------------------------------------------------------

"""
Análise de ações com beta elevado.
"""

#Pegando retornos do Ibovespa
import yfinance as yf
def catch_data(code, start_date, end_date):
    # get data on this ticker
    tickerData = yf.Ticker(code)

    # get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

    return tickerDf


ibov= catch_data('BOVA11.SA', "2010-06-05", "2021-12-31")['Close']
ibov = pd.DataFrame(ibov)
ibov.dropna()
ibov['return'] = np.log(ibov['Close']/ibov['Close'].shift(1))
ibov['return'].iloc[0] = 0

# Maiores quedas do iBovespa:
# Criando um dataframe com o beta de todas as ações nesses períodos.
"""
1ª) 2012-05-13  até 2012-06-05   ****23%
2ª) 2013-01-14  até 2013-06-10   ****27%
3ª) 2015-04-17  até 2016-01-27   ****33%
4ª) 2021-06-07  até 2021-11-25   ****21,7%
5ª) 2022-05-31  até 2022-07-14   ****19%
"""

periodos = [
    ["2012-05-13","2012-06-05"],["2013-01-14","2013-06-10"],
    ["2015-04-17","2016-01-27"],["2021-06-07","2021-11-25"],

]

betas = pd.DataFrame(index = ["2012-05-13","2013-01-14","2015-04-17","2021-06-07"],columns = df_logretorno.columns)





var = 0
cov=0

new_logretorno = ibov

for i in df_logretorno.columns:
    new_logretorno[i] = ''

#tentando preencher um dataframe com os mesmos dias do ibov.(correção de erro de dimensão)
for ativos in df_logretorno.columns:
    for dia in new_logretorno.index:
        try:
            new_logretorno[ativos][dia] = df_logretorno.loc[dia][ativos]
        except:
            new_logretorno[ativos][dia] = 0





for choque in periodos:
        ibovs=ibov.loc[choque[0]:choque[1]]["return"]
        var = np.var(ibovs)
        for ativo in betas.columns:
            if new_logretorno.loc[choque[0]:choque[1]][ativo].mean() != 0.0:

                cov = np.cov(new_logretorno.loc[choque[0]:choque[1]][ativo].astype(float),ibovs)[0][1]
                if np.isnan(cov) or np.isnan(var):
                    print("1 - restrição",ativo,choque)
                else:
                    beta = cov/var
                    #betas[choque[0]][ativo] = beta
                    betas.loc[choque[0]][ativo] = beta


#Ploting betas

import matplotlib.pyplot as plt
choque1 = []
choque2 = []
choque3 = []
choque4 = []
X = ['LREN3','PETR4','KLBN3','JBSS3','VALE3','BEEF3']
for asset in X:
    choque1.append(betas[asset]['2012-05-13'])
    choque2.append(betas[asset]['2013-01-14'])
    choque3.append(betas[asset]['2015-04-17'])
    choque4.append(betas[asset]['2021-06-07'])

N = len(X)
ind = np.arange(N)
width = 0.2

bar1 = plt.bar(ind, choque1, width, color = 'b')
bar2 = plt.bar(ind+width, choque2, width, color = 'r')
bar3 = plt.bar(ind+width*2, choque3, width, color = 'g')
bar4 = plt.bar(ind+width*3, choque4, width, color = 'y')

plt.xlabel("Ações")
plt.ylabel('Betas')
plt.title("Betas de cada empresa nas maiores quedas de índice")

plt.xticks(ind+width,X)
plt.legend( (bar1,bar2,bar3,bar4),('2012-05-13', '2013-01-14', '2015-04-17','2021-06-07'))
plt.show()

