# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:59:28 2019

@author: Daniel
"""

#Sites úteis
#https://github.com/RomelTorres/alpha_vantage
#https://www.alphavantage.co/documentation/

from alpha_vantage.timeseries import TimeSeries
#import matplotlib.pyplot as plt
import random

#file = open("C:\\Users\\Daniel\\Desktop\\Faculdade\\COMPUTAÇÃO\\Inteligência Artificial\\Aprendizado de Máquina\\Trabalhos\\T2\\nasdaqlisted.txt")
file = open("nasdaqlisted.txt")
strFile = file.read() #lê todo o arquivo
strFile = strFile.splitlines() #lista sem caracter '\n'

#ts = TimeSeries(key='614V0UUUNIY1NDMF', output_format='pandas') #Objeto da classe TimeSeries
ts = TimeSeries(key='S55HHK4M4E665SA7', output_format='pandas') #Objeto da classe TimeSeries

symbol_list = [] #Lista contendo símbolos das empresas (ticker symbols)
data_list = []  #lista contendo dataframes com dados das empresas

for i in range (5):
    symbol_list.append(strFile[random.randint(1,3445)])

for s in symbol_list:
    data_list.append(ts.get_intraday(symbol= s ,interval='5min', outputsize='full'))

i = 0
for x in data_list:
    x = x[0]
    data_list[i] = x
    i = i + 1
    
#data, meta_data = ts.get_intraday(symbol='MSFT',interval='5min', outputsize='full')

#data['1. open'].plot()
#plt.title('Intraday Times Series for the MSFT stock (5 min)')
#plt.show()
