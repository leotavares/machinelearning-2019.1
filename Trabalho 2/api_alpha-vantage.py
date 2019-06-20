# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:59:28 2019

@author: Daniel
"""

#Sites úteis
#https://github.com/RomelTorres/alpha_vantage
#https://www.alphavantage.co/documentation/

from alpha_vantage.timeseries import TimeSeries
import time
#import matplotlib.pyplot as plt
#import random
#import csv
#import json

#file = open("C:\\Users\\Daniel\\Desktop\\Faculdade\\COMPUTAÇÃO\\Inteligência Artificial\\Aprendizado de Máquina\\Trabalhos\\T2\\nasdaqlisted.txt")

#Tem que ser usado o caminho relativo ao arquivo nasdaqlisted.txt, e não o absoluto. Caso contrário, o programa funcionará somente na máquina na qual ele foi desenvolvido.
file = open("nasdaq100list.txt")
strFile = file.read() #lê todo o arquivo
strFile = strFile.splitlines() #lista sem caracter '\n'

#A chave a ser usada é a de quem vai rodar o programa e pode ser obtida gratuitamente no site da Alpha Vantage
#ts = TimeSeries(key='614V0UUUNIY1NDMF', output_format='csv') #Objeto da classe TimeSeries
ts = TimeSeries(key='S55HHK4M4E665SA7', output_format='csv') #Objeto da classe TimeSeries

#symbol_list = [] #Lista contendo símbolos das empresas (ticker symbols)
#data_list = []  #lista contendo dataframes com dados das empresas

#for i in range (5):
#    symbol_list.append(strFile[random.randint(1,3445)])

#for s in symbol_list:
#    nome_arquivo = s + '.csv'
#    arquivo = open(nome_arquivo,'w')
#    arquivo.write(ts.get_intraday(symbol= s ,interval='1min', outputsize='full'))
#    arquivo.close()
#    data_list.append(ts.get_intraday(symbol= s ,interval='1min', outputsize='full'))

#i = 0
#for x in data_list:
#    x = x[0]
#    data_list[i] = x
#    i = i + 1

#len(data_list)

i = 0
for s in strFile:
    nome_arquivo = s + '.csv'
    arquivo = open(nome_arquivo,'w')
    data, meta_data = ts.get_intraday(symbol=s,interval='1min', outputsize='full')

    for linha in data:
        count = True
        for aux in linha:
            if count:
               palavra = aux
               count = False
            else:
               palavra = palavra + ',' + aux
        palavra = palavra + '\n'
        arquivo.write(palavra)

    arquivo.close() 
    i += 1

    if i == 5:
       time.sleep(70)
       i = 0

#print(data)
#print('-------------------------------------------------------------------------------')
#print(meta_data)

#data['1. open'].plot()
#plt.title('Intraday Times Series for the MSFT stock (5 min)')
#plt.show()
