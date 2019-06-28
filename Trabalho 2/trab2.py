#Este software foi implementado usando um ambiente Debian 9.9,
#com Python 3 na versão 3.5.3, Pandas 0.19.2, NumPy 1.12.1,
#Scikit-Learn 0.18 e Matplotlib 2.0.0
#Compatibilidade com versões posteriores não é garantida

#carrega as bibliotecas a serem adicionadas

import os
import pandas as pd 
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

#carrega os arquivos contendo os valores das ações das 50 empresas mais negociadas na Nasdaq
#eles estão armazenados na pasta Nasdaq100list50+

pasta = 'Nasdaq100list50+'

caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
csvs = [arq for arq in arquivos if arq.lower().endswith(".csv")]

csvStock = []

for arq in csvs:
    csvStock = pd.read_csv(arq,parse_dates=['timestamp'],index_col='timestamp')

#após a leitura dos arquivos, são removidas as tuplas com dados ausentes e os dados absolutos 
#são transformados em variações percentuais
    
    csvStock = csvStock.dropna()
    

    csvStock['OpenPct']  = csvStock['open'].pct_change()
    csvStock['HighPct']  = csvStock['high'].pct_change()
    csvStock['LowPct']   = csvStock['low'].pct_change()
    csvStock['ClosePct'] = csvStock['close'].pct_change()
    csvStock['Name']     = arq.replace(".csv","")
    csvStock['Name']     = csvStock['Name'].astype('category')

    csvStock = csvStock[['OpenPct','HighPct','LowPct','ClosePct','Name']].dropna()

#os dados obtidos dos arquivos csv são transformados em um vetor NumPy

raw_data = np.array(csvStock.iloc[:,:-1])

#uma técnica TimeSeriesSplit é usada para dividir a série temporal em treino e teste
#no momento, usamos apenas os dados de treinamento
#a utilização dos dados de teste será implementada posteriormente

tscv = TimeSeriesSplit(n_splits=10)
for train_index, test_index in tscv.split(raw_data):
      train_data, test_data = raw_data[train_index], raw_data[test_index]

labels_true = np.array(train_data[:,-1])

#aqui é executado o algoritmo de agrupamento baseado em densidade DBSCAN
#o valor eps=0.003 obteve o melhor resultado conforme visualização gráfica

db = DBSCAN(eps=0.003, min_samples=10).fit(train_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

#abaixo estão algumas medidas obtidas do DBSCAN

print('Número estimado de grupos: %d' % n_clusters_)
print('Número estimado de outliers: %d' % n_noise_)
print("Homogeneidade: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completude: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-medida: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Índice Rand Ajustado: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Informação Mútua Ajustada: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Coeficiente Silhueta: %0.3f" % metrics.silhouette_score(train_data, labels))

#esta parte gera e mostra o gráfico dos grupos obtidos com o DBSCAN
#na melhor configuração, foi obtido um único grupo e 4 outliers

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = train_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = train_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Número estimado de grupos: %d' % n_clusters_)
plt.show()

#esta parte é uma comparação com o k-Means
#foi utilizada a configuração com 5 clusters

kmeans = KMeans(n_clusters=5, random_state=0).fit_predict(train_data)

plt.scatter(train_data[:, 0], train_data[:, 1], c=kmeans)
plt.title("Divisão em grupos estimada")
plt.show()

#outra comparação com o k-Means
#nesta comparação foi utilizadas configurações com 3 e 8 clusters

estimators = [('k_means_8', KMeans(n_clusters=8)),
              ('k_means_3', KMeans(n_clusters=3))]

fignum = 0
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4,6))
fig.subplots_adjust(hspace=0.7)
titles = ['8 clusters', '3 clusters']
for name, est in estimators:
    est.fit(train_data)
    labels = est.labels_

    axs[fignum].scatter(train_data[:, 3], train_data[:, 0],
                  c=labels.astype(np.float), edgecolor='k')
    axs[fignum].set_xticklabels([])
    axs[fignum].set_yticklabels([])
    axs[fignum].set_xlabel('Fechamento')
    axs[fignum].set_ylabel('Abertura')
    axs[fignum].set_title(titles[fignum - 1])

    fignum = fignum + 1
plt.show()
