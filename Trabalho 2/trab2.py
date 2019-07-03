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
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

#carrega os arquivos contendo os valores das ações das 50 empresas mais negociadas na Nasdaq
#eles estão armazenados na pasta Nasdaq100list50+

pasta = 'Nasdaq100list50+'

caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
csvs = [arq for arq in arquivos if arq.lower().endswith(".csv")]

arquivo = open("saída.txt","w")

csvStock = []

for arq in csvs:
    csvStock = pd.read_csv(arq,parse_dates=['timestamp'],index_col='timestamp')

#após a leitura dos arquivos, são removidas as tuplas com dados ausentes e os dados absolutos 
#são transformados em variações percentuais
    
    csvStock = csvStock.dropna()
    
    csvStock['Name']     = arq.replace(".csv","")
    csvStock['Name']     = csvStock['Name'].astype('category')

csvStock['OpenPct']  = csvStock['open'].pct_change()
csvStock['HighPct']  = csvStock['high'].pct_change()
csvStock['LowPct']   = csvStock['low'].pct_change()
csvStock['ClosePct'] = csvStock['close'].pct_change()

csvStock = csvStock[['OpenPct','HighPct','LowPct','ClosePct','Name']].dropna()

#os dados obtidos dos arquivos csv são transformados em um vetor NumPy

train_data = np.array(csvStock.iloc[:,:-1])
class_data = np.array(csvStock.iloc[:,-1])

#uma técnica TimeSeriesSplit é usada para dividir a série temporal em treino e teste
#no momento, usamos apenas os dados de treinamento
#a utilização dos dados de teste será implementada posteriormente

X_train, X_test, y_train, y_test = train_test_split(train_data, class_data, test_size=0.2)

labels_true = np.array(y_train)

#aqui é executado o algoritmo de agrupamento baseado em densidade DBSCAN
#o valor eps=0.003 obteve o melhor resultado conforme visualização gráfica

db = DBSCAN(eps=0.003, min_samples=10).fit(X_train)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

#abaixo estão algumas medidas obtidas do DBSCAN

print('Medidas Obtidas Usando o DBSCAN e os Dados Com Todos os Componentes')
print('Número estimado de grupos: %d' % n_clusters_)
print('Número estimado de outliers: %d' % n_noise_)
print("Homogeneidade: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completude: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-medida: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Índice Rand Ajustado: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Informação Mútua Ajustada: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Informação Mútua Normalizada: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
print("Informação Mútua: %0.3f" % metrics.mutual_info_score(labels_true, labels))
print("Índice Fowlkes-Mallows: %0.3f" % metrics.fowlkes_mallows_score(labels_true, labels))
print("Coeficiente Calinski-Harabaz: %0.3f" % metrics.calinski_harabaz_score(X_train, labels))
print("Coeficiente Silhueta: %0.3f" % metrics.silhouette_score(X_train, labels))

arquivo.write('Medidas Obtidas Usando o DBSCAN e os Dados Com Todos os Componentes\n')
arquivo.write('Número estimado de grupos: %d\n' % n_clusters_)
arquivo.write('Número estimado de outliers: %d\n' % n_noise_)
arquivo.write("Homogeneidade: %0.3f\n" % metrics.homogeneity_score(labels_true, labels))
arquivo.write("Completude: %0.3f\n" % metrics.completeness_score(labels_true, labels))
arquivo.write("V-medida: %0.3f\n" % metrics.v_measure_score(labels_true, labels))
arquivo.write("Índice Rand Ajustado: %0.3f\n" % metrics.adjusted_rand_score(labels_true, labels))
arquivo.write("Informação Mútua Ajustada: %0.3f\n" % metrics.adjusted_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua Normalizada: %0.3f\n" % metrics.normalized_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua: %0.3f\n" % metrics.mutual_info_score(labels_true, labels))
arquivo.write("Índice Fowlkes-Mallows: %0.3f\n" % metrics.fowlkes_mallows_score(labels_true, labels))
arquivo.write("Coeficiente Calinski-Harabaz: %0.3f\n" % metrics.calinski_harabaz_score(X_train, labels))
arquivo.write("Coeficiente Silhueta: %0.3f\n" % metrics.silhouette_score(X_train, labels))


#esta parte gera e mostra o gráfico dos grupos obtidos com o DBSCAN
#na melhor configuração, foi obtido um único grupo e 4 outliers

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X_train[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X_train[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Número estimado de grupos: %d' % n_clusters_)
plt.show()

#esta parte é uma comparação com o k-Means
#foi utilizada a configuração com 5 clusters

kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train)
labels = kmeans.labels_

print('')
print('Medidas Obtidas Usando o KMeans de 5 Clusters e os Dados Com Todos os Componentes')
print("Homogeneidade: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completude: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-medida: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Índice Rand Ajustado: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Informação Mútua Ajustada: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Informação Mútua Normalizada: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
print("Informação Mútua: %0.3f" % metrics.mutual_info_score(labels_true, labels))
print("Índice Fowlkes-Mallows: %0.3f" % metrics.fowlkes_mallows_score(labels_true, labels))
print("Coeficiente Calinski-Harabaz: %0.3f" % metrics.calinski_harabaz_score(X_train, labels))
print("Coeficiente Silhueta: %0.3f" % metrics.silhouette_score(X_train, labels))

arquivo.write('\n')
arquivo.write('Medidas Obtidas Usando o KMeans de 5 Clusters e os Dados Com Todos os Componentes\n')
arquivo.write("Homogeneidade: %0.3f\n" % metrics.homogeneity_score(labels_true, labels))
arquivo.write("Completude: %0.3f\n" % metrics.completeness_score(labels_true, labels))
arquivo.write("V-medida: %0.3f\n" % metrics.v_measure_score(labels_true, labels))
arquivo.write("Índice Rand Ajustado: %0.3f\n" % metrics.adjusted_rand_score(labels_true, labels))
arquivo.write("Informação Mútua Ajustada: %0.3f\n" % metrics.adjusted_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua Normalizada: %0.3f\n" % metrics.normalized_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua: %0.3f\n" % metrics.mutual_info_score(labels_true, labels))
arquivo.write("Índice Fowlkes-Mallows: %0.3f\n" % metrics.fowlkes_mallows_score(labels_true, labels))
arquivo.write("Coeficiente Calinski-Harabaz: %0.3f\n" % metrics.calinski_harabaz_score(X_train, labels))
arquivo.write("Coeficiente Silhueta: %0.3f\n" % metrics.silhouette_score(X_train, labels))

plt.scatter(X_train[:, 0], X_train[:, 1], c=labels.astype(np.float))
plt.title("Divisão em grupos estimada")
plt.show()

#outra comparação com o k-Means
#nesta comparação foi utilizadas configurações com 3 e 8 clusters

estimators = [('k_means_8', KMeans(n_clusters=8)),
              ('k_means_3', KMeans(n_clusters=3))]

fignum = 0
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4,6))
fig.subplots_adjust(hspace=0.7)
titles = ['8 Clusters', '3 Clusters']
for name, est in estimators:
    est.fit(X_train)
    labels = est.labels_

    print('')
    print('Medidas Obtidas Usando o KMeans de %s e os Dados Com Todos os Componentes' % titles[fignum])
    print("Homogeneidade: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completude: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-medida: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Índice Rand Ajustado: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print("Informação Mútua Ajustada: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Informação Mútua Normalizada: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
    print("Informação Mútua: %0.3f" % metrics.mutual_info_score(labels_true, labels))
    print("Índice Fowlkes-Mallows: %0.3f" % metrics.fowlkes_mallows_score(labels_true, labels)) 
    print("Coeficiente Calinski-Harabaz: %0.3f" % metrics.calinski_harabaz_score(X_train, labels))
    print("Coeficiente Silhueta: %0.3f" % metrics.silhouette_score(X_train, labels))

    arquivo.write('\n')
    arquivo.write('Medidas Obtidas Usando o KMeans de %s e os Dados Com Todos os Componentes\n' % titles[fignum])
    arquivo.write("Homogeneidade: %0.3f\n" % metrics.homogeneity_score(labels_true, labels))
    arquivo.write("Completude: %0.3f\n" % metrics.completeness_score(labels_true, labels))
    arquivo.write("V-medida: %0.3f\n" % metrics.v_measure_score(labels_true, labels))
    arquivo.write("Índice Rand Ajustado: %0.3f\n" % metrics.adjusted_rand_score(labels_true, labels))
    arquivo.write("Informação Mútua Ajustada: %0.3f\n" % metrics.adjusted_mutual_info_score(labels_true, labels))
    arquivo.write("Informação Mútua Normalizada: %0.3f\n" % metrics.normalized_mutual_info_score(labels_true, labels))
    arquivo.write("Informação Mútua: %0.3f\n" % metrics.mutual_info_score(labels_true, labels))
    arquivo.write("Índice Fowlkes-Mallows: %0.3f\n" % metrics.fowlkes_mallows_score(labels_true, labels)) 
    arquivo.write("Coeficiente Calinski-Harabaz: %0.3f\n" % metrics.calinski_harabaz_score(X_train, labels))
    arquivo.write("Coeficiente Silhueta: %0.3f\n" % metrics.silhouette_score(X_train, labels))

    axs[fignum].scatter(X_train[:, 3], X_train[:, 0],
                  c=labels.astype(np.float), edgecolor='k')
    axs[fignum].set_xticklabels([])
    axs[fignum].set_yticklabels([])
    axs[fignum].set_xlabel('Fechamento')
    axs[fignum].set_ylabel('Abertura')
    axs[fignum].set_title(titles[fignum])

    fignum = fignum + 1
plt.show()

x = StandardScaler().fit_transform(X_train)
pca = PCA(n_components=2)
principalComponents = pd.DataFrame(pca.fit_transform(x))

principalComponents.plot.scatter(x=0,y=1)
plt.show()

df_class = pd.DataFrame(y_train)
principalComponents_train = pd.concat([principalComponents,df_class],axis=1)

#aqui é executado o algoritmo de agrupamento baseado em densidade DBSCAN
#o valor eps=0.003 obteve o melhor resultado conforme visualização gráfica

labels_true = np.array(y_train)

db = DBSCAN(eps=0.17, min_samples=10).fit(principalComponents)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

#abaixo estão algumas medidas obtidas do DBSCAN

print('')
print('Medidas Obtidas Usando o DBSCAN, os Dados de Treinamento, e os Dois Componentes Principais')
print('Número estimado de grupos: %d' % n_clusters_)
print('Número estimado de outliers: %d' % n_noise_)
print("Homogeneidade: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completude: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-medida: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Índice Rand Ajustado: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Informação Mútua Ajustada: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Informação Mútua Normalizada: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
print("Informação Mútua: %0.3f" % metrics.mutual_info_score(labels_true, labels))
print("Índice Fowlkes-Mallows: %0.3f" % metrics.fowlkes_mallows_score(labels_true, labels))
print("Coeficiente Calinski-Harabaz: %0.3f" % metrics.calinski_harabaz_score(principalComponents, labels))
print("Coeficiente Silhueta: %0.3f" % metrics.silhouette_score(principalComponents, labels))

arquivo.write('\n')
arquivo.write('Medidas Obtidas Usando o DBSCAN, os Dados de Treinamento, e os Dois Componentes Principais\n')
arquivo.write('Número estimado de grupos: %d\n' % n_clusters_)
arquivo.write('Número estimado de outliers: %d\n' % n_noise_)
arquivo.write("Homogeneidade: %0.3f\n" % metrics.homogeneity_score(labels_true, labels))
arquivo.write("Completude: %0.3f\n" % metrics.completeness_score(labels_true, labels))
arquivo.write("V-medida: %0.3f\n" % metrics.v_measure_score(labels_true, labels))
arquivo.write("Índice Rand Ajustado: %0.3f\n" % metrics.adjusted_rand_score(labels_true, labels))
arquivo.write("Informação Mútua Ajustada: %0.3f\n" % metrics.adjusted_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua Normalizada: %0.3f\n" % metrics.normalized_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua: %0.3f\n" % metrics.mutual_info_score(labels_true, labels))
arquivo.write("Índice Fowlkes-Mallows: %0.3f\n" % metrics.fowlkes_mallows_score(labels_true, labels))
arquivo.write("Coeficiente Calinski-Harabaz: %0.3f\n" % metrics.calinski_harabaz_score(principalComponents, labels))
arquivo.write("Coeficiente Silhueta: %0.3f\n" % metrics.silhouette_score(principalComponents, labels))

#esta parte gera e mostra o gráfico dos grupos obtidos com o DBSCAN
#na melhor configuração, foi obtido um único grupo e 4 outliers

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = principalComponents_train[class_member_mask & core_samples_mask]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = principalComponents_train[class_member_mask & ~core_samples_mask]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Número estimado de grupos: %d' % n_clusters_)
plt.show()

#esta parte é uma comparação com o k-Means
#foi utilizada a configuração com 5 clusters

kmeans = KMeans(n_clusters=5, random_state=0).fit_predict(principalComponents)

print('')
print('Medidas Obtidas Usando o KMeans de 5 Clusters, os Dados de Treinamento, e os Dois Componentes Principais')
print("Homogeneidade: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completude: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-medida: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Índice Rand Ajustado: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Informação Mútua Ajustada: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Informação Mútua Normalizada: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
print("Informação Mútua: %0.3f" % metrics.mutual_info_score(labels_true, labels))
print("Índice Fowlkes-Mallows: %0.3f" % metrics.fowlkes_mallows_score(labels_true, labels))
print("Coeficiente Calinski-Harabaz: %0.3f" % metrics.calinski_harabaz_score(principalComponents, labels))
print("Coeficiente Silhueta: %0.3f" % metrics.silhouette_score(principalComponents, labels))

arquivo.write('\n')
arquivo.write('Medidas Obtidas Usando o KMeans de 5 Clusters, os Dados de Treinamento, e os Dois Componentes Principais\n')
arquivo.write("Homogeneidade: %0.3f\n" % metrics.homogeneity_score(labels_true, labels))
arquivo.write("Completude: %0.3f\n" % metrics.completeness_score(labels_true, labels))
arquivo.write("V-medida: %0.3f\n" % metrics.v_measure_score(labels_true, labels))
arquivo.write("Índice Rand Ajustado: %0.3f\n" % metrics.adjusted_rand_score(labels_true, labels))
arquivo.write("Informação Mútua Ajustada: %0.3f\n" % metrics.adjusted_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua Normalizada: %0.3f\n" % metrics.normalized_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua: %0.3f\n" % metrics.mutual_info_score(labels_true, labels))
arquivo.write("Índice Fowlkes-Mallows: %0.3f\n" % metrics.fowlkes_mallows_score(labels_true, labels))
arquivo.write("Coeficiente Calinski-Harabaz: %0.3f\n" % metrics.calinski_harabaz_score(principalComponents, labels))
arquivo.write("Coeficiente Silhueta: %0.3f\n" % metrics.silhouette_score(principalComponents, labels))

plt.scatter(principalComponents_train.iloc[:, 0], principalComponents_train.iloc[:, 1], c=kmeans)
plt.title("Divisão em grupos estimada")
plt.show()

#outra comparação com o k-Means
#nesta comparação foi utilizadas configurações com 3 e 8 clusters

estimators = [('k_means_8', KMeans(n_clusters=8)),
              ('k_means_3', KMeans(n_clusters=3))]

fignum = 0
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4,6))
fig.subplots_adjust(hspace=0.7)
titles = ['8 Clusters', '3 Clusters']
for name, est in estimators:
    est.fit(principalComponents)
    labels = est.labels_

    print('')
    print('Medidas Obtidas Usando o KMeans de %s, os Dados de Treinamento, e os Dois Componentes Principais' % titles[fignum]	)
    print("Homogeneidade: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completude: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-medida: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Índice Rand Ajustado: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print("Informação Mútua Ajustada: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Informação Mútua Normalizada: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
    print("Informação Mútua: %0.3f" % metrics.mutual_info_score(labels_true, labels))
    print("Índice Fowlkes-Mallows: %0.3f" % metrics.fowlkes_mallows_score(labels_true, labels))
    print("Coeficiente Calinski-Harabaz: %0.3f" % metrics.calinski_harabaz_score(principalComponents, labels))
    print("Coeficiente Silhueta: %0.3f" % metrics.silhouette_score(principalComponents, labels))

    arquivo.write('\n')
    arquivo.write('Medidas Obtidas Usando o KMeans de %s, os Dados de Treinamento, e os Dois Componentes Principais\n' % titles[fignum]	)
    arquivo.write("Homogeneidade: %0.3f\n" % metrics.homogeneity_score(labels_true, labels))
    arquivo.write("Completude: %0.3f\n" % metrics.completeness_score(labels_true, labels))
    arquivo.write("V-medida: %0.3f\n" % metrics.v_measure_score(labels_true, labels))
    arquivo.write("Índice Rand Ajustado: %0.3f\n" % metrics.adjusted_rand_score(labels_true, labels))
    arquivo.write("Informação Mútua Ajustada: %0.3f\n" % metrics.adjusted_mutual_info_score(labels_true, labels))
    arquivo.write("Informação Mútua Normalizada: %0.3f\n" % metrics.normalized_mutual_info_score(labels_true, labels))
    arquivo.write("Informação Mútua: %0.3f\n" % metrics.mutual_info_score(labels_true, labels))
    arquivo.write("Índice Fowlkes-Mallows: %0.3f\n" % metrics.fowlkes_mallows_score(labels_true, labels))
    arquivo.write("Coeficiente Calinski-Harabaz: %0.3f\n" % metrics.calinski_harabaz_score(principalComponents, labels))
    arquivo.write("Coeficiente Silhueta: %0.3f\n" % metrics.silhouette_score(principalComponents, labels))

    axs[fignum].scatter(principalComponents_train.iloc[:, 0], principalComponents_train.iloc[:, 1],
                  c=labels.astype(np.float), edgecolor='k')
    axs[fignum].set_xticklabels([])
    axs[fignum].set_yticklabels([])
#    axs[fignum].set_xlabel('Fechamento')
#    axs[fignum].set_ylabel('Abertura')
    axs[fignum].set_title(titles[fignum])

    fignum = fignum + 1
plt.show()

x = StandardScaler().fit_transform(X_test)
pca = PCA(n_components=2)
principalComponents = pd.DataFrame(pca.fit_transform(x))

principalComponents.plot.scatter(x=0,y=1)
plt.show()

df_class = pd.DataFrame(y_test)
principalComponents_test = pd.concat([principalComponents,df_class],axis=1)


#aqui é executado o algoritmo de agrupamento baseado em densidade DBSCAN
#o valor eps=0.003 obteve o melhor resultado conforme visualização gráfica

labels_true = np.array(y_test)

db = DBSCAN(eps=0.20, min_samples=10).fit(principalComponents)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

#abaixo estão algumas medidas obtidas do DBSCAN

print('')
print('Medidas Obtidas Usando o DBSCAN, os Dados de Teste, e os Dois Componentes Principais')
print('Número estimado de grupos: %d' % n_clusters_)
print('Número estimado de outliers: %d' % n_noise_)
print("Homogeneidade: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completude: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-medida: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Índice Rand Ajustado: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Informação Mútua Ajustada: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Informação Mútua Normalizada: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
print("Informação Mútua: %0.3f" % metrics.mutual_info_score(labels_true, labels))
print("Índice Fowlkes-Mallows: %0.3f" % metrics.fowlkes_mallows_score(labels_true, labels))
#print("Coeficiente Calinski-Harabaz: %0.3f" % metrics.calinski_harabaz_score(principalComponents, labels))
#print("Coeficiente Silhueta: %0.3f" % metrics.silhouette_score(principalComponents, labels))
#print("Coeficiente Silhueta Para Cada Amostra: %0.3f" % metrics.silhouette_samples(principalComponents, labels))

arquivo.write('\n')
arquivo.write('Medidas Obtidas Usando o DBSCAN, os Dados de Teste, e os Dois Componentes Principais\n')
arquivo.write('Número estimado de grupos: %d\n' % n_clusters_)
arquivo.write('Número estimado de outliers: %d\n' % n_noise_)
arquivo.write("Homogeneidade: %0.3f\n" % metrics.homogeneity_score(labels_true, labels))
arquivo.write("Completude: %0.3f\n" % metrics.completeness_score(labels_true, labels))
arquivo.write("V-medida: %0.3f\n" % metrics.v_measure_score(labels_true, labels))
arquivo.write("Índice Rand Ajustado: %0.3f\n" % metrics.adjusted_rand_score(labels_true, labels))
arquivo.write("Informação Mútua Ajustada: %0.3f\n" % metrics.adjusted_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua Normalizada: %0.3f\n" % metrics.normalized_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua: %0.3f\n" % metrics.mutual_info_score(labels_true, labels))
arquivo.write("Índice Fowlkes-Mallows: %0.3f\n" % metrics.fowlkes_mallows_score(labels_true, labels))
#arquivo.write("Coeficiente Calinski-Harabaz: %0.3f\n" % metrics.calinski_harabaz_score(principalComponents, labels))
#arquivo.write("Coeficiente Silhueta: %0.3f\n" % metrics.silhouette_score(principalComponents, labels))
#arquivo.write("Coeficiente Silhueta Para Cada Amostra: %0.3f\n" % metrics.silhouette_samples(principalComponents, labels))

#esta parte gera e mostra o gráfico dos grupos obtidos com o DBSCAN
#na melhor configuração, foi obtido um único grupo e 4 outliers

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = principalComponents_test[class_member_mask & core_samples_mask]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = principalComponents_test[class_member_mask & ~core_samples_mask]
    plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Número estimado de grupos: %d' % n_clusters_)
plt.show()

#esta parte é uma comparação com o k-Means
#foi utilizada a configuração com 5 clusters

kmeans = KMeans(n_clusters=5, random_state=0).fit_predict(principalComponents)

print('')
print('Medidas Obtidas Usando o KMeans de 5 Clusters, os Dados de Teste, e os Dois Componentes Principais')
print("Homogeneidade: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completude: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-medida: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Índice Rand Ajustado: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Informação Mútua Ajustada: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Informação Mútua Normalizada: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
print("Informação Mútua: %0.3f" % metrics.mutual_info_score(labels_true, labels))
print("Índice Fowlkes-Mallows: %0.3f" % metrics.fowlkes_mallows_score(labels_true, labels))
#print("Coeficiente Calinski-Harabaz: %0.3f" % metrics.calinski_harabaz_score(principalComponents, labels))
#print("Coeficiente Silhueta: %0.3f" % metrics.silhouette_score(principalComponents, labels))
#print("Coeficiente Silhueta Para Cada Amostra: %0.3f" % metrics.silhouette_samples(principalComponents, labels))

arquivo.write('\n')
arquivo.write('Medidas Obtidas Usando o KMeans de 5 Clusters, os Dados de Teste, e os Dois Componentes Principais\n')
arquivo.write("Homogeneidade: %0.3f\n" % metrics.homogeneity_score(labels_true, labels))
arquivo.write("Completude: %0.3f\n" % metrics.completeness_score(labels_true, labels))
arquivo.write("V-medida: %0.3f\n" % metrics.v_measure_score(labels_true, labels))
arquivo.write("Índice Rand Ajustado: %0.3f\n" % metrics.adjusted_rand_score(labels_true, labels))
arquivo.write("Informação Mútua Ajustada: %0.3f\n" % metrics.adjusted_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua Normalizada: %0.3f\n" % metrics.normalized_mutual_info_score(labels_true, labels))
arquivo.write("Informação Mútua: %0.3f\n" % metrics.mutual_info_score(labels_true, labels))
arquivo.write("Índice Fowlkes-Mallows: %0.3f\n" % metrics.fowlkes_mallows_score(labels_true, labels))
#arquivo.write("Coeficiente Calinski-Harabaz: %0.3f\n" % metrics.calinski_harabaz_score(principalComponents, labels))
#arquivo.write("Coeficiente Silhueta: %0.3f\n" % metrics.silhouette_score(principalComponents, labels))
#arquivo.write("Coeficiente Silhueta Para Cada Amostra: %0.3f\n" % metrics.silhouette_samples(principalComponents, labels))

plt.scatter(principalComponents_test.iloc[:, 0], principalComponents_test.iloc[:, 1], c=kmeans)
plt.title("Divisão em grupos estimada")
plt.show()

#outra comparação com o k-Means
#nesta comparação foi utilizadas configurações com 3 e 8 clusters

estimators = [('k_means_8', KMeans(n_clusters=8)),
              ('k_means_3', KMeans(n_clusters=3))]

fignum = 0
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4,6))
fig.subplots_adjust(hspace=0.7)
titles = ['8 Clusters', '3 Clusters']
for name, est in estimators:
    est.fit(principalComponents)
    labels = est.labels_

    print('')
    print('Medidas Obtidas Usando o KMeans de %s, os Dados de Teste, e os Dois Componentes Principais' % titles[fignum])
    print("Homogeneidade: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completude: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-medida: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Índice Rand Ajustado: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print("Informação Mútua Ajustada: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Informação Mútua Normalizada: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
    print("Informação Mútua: %0.3f" % metrics.mutual_info_score(labels_true, labels))
    print("Índice Fowlkes-Mallows: %0.3f" % metrics.fowlkes_mallows_score(labels_true, labels))
   #print("Coeficiente Calinski-Harabaz: %0.3f" % metrics.calinski_harabaz_score(principalComponents, labels))
   #print("Coeficiente Silhueta: %0.3f" % metrics.silhouette_score(principalComponents, labels))
   #print("Coeficiente Silhueta Para Cada Amostra: %0.3f" % metrics.silhouette_samples(principalComponents, labels))

    arquivo.write('\n')
    arquivo.write('Medidas Obtidas Usando o KMeans de %s, os Dados de Teste, e os Dois Componentes Principais\n' % titles[fignum])
    arquivo.write("Homogeneidade: %0.3f\n" % metrics.homogeneity_score(labels_true, labels))
    arquivo.write("Completude: %0.3f\n" % metrics.completeness_score(labels_true, labels))
    arquivo.write("V-medida: %0.3f\n" % metrics.v_measure_score(labels_true, labels))
    arquivo.write("Índice Rand Ajustado: %0.3f\n" % metrics.adjusted_rand_score(labels_true, labels))
    arquivo.write("Informação Mútua Ajustada: %0.3f\n" % metrics.adjusted_mutual_info_score(labels_true, labels))
    arquivo.write("Informação Mútua Normalizada: %0.3f\n" % metrics.normalized_mutual_info_score(labels_true, labels))
    arquivo.write("Informação Mútua: %0.3f\n" % metrics.mutual_info_score(labels_true, labels))
    arquivo.write("Índice Fowlkes-Mallows: %0.3f\n" % metrics.fowlkes_mallows_score(labels_true, labels))
   #arquivo.write("Coeficiente Calinski-Harabaz: %0.3f\n" % metrics.calinski_harabaz_score(principalComponents, labels))
   #arquivo.write("Coeficiente Silhueta: %0.3f\n" % metrics.silhouette_score(principalComponents, labels))
   #arquivo.write("Coeficiente Silhueta Para Cada Amostra: %0.3f\n" % metrics.silhouette_samples(principalComponents, labels))

    axs[fignum].scatter(principalComponents_test.iloc[:, 0], principalComponents_test.iloc[:, 1],
                  c=labels.astype(np.float), edgecolor='k')
    axs[fignum].set_xticklabels([])
    axs[fignum].set_yticklabels([])
#    axs[fignum].set_xlabel('Fechamento')
#    axs[fignum].set_ylabel('Abertura')
    axs[fignum].set_title(titles[fignum])

    fignum = fignum + 1
plt.show()

arquivo.close()
