#!/usr/bin/env python
# coding: utf-8

# Aquest codi pren les dades de tots els jugadors i en fa una PCA i k-means per trobar-ne les similituds

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.preprocessing import (
  StandardScaler,
  LabelEncoder
)

import warnings
warnings.filterwarnings('ignore')

# set visualisation style
plt.rcParams['figure.figsize'] = (12, 6)

# set random seed for reproducibility
RND_STATE = 42 #None
np.random.seed(RND_STATE)

###############################################################
############## Llegir les dades ###############################
###############################################################

# Llegim l'històric de partits i resultats
matches_df = pd.read_csv('../generated_files/results_historical.csv')

# Llegim les estadístiques de cada jugador
stats_xr = xr.open_dataset('../generated_files/stats_historical.nc', engine='scipy')
print("Individual available parameters:")
print(list(stats_xr.keys()))

# Llistat de paràmetres que considerem per cada jugador
parameters = ['WinAttackPlayed', 'WinDefensePlayed',
              'ScoredAttackPlayed', 'ScoredDefensePlayed',
              'ReceivedAttackPlayed', 'ReceivedDefensePlayed',
              'NeatGoalsAttackPlayed', 'NeatGoalsDefensePlayed',
              'CleanSheetAttackPlayed', 'CleanSheetDefensePlayed']
#              'ELOAttack', 'ELODefense', 'WeightedELO']

print("Number of original parameters considered: ", len(parameters))

###############################################################
############# Crear el dataset amb les dades ##################
###############################################################

# Triem les últimes dades de cada jugador, pels paràmetres seleccionats, i les convertim a un dataframe
player_data = stats_xr[parameters].isel(match=-1).to_dataframe().reset_index().drop(columns=['match'])

# Normalitzem les columnes de cada paràmetre per tenir una PCA més estable. Després es pot reconvertir a l'escala ELO usual
scaler = StandardScaler()
player_data_scaled = scaler.fit_transform(player_data.drop(columns=['player']))

# Fem la PCA per tots els paràmtres per saber quants components principals són necessaris per explicar la major part de la variància
pca = PCA(n_components=len(parameters))
pca.fit(player_data_scaled)

# Pintem el gràfic de sedimentació (Scree Plot) per veure quants components principals són necessaris per explicar la major part de la variància
var_individual = pca.explained_variance_ratio_ * 100
var_acumulada = np.cumsum(var_individual)
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(parameters)+1), var_individual, alpha=0.6, align="center", label="Variància individual (%)")
plt.step(range(1, len(parameters)+1), var_acumulada, where="mid", label="Variància acumulada (%)", color="red",)# Línia acumulada
plt.axhline(y=80, color="grey", linestyle="--", label="Llindar 80% variància",)
plt.ylabel("Percentatge de variància explicada")
plt.xlabel("Nombre de components principals")
plt.xticks(range(1, len(parameters)+1))
plt.title("Scree Plot - Selecció de components")
plt.legend(loc="best")
plt.grid(axis="y", linestyle=":", alpha=0.7)
plt.savefig('../results/ML/PCA_scree_plot.png', dpi=300, bbox_inches='tight')
plt.clf()

## Havent triat el nombre de components principals, fem la PCA amb aquest nombre de components
pca = PCA(n_components=4)
player_data_pca = pca.fit_transform(player_data_scaled)

# Desem les dades transformades en un nou dataframe amb els components principals
player_data_pca_df = pd.DataFrame(data=player_data_pca, columns=[f'PC{i+1}' for i in range(player_data_pca.shape[1])])

## Apliquem l'elbow method per trobar el valor de K òptim
wcss = []  # Within-Cluster Sum of Squares (Inèrcia)
k_range = range(1, 8)  # Provem valors de k de l'1 al 10
for k in k_range:
    # n_init='auto' o 10 per assegurar múltiples inicialitzacions
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(player_data_pca_df)
    wcss.append(kmeans.inertia_)
plt.plot(k_range, wcss, marker="o", linestyle="--", color="b")
plt.title("Mètode del Colze (Elbow Method)")
plt.xlabel("Nombre de clústers (k)")
plt.ylabel("Inèrcia (WCSS)")
plt.xticks(k_range)
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig('../results/ML/PCA_elbow_method.png', dpi=300, bbox_inches='tight')
plt.clf()

# Fem un K-means clustering amb 3 grups per identificar els jugadors similars
km = KMeans(n_clusters=3, random_state=RND_STATE)
km.fit(player_data_pca_df.values)
labels = km.labels_
player_data_pca_df['cluster'] = labels
player_data_pca_df['player'] = player_data['player'].values

print(labels)
# Pintem els jugadors en el nou espai PCA amb els 6 components principals
xparam, yparam = 'PC2', 'PC3'
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for label in np.unique(labels):
    plt.scatter(player_data_pca_df[player_data_pca_df['cluster'] == label][xparam],
                player_data_pca_df[player_data_pca_df['cluster'] == label][yparam],
                alpha=0.7, label=f'Cluster {label}', color=colors[label % len(colors)])
for player_name in player_data_pca_df['player']:
    plt.annotate(player_name, (player_data_pca_df.loc[player_data_pca_df['player'] == player_name, xparam].values[0],
                                player_data_pca_df.loc[player_data_pca_df['player'] == player_name, yparam].values[0]),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
plt.xlabel(xparam)
plt.ylabel(yparam)
plt.savefig('../results/ML/PCA_kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.clf()

