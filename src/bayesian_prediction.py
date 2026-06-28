#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import preprocessing
import pymc as pm
import arviz as az
import scipy.stats as stats
import itertools
from sklearn.decomposition import PCA

from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  roc_auc_score,
  confusion_matrix,
  mean_squared_error,
  r2_score,
  mean_absolute_error,
  classification_report,
  ConfusionMatrixDisplay,
  RocCurveDisplay
)
from sklearn.model_selection import (
  train_test_split,
  cross_val_score,
  KFold,
  StratifiedKFold
)

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

''' Aquest codi utilitza un model XGBoost per predir els resultats dels partits a partir de les estadístiques dels jugadors. '''

# Per poder entrenar el model, ens cal donar-li informació de qui està jugant (amb quins punts forts i punts febles) i quin és el resultat. Per tant, hem de llegir el fitxer amb les estadístiques de cada jugador i també el fitxer amb el registre històric de resultats.

###############################################################
############## Llegir les dades ###############################
###############################################################

# Llegim l'històric de partits i resultats
matches_df = pd.read_csv('../generated_files/results_historical.csv')

# Llegim les estadístiques de cada jugador
stats_xr = xr.open_dataset('../generated_files/stats_historical.nc', engine='scipy')
print("Individual available parameters:")
print(list(stats_xr.keys()))

# Llegim les estadístiques creuades (com un jugador es comporta contra un altre)
#frequencies_xr = xr.open_dataset('../generated_files/teammates_historical.nc', engine='scipy')
#print("Team available parameters:")
#print(list(frequencies_xr.keys()))

###############################################################
############# Crear el dataset amb les dades ##################
###############################################################

# Codifiquem el nom de cada jugador
encoder_names = preprocessing.LabelEncoder()
players_names = np.unique(matches_df[['Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4']].values.flatten())
player_codes = encoder_names.fit_transform(players_names)
player_codes_dict = {players_names[i]: player_codes[i] for i in range(len(players_names))}

# Calculem la diferència de gols entre local i visitant, que serà la variable objectiu per predir el resultat del partit
goal_diff = matches_df['Local'] - matches_df['Visitant']
#Stats_training['Goal_Diff'] = goal_diff[1:].values # hem tret el primer partit perquè es faran servir les estadístiques del partit anterior per predir cada resultat, i per al primer partit no hi ha dades prèvies

######## Gràfiques prèvies ############
## Pintem la comparativa entre ELO difference i goal difference, per veure si hi ha correlació
#plt.scatter(Stats_training['ELODifference'], Stats_training['Goal_Diff'], alpha=0.5)
#plt.xlabel("ELO difference")
#plt.ylabel("Goal difference")
#plt.show()

##### Triem les dades
last_elo_attack = stats_xr['ELOAttack'].isel(match=-1).values
last_elo_defense = stats_xr['ELODefense'].isel(match=-1).values
goal_diff = matches_df['Local'] - matches_df['Visitant']

###############################################################
########## Estandaritzem els valors d'ELO #####################
###############################################################
# Cal estandaritzar els valors d'ELO per tenir un model més estable. Després es pot reconvertir a l'escala ELO usual

scaler_attack = preprocessing.StandardScaler() # define scaler
last_elo_attack_std = scaler_attack.fit_transform(last_elo_attack.reshape(-1,1)) # fit scaler to training data and transform it
scaler_defense = preprocessing.StandardScaler() # define scaler
last_elo_defense_std = scaler_defense.fit_transform(last_elo_defense.reshape(-1,1)) # fit scaler to training data and transform it

last_elo_attack_std = last_elo_attack_std.reshape(-1) # reshape to 1D array
last_elo_defense_std = last_elo_defense_std.reshape(-1) # reshape to 1D array
################################################################
n_players = len(players_names)
n_matches = matches_df.shape[0]
print("Number of players: ", n_players)
print("Number of games: ", n_matches)

# Índexs dels jugadors que juguen a cada partit (per a cada rol: atacant i defensor, local i visitant)
atk_local = [player_codes_dict[matches_df['Jugador 1'].iloc[i]] for i in range(n_matches)]
def_local = [player_codes_dict[matches_df['Jugador 2'].iloc[i]] for i in range(n_matches)]
atk_visit = [player_codes_dict[matches_df['Jugador 3'].iloc[i]] for i in range(n_matches)]
def_visit = [player_codes_dict[matches_df['Jugador 4'].iloc[i]] for i in range(n_matches)]

mean_goals_game = np.mean(matches_df['Local'] + matches_df['Visitant'])
std_goals_game = np.std(matches_df['Local'] + matches_df['Visitant'])
log_mean_goals_game = np.log(mean_goals_game)
log_std_goals_game = np.log(std_goals_game)
print("Goals per game: ", mean_goals_game, "+-", std_goals_game)
###############################################################
################### Model bayesià #############################
###############################################################
# Definició del model probabilístic
with pm.Model() as foosball_poisson_model:

    # 1. Priors d'habilitat. Una distribució normal per a cada jugador, amb mitjana i desviació estàndard basades en l'ELO estandarditzat.
    skill_attack = pm.Normal("skill_attack", mu=last_elo_attack_std, sigma=0.1, shape=n_players)
    skill_defense = pm.Normal("skill_defense", mu=last_elo_defense_std, sigma=0.1, shape=n_players)

    # Intercept (La mitjana base de gols que es fan en un partit normal de futbolí)
    baseline_scoring = pm.Normal("baseline_scoring", mu=0, sigma=1)

    # 2. LÒGICA DE LES TAXES DE GOL (λ - Lambda)
    # L'esperança de gols es calcula com a exponencial (funció link logarítmica de Poisson)
    # Gols esperats del Local: El seu atac VS la teva defensa
    lambda_local = pm.math.exp(baseline_scoring + skill_attack[atk_local] - skill_defense[def_visit])
    # Gols esperats del Visitant: El seu atac VS la teva defensa
    lambda_visitant = pm.math.exp(baseline_scoring + skill_attack[atk_visit] - skill_defense[def_local])

    # 3. LIKELIHOOD (Dues distribucions de Poisson independents)
    # Connectem les lambdes teòriques amb els teus gols de veritat de la base de dades
    obs_local = pm.Poisson("obs_local", mu=lambda_local, observed=matches_df['Local'])
    obs_visitant = pm.Poisson("obs_visitant", mu=lambda_visitant, observed=matches_df['Visitant'])

    # Executem el mostreig MCMC
    trace = pm.sample(draws=1000, tune=1000, return_inferencedata=True, random_seed=42)

    # Generem noves dades (posterior predictive) d'acord amb els paràmetres que s'han ajustat
    ppc = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

##########################################################
####### Comprovació que la inferència ha convergit #######
##########################################################
# Imprimim el sumari
summary = az.summary(trace, var_names=["skill_attack", "skill_defense", "baseline_scoring"], round_to=2)
print(summary)

## Posterior predictive plot
fig = az.plot_ppc(data = trace, kind='kde', num_pp_samples=100, figsize=(10,10), mean=True)
plt.savefig('../results/Bayesian_prediction/posterior_predictive_bayesian.png', dpi=300, bbox_inches='tight')
plt.show()
###########################################################
############ Simulació d'un nou partit ####################
###########################################################
# Jugadors que juguen al nou partit
j1, j2, j3, j4 = 'Gisela', 'Rebeca', 'Víctor', 'Pau' #exemple

print("\n\nPredicció de resultat amb els següents equips: ")
print(f"Equip Local: {j1} + {j2}")
print(f"Equip Visitant: {j3} + {j4}\n")
# Extraiem els valors mitjans de les habilitats dels jugadors
baseline_mean = summary.loc["baseline_scoring", "mean"]
atk_j1 = summary.loc[f"skill_attack[{player_codes_dict[j1]}]", "mean"]
def_j2 = summary.loc[f"skill_defense[{player_codes_dict[j2]}]", "mean"]
atk_j3 = summary.loc[f"skill_attack[{player_codes_dict[j3]}]", "mean"]
def_j4 = summary.loc[f"skill_defense[{player_codes_dict[j4]}]", "mean"]

# Calculem la lambda del nou partit (nombre de gols esperats per partit) per a cada equip
lambda_L = np.exp(baseline_mean + atk_j1 - def_j4)
lambda_V = np.exp(baseline_mean + atk_j3 - def_j2)

# Calculem les probabilitats de fer 0, 1, 2 o 3 gols per a cadascú d'acord amb la distribució de Poisson
prob_L = [stats.poisson.pmf(k, lambda_L) for k in [0, 1, 2, 3]]
prob_V = [stats.poisson.pmf(k, lambda_V) for k in [0, 1, 2, 3]]

# Normalitzem perquè sumin 1 (per corregir el tall del futbolí a 3 gols)
prob_L /= np.sum(prob_L)
prob_V /= np.sum(prob_V)

# Creem la llista de marcadors exactes (6 combinacions reals de final de partit)
resultats_possibles = {
    "3-0": prob_L[3] * prob_V[0],
    "3-1": prob_L[3] * prob_V[1],
    "3-2": prob_L[3] * prob_V[2],
    "0-3": prob_L[0] * prob_V[3],
    "1-3": prob_L[1] * prob_V[3],
    "2-3": prob_L[2] * prob_V[3]
}
## Normalitzem respecte els únics resultats possibles
total_sum = 0 # suma de probabilitats
for prob in resultats_possibles.values():
    total_sum += prob

# Normalitzem cada valor i imprimim
print("Resultats amb la mitjana de la posterior:")
for marcador, p in resultats_possibles.items():
    resultats_possibles[marcador] = resultats_possibles[marcador] / total_sum
    print(f"Probabilitat de {marcador}: {(p/total_sum)*100:.1f}%")


#########################################################
#### Predicció de resultat a partir de la posterior #####
#########################################################
#Extraiem la posterior
posterior = trace.posterior

# Desem tots els valors de la posterior en arrays 1D per a cada jugador i per al baseline
baseline_posterior = posterior['baseline_scoring'].values.flatten()
skill_atk_l_posterior = posterior['skill_attack'].values[:, :, player_codes_dict[j1]].flatten()
skill_def_l_posterior = posterior['skill_defense'].values[:, :, player_codes_dict[j2]].flatten()
skill_atk_v_posterior = posterior['skill_attack'].values[:, :, player_codes_dict[j3]].flatten()
skill_def_v_posterior = posterior['skill_defense'].values[:, :, player_codes_dict[j4]].flatten()

# Calculem les lambdes per a cada simulació de la posterior
lambda_l_posterior = np.exp(baseline_posterior + skill_atk_l_posterior - skill_def_v_posterior)
lambda_v_posterior = np.exp(baseline_posterior + skill_atk_v_posterior - skill_def_l_posterior)

# Simulem la probabilitat que cada equip faci 0, 1, 2 o 3 gols a cada partit
gols = np.array([0, 1, 2, 3])[:, np.newaxis]  # Shape: (4, 1)
prob_L_posterior = stats.poisson.pmf(gols, lambda_l_posterior)
prob_V_posterior = stats.poisson.pmf(gols, lambda_v_posterior)

# Normalitzem perquè sumin 1 (per corregir el tall del futbolí a 3 gols)
prob_L_posterior = prob_L_posterior / np.sum(prob_L, axis=0)
prob_V_posterior = prob_V_posterior / np.sum(prob_V, axis=0)

# Creem la matriu de marcadors exactes (6 combinacions reals de final de partit)
resultats_possibles = {
    "3-0": np.array([np.mean(prob_L_posterior[3] * prob_V_posterior[0]), np.std(prob_L_posterior[3] * prob_V_posterior[0])]),
    "3-1": np.array([np.mean(prob_L_posterior[3] * prob_V_posterior[1]), np.std(prob_L_posterior[3] * prob_V_posterior[1])]),
    "3-2": np.array([np.mean(prob_L_posterior[3] * prob_V_posterior[2]), np.std(prob_L_posterior[3] * prob_V_posterior[2])]),
    "0-3": np.array([np.mean(prob_L_posterior[0] * prob_V_posterior[3]), np.std(prob_L_posterior[0] * prob_V_posterior[3])]),
    "1-3": np.array([np.mean(prob_L_posterior[1] * prob_V_posterior[3]), np.std(prob_L_posterior[1] * prob_V_posterior[3])]),
    "2-3": np.array([np.mean(prob_L_posterior[2] * prob_V_posterior[3]), np.std(prob_L_posterior[2] * prob_V_posterior[3])])
}

## Normalitzem respecte els únics resultats possibles
total_sum = 0 # suma de probabilitats
for prob in resultats_possibles.values():
    total_sum += prob[0]

# Normalitzem cada valor i imprimim
print("\nResultats amb la posterior:")
# Ara el teu model et dirà exactament quina probabilitat hi ha de cada marcador!
for marcador, p in resultats_possibles.items():
    print(resultats_possibles[marcador])
    print(p, p[0]/total_sum, total_sum)
    resultats_possibles[marcador] = np.array([p[0] / total_sum, p[1]/total_sum])
    print(f"Probabilitat de {marcador}: {p[0]/total_sum*100:.1f} +- {p[1]/total_sum*100:.1f}%")

############################################################
############# Gràfica de possibles resultats ###############
############################################################
###### Serà una matriu 6 x 6 amb els possibles resultats
resultats_matriu = np.zeros((4,4))
for gols_local, gols_visitant in list(itertools.product([0, 1, 2, 3], [0, 1, 2, 3])):
    key = str(gols_local) + '-' + str(gols_visitant)
    print(key)
    if key in resultats_possibles.keys():
        prob = resultats_possibles[key][0]
        plt.text(gols_visitant, gols_local, str(prob)[:4])
    else:
        prob = np.nan

    resultats_matriu[gols_local, gols_visitant] = prob

plt.title(f"Probabilitat de gols\nLocal:{j1}-{j2}    Visitant:{j3}-{j4}")
plt.imshow(resultats_matriu)
plt.xticks([0, 1, 2, 3])
plt.xlabel('Gols visitant')
plt.yticks([0,1,2,3])
plt.ylabel("Gols local")
plt.savefig('../results/Bayesian_prediction/result_probability_bayesian.png', dpi=300, bbox_inches='tight')
plt.show()

############################################################
########### Gràfica de forest plot #########################
############################################################
# Gràfica de bosc per a l'atac de tots els jugadors
#az.plot_forest(trace, var_names=["skill_attack", "skill_defense"], combined=True)

means_elo_atk = []
lower_bounds_atk = []
upper_bounds_atk = []
means_elo_def = []
lower_bounds_def = []
upper_bounds_def = []
for i in range(n_players):
    player_samples_atk = trace.posterior['skill_attack'].values[:,:, i].flatten()
    player_samples_def = trace.posterior['skill_defense'].values[:,:, i].flatten()

    player_samples_atk_elo = scaler_attack.inverse_transform(player_samples_atk.reshape(-1, 1)).flatten()
    player_samples_def_elo = scaler_defense.inverse_transform(player_samples_def.reshape(-1, 1)).flatten()

    mean_elo_atk = np.mean(player_samples_atk_elo)
    lower_bound_atk = np.percentile(player_samples_atk_elo, 2.5)
    upper_bound_atk = np.percentile(player_samples_atk_elo, 97.5)
    mean_elo_def = np.mean(player_samples_def_elo)
    lower_bound_def = np.percentile(player_samples_def_elo, 2.5)
    upper_bound_def = np.percentile(player_samples_def_elo, 97.5)
    means_elo_atk.append(mean_elo_atk)
    lower_bounds_atk.append(lower_bound_atk)
    upper_bounds_atk.append(upper_bound_atk)
    means_elo_def.append(mean_elo_def)
    lower_bounds_def.append(lower_bound_def)
    upper_bounds_def.append(upper_bound_def)

# Calculem la longitud dels rutes d'error (requerit per plt.errorbar)
error_left_atk = np.array(means_elo_atk) - np.array(lower_bounds_atk)
error_right_atk = np.array(upper_bounds_atk) - np.array(means_elo_atk)
error_left_def = np.array(means_elo_def) - np.array(lower_bounds_def)
error_right_def = np.array(upper_bounds_def) - np.array(means_elo_def)

# Noms dels jugadors per a l'eix Y (opcional)
noms_jugadors = [f" {encoder_names.inverse_transform([i])[0]}" for i in range(n_players)]

# Pintem el Forest Plot manual
fig, axs = plt.subplots(figsize=(8, 10), ncols = 1, nrows = 2)
axs = axs.flatten()

axs[0].errorbar(means_elo_atk, noms_jugadors, xerr=[error_left_atk, error_right_atk], fmt='o', color='teal', capsize=5,
             elinewidth=2)
axs[1].errorbar(means_elo_def, noms_jugadors, xerr=[error_left_def, error_right_def], fmt='o', color='teal', capsize=5,
             elinewidth=2)
axs[0].set_title("ELO Attack (94% HDI)")
axs[1].set_title("ELO Defense (94% HDI)")

axs[1].set_xlabel("ELO points")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('../results/Bayesian_prediction/ELO_individual_bayesian.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

########### Gràfica de ELO Attack vs ELO Defense
fig,ax = plt.subplots(figsize=(8,5))

ax.errorbar(means_elo_atk, means_elo_def, xerr=[error_left_atk, error_right_atk], yerr=[error_left_def, error_right_def],
            fmt='o', color ='teal', capsize=5, elinewidth=2)
ax.hlines(1000, xmin=800, xmax=1150, linestyle='--', color ='gray', alpha = 0.7)
ax.vlines(1000, ymin=850, ymax=1050, linestyle='--', color ='gray', alpha = 0.7)

for i in range(n_players):
    nom_jugador = encoder_names.inverse_transform([i])[0]
    ax.text(means_elo_atk[i], means_elo_def[i], nom_jugador, fontsize=9)
ax.set_xlabel("ELO Attack (94% HDI)")
ax.set_ylabel("ELO Defense (94% HDI)")

plt.savefig('../results/Bayesian_prediction/ELO_paired_bayesian.png', dpi=300, bbox_inches='tight')
plt.show()