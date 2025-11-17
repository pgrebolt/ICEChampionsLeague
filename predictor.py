#!/usr/bin/env python
# coding: utf-8

# ### Prediccions amb el model
# 
# Si volem aprofitar el model per fer prediccions, li hem de passar els paràmetres amb els quals s'ha entrenat el model (dins de `X_train`). Per nosaltres és més fàcil escriure el nom dels jugadors que juguen. A partir d'aquí, crearem la llista de valors dels jugadors i en farem la predicció amb el model.

# In[ ]:


import numpy as np
import pandas as pd
import xarray as xr

import joblib

from sklearn import preprocessing

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


# Llegim les estadístiques de cada jugador
stats_xr = xr.open_dataset('stats.nc', engine='scipy')
stats_xr


# In[ ]:


# Llegim l'scaler
scaler = joblib.load('scaler.pkl')
encoder_scores = joblib.load('encoder_scores.pkl')


# In[ ]:


# Paràmetres que considerem al model, en funció de si el jugador és atacant o defensor (COMPROVAR QUE ÉS EL MATEIX QUE A L'ALTRE NOTEBOOK)
considered_stats_defense = ['GamesPlayed', 'WeigthedELO', 'PlayedDefense', 'WinPlayedDefense', 'ScoredDefensePlayed', 'ReceivedDefensePlayed', 'ELODefense', 'DefenseIndex']
considered_stats_attack = ['GamesPlayed', 'WeigthedELO', 'PlayedAttack', 'WinPlayedAttack', 'ScoredAttackPlayed', 'ReceivedAttackPlayed', 'ELOAttack', 'AttackIndex']


# In[ ]:


# Si volem carregar el model
model = keras.models.load_model('sequential.keras')


# In[ ]:


# Jugadors, per alineació
player_list = ['Víctor', 'Elena', 'Guille', 'Luis']

# Creem la llista d'estadístiques per fer la predicció
stats_match = []
for player in player_list:
        if player in [player_list[0], player_list[2]]: # defensors
            # Triem les estadístiques més recents dels jugadors
            player_stats = stats_xr.isel(matchday=-1).sel(player=player)[considered_stats_defense].to_array().values
        elif player in [player_list[1], player_list[3]]: # atacants
            # Triem les estadístiques més recents dels jugadors
            player_stats = stats_xr.isel(matchday=-1).sel(player=player)[considered_stats_attack].to_array().values
        stats_match = stats_match + list(player_stats) # adjuntem les estadístiques del jugador a les dades d'aquest partit

# Tornem a estandaritzar els valors d'acord a com hem fet amb els valors d'entrenament
stats_match = np.array([stats_match]) # scaler espera una matriu 2D
stats_match_stand = scaler.transform(stats_match.astype(float))

# Fem la predicció
score_prediction = model.predict(stats_match_stand)
score_prediction_output = score_prediction.argmax(axis = 1)

score_prediction_output_label = encoder_scores.inverse_transform(score_prediction_output)

print('Predicted result: ', score_prediction_output_label)

