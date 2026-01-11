#!/usr/bin/env python
# coding: utf-8

# ### Prediccions amb el model
# 
# Si volem aprofitar el model per fer prediccions, li hem de passar els paràmetres amb els quals s'ha entrenat el model (dins de `X_train`). Per nosaltres és més fàcil escriure el nom dels jugadors que juguen. A partir d'aquí, crearem la llista de valors dels jugadors i en farem la predicció amb el model.

import numpy as np
import pandas as pd
import xarray as xr

import joblib

from sklearn import preprocessing

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Llegim les estadístiques de cada jugador
stats_xr = xr.open_dataset('../generated_files/stats.nc', engine='scipy')
last_mach = stats_xr['match'].values[-1]

# Llegim l'scaler
scaler = joblib.load('../generated_files/scaler.pkl')
encoder_winning = joblib.load('../generated_files/encoder_winning.pkl')

# Paràmetres que considerem al model, en funció de si el jugador és atacant o defensor (COMPROVAR QUE ÉS EL MATEIX QUE A L'ALTRE NOTEBOOK)
considered_stats_defense = ['WinDefensePlayed', 'WinPlayedMatchday']
considered_stats_attack = ['WinAttackPlayed', 'WinPlayedMatchday']
considered_stats_teams = ['ELOAttackDefenseDifference',
                          'ELODefenseAttackDifference',
                          'WinsLocal', 'WinsVisitant']

# Si volem carregar el model
pipeline = joblib.load('../generated_files/xgb_winning_best_pipeline.joblib')

# Jugadors, per alineació
player_list = ['Víctor', 'Antía', 'Guille', 'Luis']

# Creem la llista d'estadístiques per fer la predicció
stats_match = []
for player in player_list:
    if player in player_list[:1]:  # defensors
        # Triem les estadístiques dels jugadors en aquest partit
        player_stats = stats_xr.sel(match=last_match, player=player)[considered_stats_defense].to_array().values
    elif player in player_list[1:]:  # atacants
        # Triem les estadístiques dels jugadors en aquest partit
        player_stats = stats_xr.sel(match=last_match, player=player)[considered_stats_attack].to_array().values
    stats_match = stats_match + list(player_stats)  # adjuntem les estadístiques del jugador a les dades d'aquest partit

    # Afegim a la llista paràmetres del partit, creuant els dos equips
    elo_attack_difference = (stats_xr.sel(match=last_match, player=player_list[1])['ELOAttack'].values.item() -
                             stats_xr.sel(match=last_match, player=player_list[3])['ELOAttack'].values.item())
    elo_defense_difference = (stats_xr.sel(match=last_match, player=player_list[0])['ELOAttack'].values.item() -
                              stats_xr.sel(match=last_match, player=player_list[2])['ELOAttack'].values.item())
    elo_attackh_defensea_difference = (stats_xr.sel(match=match, player=match_df['Jugador 2'])['ELOAttack'].values.item() -
                                       stats_xr.sel(match=match, player=match_df['Jugador 3'])[
                                           'ELODefense'].values.item())  # diferència ELO atacant-defensor rivals
    elo_defenseh_attacka_difference = (stats_xr.sel(match=match, player=match_df['Jugador 1'])['ELODefense'].values.item() -
                                       stats_xr.sel(match=match, player=match_df['Jugador 4'])['ELOAttack'].values.item())
    close_wins_local = frequencies_xr.sel(teammate=match_df['Jugador 1'], player=match_df['Jugador 2'])[
        'CloseWinsPlayed'].values.item()
    close_wins_visitant = frequencies_xr.sel(teammate=match_df['Jugador 1'], player=match_df['Jugador 2'])[
        'CloseWinsPlayed'].values.item()
    receivedgoals_defense_defense_local = \
    frequencies_xr.sel(defender=match_df['Jugador 1'], defender_rival=match_df['Jugador 3'])[
        'ReceivedGoalsGamesDefenseDefense'].values.item()
    receivedgoals_defense_defense_visitant = \
    frequencies_xr.sel(defender=match_df['Jugador 3'], defender_rival=match_df['Jugador 1'])[
        'ReceivedGoalsGamesDefenseDefense'].values.item()
    receivedgoals_attack_defense_local = \
    frequencies_xr.sel(defender=match_df['Jugador 1'], attacker_rival=match_df['Jugador 4'])[
        'ReceivedGoalsGamesAttackDefense'].values.item()
    receivedgoals_attack_defense_visitant = \
    frequencies_xr.sel(defender=match_df['Jugador 3'], attacker_rival=match_df['Jugador 2'])[
        'ReceivedGoalsGamesAttackDefense'].values.item()
    team_wins_local = frequencies_xr.sel(teammate=match_df['Jugador 1'], player=match_df['Jugador 2'])[
        'TeammatesWinsPlayed'].values.item()
    team_wins_visitant = frequencies_xr.sel(teammate=match_df['Jugador 3'], player=match_df['Jugador 4'])[
        'TeammatesWinsPlayed'].values.item()
    #    stats_match = stats_match + [elo_attack_difference, elo_defense_difference, elo_attackh_defensea_difference, elo_defenseh_attacka_difference,
    #                                 close_wins_local, close_wins_visitant,
    #                                 receivedgoals_defense_defense_local, receivedgoals_defense_defense_visitant,
    #                                 receivedgoals_attack_defense_local, receivedgoals_attack_defense_visitant,
    #                                 team_wins_local, team_wins_visitant]
    stats_match = stats_match + [elo_attackh_defensea_difference, elo_defenseh_attacka_difference,
                                 team_wins_local, team_wins_visitant]

# Tornem a estandaritzar els valors d'acord a com hem fet amb els valors d'entrenament
stats_match = np.array([stats_match]) # scaler espera una matriu 2D
stats_match_stand = scaler.transform(stats_match.astype(float))

# Fem la predicció
score_prediction = model.predict(stats_match_stand)
score_prediction_output = score_prediction.argmax(axis = 1)

score_prediction_output_label = encoder_scores.inverse_transform(score_prediction_output)

print('Predicted result: ', score_prediction_output_label)