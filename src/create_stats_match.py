#!/usr/bin/env python
# coding: utf-8

# En aquest codi analitzarem els resultats de cada partida i escriurem les classificacions corresponents.

# Per silenciar un warning vinculat amb el Jupyter Notebook
import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Importem les llibreries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr # per guardar les dades 3D
from collections import Counter
from sklearn.preprocessing import StandardScaler

# Definim tab20 com la paleta per defecte dels plots
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)

# Actualitzem els ELO a cada partit
#for nmatch in range(matches):

#    match_df = data_df.iloc[nmatch].reset_index()

    # Noms dels jugadors
#    j1_name, j2_name, j3_name, j4_name = match_df['Jugador 1'].values[0], match_df['Jugador 2'].values[0], match_df['Jugador 3'].values[0], match_df['Jugador 4'].values[0]

    # Nombre de partits jugats com a atacant o com a defensa (el +1 serveix per facilitar el càlcul posterior i indica que en el partit actual també es guanya experiència)
     #    n1_played = playeddefense_matches.iloc[nmatch][j1_name].values[0] +1 # TODO: fer que playeddefense_matches tingui una columna d'índex de partit, per poder quadrar ara (igual pels altres dataframes)
     #    n2_played = playedattack_matchdays[playedattack_matchdays.index == nmatchday][j2_name].values[0] +1
     #    n3_played = playeddefense_matchdays[playeddefense_matchdays.index == nmatchday][j3_name].values[0] +1
     #    n4_played = playedattack_matchdays[playedattack_matchdays.index == nmatchday][j4_name].values[0] +1

    # ELO mitjà per equip
     #    ELO_local = (elo_rating_defense[j1_name]*n1_played + elo_rating_attack[j2_name]*n2_played) / (n1_played + n2_played)
     #    ELO_visitant = (elo_rating_defense[j3_name]*n3_played + elo_rating_attack[j4_name]*n4_played) / (n3_played + n4_played)

    # Probabilitats de victòria
     #    P_local = 1 / (1 + 10**((ELO_visitant - ELO_local) / 400))
     #    P_visitant = 1 / (1 + 10**((ELO_local - ELO_visitant) / 400))

    # Outcome segons local (1 si guanya local, 0 si guanya visitant)
     #    guanyador = match_df['Guanyador'].values[0]
     #    outcome_local = 1 if (guanyador=='Local') else 0
     #    outcome_visitant = 1 if (guanyador=='Visitant') else 0

    # Paràmetres de ponderació
     #    mu, lamb = 0.3, 0.7

    # Gols que ha rebut cada equip
     #    gols_rebuts_local = match_df['Gols 3'].values[0] + match_df['Gols 4'].values[0]
     #    gols_rebuts_visitant = match_df['Gols 1'].values[0] + match_df['Gols 2'].values[0]

    # Rendiment sobre la contribució al partit segons posició
     #    r_p1 = lamb * (1 - gols_rebuts_local / 3) + mu * (match_df['Gols 1'].values[0] / 3) # defensa
     #    r_p2 = mu * (1 - gols_rebuts_local / 3) + lamb * (match_df['Gols 2'].values[0] / 3) # atac
     #    r_p3 = lamb * (1 - gols_rebuts_visitant / 3) + mu * (match_df['Gols 3'].values[0] / 3)  # defensa
     #    r_p4 = mu * (1 - gols_rebuts_visitant / 3) + lamb * (match_df['Gols 4'].values[0] / 3) # atac

    # Imposem un rendiment mínim de 0.1
     #    r_p1 = max(r_p1, 0.1)
     #    r_p2 = max(r_p2, 0.1)
     #    r_p3 = max(r_p3, 0.1)
     #    r_p4 = max(r_p4, 0.1)

    # Suma dels rendiments
     #    r_local = r_p1 + r_p2
     #    r_visitant = r_p3 + r_p4

    # Actualitzem la ponderació de rendiment pels perdedors
     #    if guanyador == 'Local':
     #        r_p3 = 1 - r_p3
     #        r_p4 = 1 - r_p4

        # Imposem el rendiment mínim
     #        r_p3 = min(r_p3, 0.9)
     #        r_p4 = min(r_p4, 0.9)

     #        r_visitant = r_p3 + r_p4
     #    elif guanyador == 'Visitant':
     #        r_p1 = 1 - r_p1
     #        r_p2 = 1 - r_p2

     #        # Imposem el rendiment mínim
     #        r_p1 = min(r_p1, 0.9)
     #        r_p2 = min(r_p2, 0.9)

     #        r_local = r_p1 + r_p2

    # Actualitzem ELOS
     #    elo_rating_defense[j1_name] = elo_rating_defense[j1_name] + K * (outcome_local - P_local) * r_p1 / r_local
     #    elo_rating_attack[j2_name] = elo_rating_attack[j2_name] + K * (outcome_local - P_local) * r_p2 / r_local
     #    elo_rating_defense[j3_name] = elo_rating_defense[j3_name] + K * (outcome_visitant - P_visitant) * r_p3 / r_visitant
     #    elo_rating_attack[j4_name] = elo_rating_attack[j4_name] + K * (outcome_visitant - P_visitant) * r_p4 / r_visitant


# Funció per actualitzar l'ELO després de cada partit
def update_elo(results_df, parameters):
    # ELO inicial
    initial_ELO = 1000.

    # ELO update constant
    K = 30.

    # Paràmetres de ponderació
    mu, lamb = 0.3, 0.7

    #################
    # Pel primer partit, tots els jugadors tenen l'ELO inicial
    if results_df.shape[0] == 1:  # pel primer partit
        return ({players_names[player_index]: initial_ELO for player_index in range(len(players_names))},
                {players_names[player_index]: initial_ELO for player_index in range(len(players_names))},
                {players_names[player_index]: 1/len(players_names) for player_index in range(len(players_names))})
    #################

    # Obtenim els noms jugadors del partit actual (anotat com a [-1])
    home_defender_1 = results_df.iloc[-1]['Jugador 1']
    home_attacker_1 = results_df.iloc[-1]['Jugador 2']
    away_defender_1 = results_df.iloc[-1]['Jugador 3']
    away_attacker_1 = results_df.iloc[-1]['Jugador 4']

    # Obtenim l'índex que ocupa cada jugador a la llista de noms
    home_defender_index = np.where(players_names == home_defender_1)[0][0]
    home_attacker_index =  np.where(players_names == home_attacker_1)[0][0]
    away_defender_index =  np.where(players_names == away_defender_1)[0][0]
    away_attacker_index =  np.where(players_names == away_attacker_1)[0][0]

    # ELO (rating) actuals (el [-1] representa l'últim partit jugat, l'últim ELO)
    R_home_defender = parameters['ELODefense'][-1][home_defender_index]
    R_home_attacker = parameters['ELOAttack'][-1][home_attacker_index]
    R_away_defender = parameters['ELODefense'][-1][away_defender_index]
    R_away_attacker = parameters['ELOAttack'][-1][away_attacker_index]

    # Partits jugats per cada jugador
    n_home_defender = parameters['PlayedDefense'][-1][home_defender_index]
    n_home_attacker = parameters['PlayedAttack'][-1][home_attacker_index]
    n_away_defender = parameters['PlayedDefense'][-1][away_defender_index]
    n_away_attacker = parameters['PlayedAttack'][-1][away_attacker_index]

    # ELO mitjà per equip
    E_home = (R_home_defender * n_home_defender + R_home_attacker * n_home_attacker) / (n_home_defender + n_home_attacker)
    E_away = (R_away_defender * n_away_defender + R_away_attacker * n_away_attacker) / (n_away_defender + n_away_attacker)

    # Probabilitats de victòria segons ELO
    P_home = 1 / (1 + 10 ** ((E_away - E_home) / 400))
    P_away = 1 / (1 + 10 ** ((E_home - E_away) / 400))

    # Resultat del partit
    guanyador = results_df.iloc[-1]['Guanyador']
    if guanyador == 'Local':
        S_home = 1
        S_away = 0
    else:
        S_home = 0
        S_away = 1

    ## Rendiments individualitzats

    # Gols que ha rebut cada equip
    received_home = results_df.iloc[-1]['Visitant']
    received_away = results_df.iloc[-1]['Local']

    # Rendiment sobre la contribució al partit segons posició
    r_defender_home = lamb * (1 - received_home/3) + mu*(results_df.iloc[-1]['Gols 1']/3) # defensa
    r_attacker_home = mu * (1 - received_home/3) + lamb*(results_df.iloc[-1]['Gols 2']/3) # atac
    r_defender_away = lamb * (1 - received_away/3) + mu*(results_df.iloc[-1]['Gols 3']/3)  # defensa
    r_attacker_away = mu * (1 - received_away/3) + lamb*(results_df.iloc[-1]['Gols 4']/3) # atac

    # Imposem un rendiment mínim de 0.1
    r_defender_home = max(r_defender_home, 0.1)
    r_attacker_home = max(r_attacker_home, 0.1)
    r_defender_away = max(r_defender_away, 0.1)
    r_attacker_away = max(r_attacker_away, 0.1)

    # Suma dels rendiments
    r_home = r_defender_home + r_attacker_home
    r_away = r_defender_away + r_attacker_away

    # Actualitzem la ponderació de rendiment pels perdedors (amb un rendiment mínim)
    if guanyador == 'Local':
        r_defender_away = min(1 - r_defender_away, 0.9)
        r_attacker_away = min(1 - r_attacker_away, 0.9)
        r_away = r_defender_away + r_attacker_away
    elif guanyador == 'Visitant':
        r_defender_home = min(1 - r_defender_home, 0.9)
        r_attacker_home = min(1 - r_attacker_home, 0.9)
        r_home = r_defender_home + r_attacker_home

    ## Actualització dels ràtings ELO
    R_home_defender_new = R_home_defender + K * (S_home - P_home) * r_defender_home / r_home
    R_home_attacker_new = R_home_attacker + K * (S_home - P_home) * r_attacker_home / r_home
    R_away_defender_new = R_away_defender + K * (S_away - P_away) * r_defender_away / r_away
    R_away_attacker_new = R_away_attacker + K * (S_away - P_away) * r_attacker_away / r_away

    ## ELO ponderat
    # Normalizem ELOs (normalització min-max)
    defense_rng = max(parameters['ELODefense'][-1]) - min(parameters['ELODefense'][-1]) # denominador (max - min)
    attack_rng = max(parameters['ELOAttack'][-1]) - min(parameters['ELOAttack'][-1])
    defense_rng = defense_rng if defense_rng != 0 else 1  # Evitem divisió per 0
    attack_rng = attack_rng if attack_rng != 0 else 1  # Evitem divisió per 0
    normalized_ELO_defense = [(parameters['ELODefense'][-1][player_index] - min(parameters['ELODefense'][-1])) / defense_rng for player_index in range(len(players_names))]
    normalized_ELO_attack = [(parameters['ELOAttack'][-1][player_index] - min(parameters['ELOAttack'][-1])) / attack_rng for player_index in range(len(players_names))]

    # Calculem el valor ponderat
    weight_attack = parameters['PlayedAttack'][-1] / parameters['GamesPlayed'][-1]
    weight_defense = parameters['PlayedDefense'][-1] / parameters['GamesPlayed'][-1]
    weight_attack = np.where(weight_attack==np.nan, 1., weight_attack)  # Evitem NaN (si no s'ha jugat cap partit encara)
    weight_defense = np.where(weight_defense==np.nan, 1., weight_defense)

    weighted_ELO = {players_names[player_index]: weight_defense * normalized_ELO_defense[player_index] + weight_attack * normalized_ELO_attack[player_index] for player_index in range(len(players_names))}

    # Actualitzem la matriu amb tots els ELO (només els jugadors del partit actual)
    parameters['ELODefense'][-1][home_defender_index] = R_home_defender_new
    parameters['ELOAttack'][-1][home_attacker_index] = R_home_attacker_new
    parameters['ELODefense'][-1][away_defender_index] = R_away_defender_new
    parameters['ELOAttack'][-1][away_attacker_index] = R_away_attacker_new

    # Tornem un diccionari per atac i un altre per defensa
    return ({players_names[player_index]: parameters['ELODefense'][-1][player_index] for player_index in range(len(players_names))},
            {players_names[player_index]: parameters['ELOAttack'][-1][player_index] for player_index in range(len(players_names))},
            weighted_ELO)

# Carreguem les dades
season = 'historical' # 2,3, 4, historical
if season == 'historical':
    data_df = pd.read_csv(f'../generated_files/results_{season}.csv')
else:
    data_df = pd.read_csv(f'../generated_files/results_Season_{season}.csv')
data_df = pd.read_csv(f'../generated_files/results_prova.csv')
# Emplenem els espais en blanc amb 0
data_df = data_df.fillna(0.)

# Obtenim una llista amb tots els noms dels participants
players_names = np.unique(data_df[['Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4']].values.flatten())

# Llista de dies jugats
if season == 'historical':
    matchdays = pd.unique(data_df['Total_D'])
else:
    matchdays = pd.unique(data_df['D'])

# Nombre de partits jugats
matches = data_df.shape[0]

# Primer partit on guanya el visitant
first_away_win = data_df.index[data_df['Guanyador'] == 'Visitant'][0]

# Comptem quants partits ha jugat cada participant
all_players = data_df['Jugador 1'].tolist() + data_df['Jugador 2'].tolist() + data_df['Jugador 3'].tolist() + data_df['Jugador 4'].tolist()

games_count = dict(Counter(all_players))

# ### Partits jugats i victòries
# Llista amb els noms dels paràmetres
params = [
    "GamesPlayed", "PlayedAttack", "PlayedDefense",
    "WinPlayed", "WinAttackPlayed", "WinDefensePlayed",
    "Scored", "ScoredPlayed", "ScoredAttack", "ScoredDefense",
    "ScoredAttackPlayed", "ScoredDefensePlayed",
    "Received", "ReceivedPlayed", "ReceivedAttack", "ReceivedDefense",
    "ReceivedAttackPlayed", "ReceivedDefensePlayed",
    "ELOAttack", "ELODefense", "WeightedELO"
]

# Diccionaris on anirem guardant les dades abans de crear el xarray Dataset
_acc = {p: [] for p in params}
_match_coords = [] # llista amb el número de jornada

# Funció per convertir un diccionari a una fila d'una matriu numpy
def _dict_to_row(d, players):
    # guaranteed order by players list
    return np.array([d.get(p, 0) for p in players], dtype=float)

# En aquest dataframe hi guardem les estadístiques finals després de cada jornada
played_matchdays = pd.DataFrame(columns=players_names) # jugats total
playedattack_matchdays = pd.DataFrame(columns = players_names) # jugats atac
playeddefense_matchdays = pd.DataFrame(columns = players_names) # jugats defensa
winplayed_matchdays = pd.DataFrame(columns=players_names) # guanyats / jugats
winattackplayed_matchdays = pd.DataFrame(columns=players_names) # guanyats / jugats (atac)
windefenseplayed_matchdays = pd.DataFrame(columns=players_names) # guanyats / jugats (defensa)
elo_attack_matchdays = pd.DataFrame(columns = players_names)
elo_defense_matchdays = pd.DataFrame(columns = players_names)

# Inicialitzem els diccionaris. Afegim tots els jugadors amb 0 partits jugats
playeddefense = {player: 0 for player in players_names}
playedattack = {player: 0 for player in players_names}
played_j1 = {player: 0 for player in players_names}
played_j2 = {player: 0 for player in players_names}
played_j3 = {player: 0 for player in players_names}
played_j4 = {player: 0 for player in players_names}

#ELO rating inicial
elo_rating_attack = {player: 1000. for player in players_names}
elo_rating_defense = {player: 1000. for player in players_names}

# Iterem per cada partit
for match in range(1, matches+1):
    previous_matches = data_df.iloc[:match] # dataframe amb tots els partits previs

    # Recompte de partits jugats per cada jugador
    played_j1_match = previous_matches.groupby('Jugador 1').size()
    played_j3_match = previous_matches.groupby('Jugador 3').size()
    playeddefense_match = {player: played_j1_match.get(player, 0) + played_j3_match.get(player, 0) for player in players_names}
    # crec que seria més eficient fer un dataframe amb només els jugadors que apareixen a cada groupby, i ajuntar-lo a un diccionari comú

    played_j2_match = previous_matches.groupby('Jugador 2').size()
    played_j4_match = previous_matches.groupby('Jugador 4').size()
    playedattack_match = {k: played_j2_match.get(k, 0) + played_j4_match.get(k, 0) for k in players_names}

    players_so_far = list(set(list(playedattack_match.keys()) + list(playeddefense_match.keys()))) # llista de tots els jugadors que han jugat fins ara (en qualsevol posició)
    played = {k: playeddefense_match.get(k, 0) + playedattack_match.get(k, 0) for k in players_names}

    ## Recompte de gols anotats
    # Com a defensor
    scored_j1_match = previous_matches.groupby('Jugador 1')['Gols 1'].sum()
    scored_j3_match = previous_matches.groupby('Jugador 3')['Gols 3'].sum()
    scoreddefense_match = {k: scored_j1_match.get(k, 0) + scored_j3_match.get(k, 0) for k in players_names}

    # Com a atacant
    scored_j2_match = previous_matches.groupby('Jugador 2')['Gols 2'].sum()
    scored_j4_match = previous_matches.groupby('Jugador 4')['Gols 4'].sum()
    scoredattack_match = {k: scored_j2_match.get(k, 0) + scored_j4_match.get(k, 0) for k in players_names}

    # En qualsevol posició
    scored_match = {k: scoreddefense_match.get(k, 0) + scoredattack_match.get(k, 0) for k in players_names}

    ## Recompte de gols rebuts com a defensor
    # Com a defensor (no distingim entre si ha marcat l'atacant o del defensa contrari)
    received_j1_match = previous_matches.groupby('Jugador 1')[['Gols 3', 'Gols 4']].sum().sum(axis=1)
    received_j3_match = previous_matches.groupby('Jugador 3')[['Gols 1', 'Gols 2']].sum().sum(axis=1)
    receiveddefense_match = {k: received_j1_match.get(k, 0) + received_j3_match.get(k, 0) for k in players_names}

    # Com a atacant
    received_j2_match = previous_matches.groupby('Jugador 2')[['Gols 3', 'Gols 4']].sum().sum(axis=1)
    received_j4_match = previous_matches.groupby('Jugador 4')[['Gols 1', 'Gols 2']].sum().sum(axis=1)
    receivedattack_match = {k: received_j2_match.get(k, 0) + received_j4_match.get(k, 0) for k in players_names}

    # En qualsevol posició
    received_match = {k: receiveddefense_match.get(k, 0) + receivedattack_match.get(k, 0) for k in players_names}

    ## Recompte de partits guanyats
    if match < first_away_win+1: # pel primer partit
        try: # si ha guayat el local
            windefense_match = previous_matches.groupby('Guanyador')['Jugador 1'].value_counts()['Local']
            winattack_match = previous_matches.groupby('Guanyador')['Jugador 2'].value_counts()['Local']
        except KeyError:
            windefense_match = previous_matches.groupby('Guanyador')['Jugador 3'].value_counts()['Visitant']
            winattack_match = previous_matches.groupby('Guanyador')['Jugador 4'].value_counts()['Visitant']
    else:
        windefense_local_match = previous_matches.groupby('Guanyador')['Jugador 1'].value_counts()['Local']
        winattack_local_match = previous_matches.groupby('Guanyador')['Jugador 2'].value_counts()['Local']
        windefense_visitant_match = previous_matches.groupby('Guanyador')['Jugador 3'].value_counts()['Visitant']
        winattack_visitant_match = previous_matches.groupby('Guanyador')['Jugador 4'].value_counts()['Visitant']

        # Partits en total guanyats com a defensa i com a atacant
        windefense_match = {k: windefense_local_match.get(k, 0) + windefense_visitant_match.get(k, 0) for k in players_names}
        winattack_match = {k: winattack_local_match.get(k, 0) + winattack_visitant_match.get(k, 0) for k in players_names}

    # Partits en total guanyats en qualsevol posició
    win_match = {k: windefense_match.get(k, 0) + winattack_match.get(k, 0) for k in players_names}

    ## Càlcul de ràtios
    # Anotats
    scoredattackplayed_match = {k: scoredattack_match.get(k, 0) / (playedattack_match.get(k, 1) if playedattack_match.get(k, 1) else 1) for k in players_names} # evitem divisió per 0
    scoreddefenseplayed_match = {k: scoreddefense_match.get(k, 0) / (playeddefense_match.get(k, 1) if playeddefense_match.get(k, 1) else 1) for k in players_names}
    scoredplayed_match = {k: scored_match.get(k, 0) / (played.get(k, 1) if played.get(k, 1) else 1) for k in players_names}

    # Rebuts
    receivedattackplayed_match = {k: receivedattack_match.get(k, 0) / (playedattack_match.get(k, 1) if playedattack_match.get(k, 1) else 1) for k in players_names} # evitem divisió per 0
    receiveddefenseplayed_match = {k: receiveddefense_match.get(k, 0) / (playeddefense_match.get(k, 1) if playeddefense_match.get(k, 1) else 1) for k in players_names}
    receivedplayed_match = {k: received_match.get(k, 0) / (played.get(k, 1) if played.get(k, 1) else 1) for k in players_names}

    ## Càlcul d'ELO actualitzat després d'aquest partit
    elo_defense_match, elo_attack_match, elo_weighted_match = update_elo(previous_matches, _acc)

    print(elo_defense_match)
    print(elo_weighted_match)

    # Guanyats
    winattackplayed_match = {k: winattack_match.get(k, 0) / (playedattack_match.get(k, 1) if playedattack_match.get(k, 1) else 1) for k in players_names} # evitem divisió per 0
    windefenseplayed_match = {k: windefense_match.get(k, 0) / (playeddefense_match.get(k, 1) if playeddefense_match.get(k, 1) else 1) for k in players_names}
    winplayed_match = {k: win_match.get(k, 0) / (played.get(k, 1) if played.get(k, 1) else 1) for k in players_names}

    ## Guardem a dataframes. Cada fila correspon a una jornada, i cada columna a un jugador
    _acc["GamesPlayed"].append(_dict_to_row(played, players_names))
    _acc["PlayedAttack"].append(_dict_to_row(playedattack_match, players_names))
    _acc["PlayedDefense"].append(_dict_to_row(playeddefense_match, players_names))
    _acc["WinPlayed"].append(_dict_to_row(winplayed_match, players_names))
    _acc["WinAttackPlayed"].append(_dict_to_row(winattackplayed_match, players_names))
    _acc["WinDefensePlayed"].append(_dict_to_row(windefenseplayed_match, players_names))

    # scored / received (only if you compute them per matchday)
    _acc["Scored"].append(_dict_to_row(scored_match, players_names))
    _acc["ScoredPlayed"].append(_dict_to_row(scoredplayed_match, players_names))
    _acc["ScoredAttack"].append(_dict_to_row(scoredattack_match, players_names))
    _acc["ScoredDefense"].append(_dict_to_row(scoreddefense_match, players_names))
    _acc["ScoredAttackPlayed"].append(_dict_to_row(scoredattackplayed_match, players_names))
    _acc["ScoredDefensePlayed"].append(_dict_to_row(scoreddefenseplayed_match, players_names))

    _acc["Received"].append(_dict_to_row(received_match, players_names))
    _acc["ReceivedPlayed"].append(_dict_to_row(receivedplayed_match, players_names))
    _acc["ReceivedAttack"].append(_dict_to_row(receivedattack_match, players_names))
    _acc["ReceivedDefense"].append(_dict_to_row(receiveddefense_match, players_names))
    _acc["ReceivedAttackPlayed"].append(_dict_to_row(receivedattackplayed_match, players_names))
    _acc["ReceivedDefensePlayed"].append(_dict_to_row(receiveddefenseplayed_match, players_names))

    # ELO dicts (ensure these are per-matchday snapshots or dictionaries of current ratings)
    _acc["ELOAttack"].append(_dict_to_row(elo_attack_match, players_names))
    _acc["ELODefense"].append(_dict_to_row(elo_defense_match, players_names))
    _acc["WeightedELO"].append(_dict_to_row(elo_weighted_match, players_names))

    # track matchday coordinate (use whatever label you want; here nmatchday+1)
    _match_coords.append(match)

    # Si es vol fer frequencies
    # previous_matches.groupby(['Jugador 1', 'Jugador 2'])['Jugador 1'].count() #això compta les vegades que hi ha hagut una parella

## Creem el dataset de xarray
_data_vars = {
    p: (("match", "player"), np.vstack(_acc[p]) if len(_acc[p]) > 0 else np.empty((0, len(players_names))))
    for p in params
}

ds = xr.Dataset(
    data_vars=_data_vars,
    coords={"match": _match_coords, "player": players_names}
)
print(ds)
print(matches)
print(ds['ReceivedAttack'].sel(match=matches, player='Pau').max())

# Afegim el càlcul dels índexs d'atac i de defensa en base al paràmetres que ja hem calculat:
#
#     attack_index = goals_attack * games_attack / games_total + goals_defense * games_defense / games_total
#
# També afegim el càlcul d'ELO total ponderat pel nombre de partits que ha jugat cada jugador a cada posició. Cal tenir en compte si el jugador no ha jugat cap partit. Utilitzem un nombre total de partits fals per fer el recompte. Per evitar divisions per 0, on hi havia un 0 al nombre de partits jugats hi posem un 1. El 0 de la divisió el farà el numerador.

#filtered_games_played = (ds['GamesPlayed']).where(ds['GamesPlayed'] != 0, 1)

#dataset['AttackIndex'] = dataset['ScoredAttack'] * dataset['PlayedAttack'] / filtered_games_played + dataset['ScoredDefense'] * dataset['PlayedDefense'] / filtered_games_played
#dataset['DefenseIndex'] = dataset['ReceivedAttack'] * dataset['PlayedAttack'] / filtered_games_played + dataset['ScoredDefense'] * dataset['PlayedDefense'] / filtered_games_played

# Weighted ELO a partir del valors normalitzats min-max
normalized_ELO_attack = (dataset['ELOAttack'] - dataset['ELOAttack'].min()) / (dataset['ELOAttack'].max() - dataset['ELOAttack'].min())
normalized_ELO_defense = (dataset['ELODefense'] - dataset['ELODefense'].min()) / (dataset['ELODefense'].max() - dataset['ELODefense'].min())

dataset['WeightedELO'] = normalized_ELO_attack * dataset['PlayedAttack'] / filtered_games_played + normalized_ELO_defense * dataset['PlayedDefense'] / filtered_games_played

# Si algun jugador només ha jugat en una posició, pertorba la normalització min-max. Fem que el seu valor sigui nan
#print(dataset['WeightedELO'].max(), dataset['WeightedELO'].min())
dataset['WeightedELO'] = dataset['WeightedELO'].where(dataset['WeightedELO'] < 500)

## Desem el dataset a un fitxer netCDF
#if season == 'historical':
#    dataset.to_netcdf('../generated_files/stats_historical.nc', mode='w')
#else:
#    dataset.to_netcdf('../generated_files/stats.nc', mode='w')

# ds = xr.open_dataset('stats.nc', engine ='netcdf4') # si volem obrir el fitxer

#print("Stats created successfully.")

    # A partir de previous_matches faria el recompte de gols anotats per cada jugador i estadístiques similars.
    # ELO s'ha de calcular per cada partit inidividualment, perquè es vagi actualitzant
    #print(previous_matches.groupby('Guanyador').count())


