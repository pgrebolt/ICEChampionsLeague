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

# Carreguem les dades
season = '5' # 2,3, 4, historical
if season == 'historical':
    data_df = pd.read_csv(f'../generated_files/results_{season}.csv')
else:
    data_df = pd.read_csv(f'../generated_files/results_Season_{season}.csv')
#data_df = pd.read_csv(f'../generated_files/results_prova.csv')
# Emplenem els espais en blanc amb 0
data_df = data_df.fillna(0.)

# Funció per actualitzar l'ELO després de cada partit
def update_elo(results_df, parameters):
    # Condició de si és el primer partit
    first_game = True if results_df.shape[0] == 1 else False

    # ELO inicial
    initial_ELO = 1000.

    # ELO update constant
    K = 30.

    # Paràmetres de ponderació
    mu, lamb = 0.7, 0.3

    #################
    # Pel primer partit, tots els jugadors tenen l'ELO inicial
    #if results_df.shape[0] == 1:  # pel primer partit
    #    return ({players_names[player_index]: initial_ELO for player_index in range(len(players_names))},
    #            {players_names[player_index]: initial_ELO for player_index in range(len(players_names))},
    #            {players_names[player_index]: 1/len(players_names) for player_index in range(len(players_names))})
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

    if first_game: # pel primer partit, assignar els valors inicials
        R_home_defender, R_home_attacker, R_away_defender, R_away_attacker = initial_ELO, initial_ELO, initial_ELO, initial_ELO
        n_home_defender, n_home_attacker, n_away_defender, n_away_attacker = 0., 0., 0., 0.
    else:
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
    if n_home_defender == 0. and n_home_attacker == 0.: # si és el primer partit en cada posició de tots dos jugadors, assigna ELO inicial (evita divisió per 0)
        E_home = initial_ELO
    else:
        E_home = (R_home_defender * n_home_defender + R_home_attacker * n_home_attacker) / (n_home_defender + n_home_attacker)
    if n_away_defender == 0. and n_away_attacker == 0.:
        E_away = initial_ELO
    else:
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
    r_defender_home = mu * (1 - received_home/3) + lamb*(results_df.iloc[-1]['Gols 1']/3) # defensa
    r_attacker_home = lamb * (1 - received_home/3) + mu*(results_df.iloc[-1]['Gols 2']/3) # atac
    r_defender_away = mu * (1 - received_away/3) + lamb*(results_df.iloc[-1]['Gols 3']/3)  # defensa
    r_attacker_away = lamb * (1 - received_away/3) + mu*(results_df.iloc[-1]['Gols 4']/3) # atac

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
    #print(R_away_defender, S_away, P_away, r_defender_away, r_attacker_away, r_away, R_away_defender_new, R_away_attacker_new)

    ## ELO ponderat
    if first_game:
        # Si és el primer partit, assignem un ELO ponderat neutre (0.5)
        weighted_ELO = {players_names[player_index]: 0.5 for player_index in range(len(players_names))}
    else:
        # Normalizem ELOs (normalització min-max)
        defense_rng = max(parameters['ELODefense'][-1]) - min(parameters['ELODefense'][-1]) # denominador (max - min)
        attack_rng = max(parameters['ELOAttack'][-1]) - min(parameters['ELOAttack'][-1])
        defense_rng = defense_rng if defense_rng != 0 else 1  # Evitem divisió per 0
        attack_rng = attack_rng if attack_rng != 0 else 1  # Evitem divisió per 0
        normalized_ELO_defense = [(parameters['ELODefense'][-1][player_index] - min(parameters['ELODefense'][-1])) / defense_rng for player_index in range(len(players_names))]
        normalized_ELO_attack = [(parameters['ELOAttack'][-1][player_index] - min(parameters['ELOAttack'][-1])) / attack_rng for player_index in range(len(players_names))]
        if np.all(normalized_ELO_defense==np.float64(0.)): # pel primer partit, assignar manualment els pesos a 0.5
            normalized_ELO_defense = 0.5*np.ones(len(players_names))
        if np.all(normalized_ELO_attack == np.float64(0.)):
            normalized_ELO_attack = 0.5*np.ones(len(players_names))
        # Calculem el valor ponderat
        weight_attack = np.divide(parameters['PlayedAttack'][-1], parameters['GamesPlayed'][-1],
                                  out=np.zeros_like(parameters['PlayedAttack'][-1]), where=parameters['GamesPlayed'][-1]!=0)
        weight_defense = np.divide(parameters['PlayedDefense'][-1], parameters['GamesPlayed'][-1],
                                  out=np.zeros_like(parameters['PlayedDefense'][-1]),
                                  where=parameters['GamesPlayed'][-1] != 0)
    #    weight_attack = parameters['PlayedAttack'][-1] / parameters['GamesPlayed'][-1]
        #weight_defense = parameters['PlayedDefense'][-1] / parameters['GamesPlayed'][-1]
        #weight_attack = np.where(weight_attack==np.nan, 1., weight_attack)  # Evitem NaN (si no s'ha jugat cap partit encara)
        #weight_defense = np.where(weight_defense==np.nan, 1., weight_defense)

        weighted_ELO = {players_names[player_index]: weight_defense[player_index] * normalized_ELO_defense[player_index] + weight_attack[player_index] * normalized_ELO_attack[player_index] for player_index in range(len(players_names))}

    ## Retornem els nous ELOs
    if first_game:
        # Llista amb els ELO després del primer partit
        first_elos_attack = initial_ELO * np.ones(len(players_names)) # llista amb ELO igual per tothom
        first_elos_defense = initial_ELO * np.ones(len(players_names))
        first_elos_defense[home_defender_index] = R_home_defender_new # actualitzem només els jugadors del partit
        first_elos_attack[home_attacker_index] = R_home_attacker_new
        first_elos_defense[away_defender_index] = R_away_defender_new
        first_elos_attack[away_attacker_index] = R_away_attacker_new

        # Retorn en forma de diccionari
        return ({players_names[player_index]: first_elos_defense[player_index] for player_index in range(len(players_names))},
                {players_names[player_index]: first_elos_attack[player_index] for player_index in range(len(players_names))},
                weighted_ELO)

    else:
        # Actualitzem la matriu amb tots els ELO (només els jugadors del partit actual)
        parameters['ELODefense'][-1][home_defender_index] = R_home_defender_new
        parameters['ELOAttack'][-1][home_attacker_index] = R_home_attacker_new
        parameters['ELODefense'][-1][away_defender_index] = R_away_defender_new
        parameters['ELOAttack'][-1][away_attacker_index] = R_away_attacker_new

        # Tornem un diccionari per atac, un altre per defensa i un altre per weighted
        return ({players_names[player_index]: parameters['ELODefense'][-1][player_index] for player_index in range(len(players_names))},
                {players_names[player_index]: parameters['ELOAttack'][-1][player_index] for player_index in range(len(players_names))},
                weighted_ELO)


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

    #print(elo_attack_match)
    #print(elo_weighted_match)

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
    #TODO: no està creant bé els diccionaris. El primer partit no es desa i l'últim i el penúltim són iguals
    #print(elo_attack_match)
    #print(_acc["ELOAttack"])
    _acc["ELOAttack"].append(_dict_to_row(elo_attack_match, players_names))
    #print(_acc["ELOAttack"], '\n')
    _acc["ELODefense"].append(_dict_to_row(elo_defense_match, players_names))
    _acc["WeightedELO"].append(_dict_to_row(elo_weighted_match, players_names))

    # track matchday coordinate
    _match_coords.append(match)

    # Si es vol fer frequencies
    # previous_matches.groupby(['Jugador 1', 'Jugador 2'])['Jugador 1'].count() #això compta les vegades que hi ha hagut una parella

## Creem el dataset de xarray
# Creem un diccionari de paràmetre (key) i (value->) tupla amb el nom de dimensions i una matriu numpy dels paràmetres. És la manera que a xarray li agrada
_data_vars = {
    p: (("match", "player"), np.vstack(_acc[p]) if len(_acc[p]) > 0 else np.empty((0, len(players_names))))
    for p in params
}

# Creem el dataset a partir del diccionari
ds = xr.Dataset(
    data_vars=_data_vars,
    coords={"match": _match_coords, "player": players_names}
)
#print(ds)
#print(matches)
#print(_match_coords, 'aaaaa')
#print(ds['ELOAttack'].sel(match=5))

# Afegim el càlcul dels índexs d'atac i de defensa en base al paràmetres que ja hem calculat:
#
#     attack_index = goals_attack * games_attack / games_total + goals_defense * games_defense / games_total
#
# També afegim el càlcul d'ELO total ponderat pel nombre de partits que ha jugat cada jugador a cada posició. Cal tenir en compte si el jugador no ha jugat cap partit. Utilitzem un nombre total de partits fals per fer el recompte. Per evitar divisions per 0, on hi havia un 0 al nombre de partits jugats hi posem un 1. El 0 de la divisió el farà el numerador.

#filtered_games_played = (ds['GamesPlayed']).where(ds['GamesPlayed'] != 0, 1)

#dataset['AttackIndex'] = dataset['ScoredAttack'] * dataset['PlayedAttack'] / filtered_games_played + dataset['ScoredDefense'] * dataset['PlayedDefense'] / filtered_games_played
#dataset['DefenseIndex'] = dataset['ReceivedAttack'] * dataset['PlayedAttack'] / filtered_games_played + dataset['ScoredDefense'] * dataset['PlayedDefense'] / filtered_games_played

## Desem el dataset a un fitxer netCDF
if season == 'historical':
    ds.to_netcdf('../generated_files/stats_historical.nc', mode='w')
else:
    ds.to_netcdf(f'../generated_files/stats_Season_{season}.nc', mode='w')

#  ds = xr.open_dataset('stats.nc', engine ='netcdf4') # si volem obrir el fitxer

print("Stats created successfully.")

    # A partir de previous_matches faria el recompte de gols anotats per cada jugador i estadístiques similars.
    # ELO s'ha de calcular per cada partit inidividualment, perquè es vagi actualitzant
    #print(previous_matches.groupby('Guanyador').count())
