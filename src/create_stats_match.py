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

# Definim tab20 com la paleta per defecte dels plots
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)

# Definim una funció que afegeix un nou matchday al ataframe
def join_matchdays(master_dataframe, dict_to_join):
     # Create dataframe with results of this matchday
    matchday_results = pd.DataFrame(dict_to_join.items()).transpose().reset_index(drop=True) # la llista vertical de resultats per jugador, la passem a fila
    matchday_results.columns = matchday_results.iloc[0] # definim que els noms de la columna són els noms dels jugadors (que surten a la 1a fila)
    matchday_ratio = matchday_results.drop(matchday_results.index[0]) #esborrem la primera fila, que conté els noms del jugadors

    # Agrupem els resultats d'aquesta jornada amb els de les anteriors (columna = nom jugador; fila = matchday)
    master_dataframe = pd.concat([master_dataframe, matchday_ratio], ignore_index=True)

    return master_dataframe

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
    "ReceivedAttackPlayed", "ReceivedDefensePlayed"
#    "ELOAttack", "ELODefense"
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

# TODO:CREAR UN ÚNIC DATAFRAME AMB TOTES LES KEYS, QUE ES VAGI ACTUALITZANT. No puc sumar diccionaris
playeddefense = {player: 0 for player in players_names}
playedattack = {player: 0 for player in players_names}
played_j1 = {player: 0 for player in players_names}
played_j2 = {player: 0 for player in players_names}
played_j3 = {player: 0 for player in players_names}
played_j4 = {player: 0 for player in players_names}

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


    ## Càlcul d'ELO
    #TODO: fer el càlcul d'elo amb una funció que es cridarà aquí

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
    #_acc["ELOAttack"].append(_dict_to_row(elo_rating_attack, players_names))
    #_acc["ELODefense"].append(_dict_to_row(elo_rating_defense, players_names))

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
print(ds['GamesPlayed'].sel(match=matches, player='Pau').max())
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

for nmatchday in range(len(matchdays)): #AIXÒ INCLOURE DINS L'ANTERIOR LOOP (partit a partit. posar una marca quan es canvii de matchday)
    # Initialize an empty dictionary to store data
    played_counts = {}
    winplayed_counts = {}
    playedattack_counts = {}
    playeddefense_counts = {}
    win_counts = {}
    winattack_counts = {}
    windefense_counts = {}
    winattackplayed_counts = {}
    windefenseplayed_counts = {}

    for player in players_names: # set all initial wins to 0
        win_counts[player] = 0

    # Select matchdays
    if season == 'historical':
        matchday_df = data_df.loc[data_df['Total_D'] <= nmatchday+1]
    else:
        matchday_df = data_df.loc[data_df['D'] <= nmatchday+1]

    # Home wins
    home_wins = matchday_df[matchday_df['Local'] > matchday_df['Visitant']]
    for player in home_wins['Jugador 1'].tolist() + home_wins['Jugador 2'].tolist(): #pick from list of all winners
        win_counts[player] = win_counts.get(player, 0) + 1

    # Away wins
    away_wins = matchday_df[matchday_df['Visitant'] > matchday_df['Local']]
    for player in away_wins['Jugador 3'].tolist() + away_wins['Jugador 4'].tolist():
        win_counts[player] = win_counts.get(player, 0) + 1

    # Attack wins
    for player in home_wins['Jugador 2'].tolist() + away_wins['Jugador 4'].tolist():
        winattack_counts[player] = winattack_counts.get(player, 0) + 1

    # Defense wins 
    for player in home_wins['Jugador 1'].tolist() + away_wins['Jugador 3'].tolist():
        windefense_counts[player] = windefense_counts.get(player, 0) + 1

    # Games played
    for player in players_names:
        # Comptem quantes vegades el nom del jugador apareix al registre de partits
        games_played = (matchday_df[['Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4']] == player).sum().sum()
        games_playedattack = (matchday_df[['Jugador 2', 'Jugador 4']] == player).sum().sum()
        games_playeddefense = (matchday_df[['Jugador 1', 'Jugador 3']] == player).sum().sum()

        # Desem a un diccionari el recompte de partits jugats
        played_counts[player] = games_played
        playedattack_counts[player] = games_playedattack
        playeddefense_counts[player] = games_playeddefense

        # Desem a un diccionari la ràtio entre partits guanyats i partits jugats
        if games_played == 0:
            winplayed_counts[player] = 0
        else:
            winplayed_counts[player] = win_counts.get(player, 0) / games_played
        if games_playedattack == 0:
            winattackplayed_counts[player] = 0
        else:
            winattackplayed_counts[player] = winattack_counts.get(player, 0) / games_playedattack
        if games_playeddefense == 0:
            windefenseplayed_counts[player] = 0
        else:
            windefenseplayed_counts[player] = windefense_counts.get(player, 0) / games_playeddefense
        #print(player)
        #win_counts[player] = win_counts.get(player, 0) / 

    # Agrupem els resultats d'aquesta jornada amb els de les anteriors (columna = nom jugador; fila = matchday)
    winplayed_matchdays = join_matchdays(winplayed_matchdays, winplayed_counts)
    played_matchdays = join_matchdays(played_matchdays, played_counts)
    playedattack_matchdays = join_matchdays(playedattack_matchdays, playedattack_counts)
    playeddefense_matchdays = join_matchdays(playeddefense_matchdays, playeddefense_counts)
    winattackplayed_matchdays = join_matchdays(winattackplayed_matchdays, winattackplayed_counts)
    windefenseplayed_matchdays = join_matchdays(windefenseplayed_matchdays, windefenseplayed_counts)

# DataFrames on hi guardarem els valors ELO a cada jornada
elo_attack_matchdays = pd.DataFrame(columns = players_names)
elo_defense_matchdays = pd.DataFrame(columns = players_names)

#ELO rating inicial
elo_rating_attack = {}
elo_rating_defense = {}
for player in players_names:
    elo_rating_attack[player] = 1000.
    elo_rating_defense[player] = 1000.

K = 30. # ELO update constant

# Actualitzem els ELO a cada partit
for nmatch in range(matches):

    match_df = data_df.iloc[nmatch].reset_index()

    # Noms dels jugadors
    j1_name, j2_name, j3_name, j4_name = match_df['Jugador 1'].values[0], match_df['Jugador 2'].values[0], match_df['Jugador 3'].values[0], match_df['Jugador 4'].values[0]

    # Nombre de partits jugats com a atacant o com a defensa (el +1 serveix per facilitar el càlcul posterior i indica que en el partit actual també es guanya experiència)
    n1_played = playeddefense_matches.iloc[nmatch][j1_name].values[0] +1 # TODO: fer que playeddefense_matches tingui una columna d'índex de partit, per poder quadrar ara (igual pels altres dataframes)
    n2_played = playedattack_matchdays[playedattack_matchdays.index == nmatchday][j2_name].values[0] +1
    n3_played = playeddefense_matchdays[playeddefense_matchdays.index == nmatchday][j3_name].values[0] +1
    n4_played = playedattack_matchdays[playedattack_matchdays.index == nmatchday][j4_name].values[0] +1

    # ELO mitjà per equip
    ELO_local = (elo_rating_defense[j1_name]*n1_played + elo_rating_attack[j2_name]*n2_played) / (n1_played + n2_played)
    ELO_visitant = (elo_rating_defense[j3_name]*n3_played + elo_rating_attack[j4_name]*n4_played) / (n3_played + n4_played)

    # Probabilitats de victòria
    P_local = 1 / (1 + 10**((ELO_visitant - ELO_local) / 400))
    P_visitant = 1 / (1 + 10**((ELO_local - ELO_visitant) / 400))

    # Outcome segons local (1 si guanya local, 0 si guanya visitant)
    guanyador = match_df['Guanyador'].values[0]
    outcome_local = 1 if (guanyador=='Local') else 0
    outcome_visitant = 1 if (guanyador=='Visitant') else 0

    # Paràmetres de ponderació
    mu, lamb = 0.3, 0.7

    # Gols que ha rebut cada equip
    gols_rebuts_local = match_df['Gols 3'].values[0] + match_df['Gols 4'].values[0]
    gols_rebuts_visitant = match_df['Gols 1'].values[0] + match_df['Gols 2'].values[0]

    # Rendiment sobre la contribució al partit segons posició
    r_p1 = lamb * (1 - gols_rebuts_local / 3) + mu * (match_df['Gols 1'].values[0] / 3) # defensa
    r_p2 = mu * (1 - gols_rebuts_local / 3) + lamb * (match_df['Gols 2'].values[0] / 3) # atac
    r_p3 = lamb * (1 - gols_rebuts_visitant / 3) + mu * (match_df['Gols 3'].values[0] / 3)  # defensa
    r_p4 = mu * (1 - gols_rebuts_visitant / 3) + lamb * (match_df['Gols 4'].values[0] / 3) # atac

    # Imposem un rendiment mínim de 0.1
    r_p1 = max(r_p1, 0.1)
    r_p2 = max(r_p2, 0.1)
    r_p3 = max(r_p3, 0.1)
    r_p4 = max(r_p4, 0.1)

    # Suma dels rendiments
    r_local = r_p1 + r_p2
    r_visitant = r_p3 + r_p4

    # Actualitzem la ponderació de rendiment pels perdedors
    if guanyador == 'Local':
        r_p3 = 1 - r_p3
        r_p4 = 1 - r_p4

        # Imposem el rendiment mínim
        r_p3 = min(r_p3, 0.9)
        r_p4 = min(r_p4, 0.9)

        r_visitant = r_p3 + r_p4
    elif guanyador == 'Visitant':
        r_p1 = 1 - r_p1
        r_p2 = 1 - r_p2

        # Imposem el rendiment mínim
        r_p1 = min(r_p1, 0.9)
        r_p2 = min(r_p2, 0.9)

        r_local = r_p1 + r_p2

    # Actualitzem ELOS
    elo_rating_defense[j1_name] = elo_rating_defense[j1_name] + K * (outcome_local - P_local) * r_p1 / r_local
    elo_rating_attack[j2_name] = elo_rating_attack[j2_name] + K * (outcome_local - P_local) * r_p2 / r_local
    elo_rating_defense[j3_name] = elo_rating_defense[j3_name] + K * (outcome_visitant - P_visitant) * r_p3 / r_visitant
    elo_rating_attack[j4_name] = elo_rating_attack[j4_name] + K * (outcome_visitant - P_visitant) * r_p4 / r_visitant

    # Afegim l'ELO actualitzat d'aquest matchday
    elo_attack_matchdays = join_matchdays(elo_attack_matchdays, elo_rating_attack)
    elo_defense_matchdays = join_matchdays(elo_defense_matchdays, elo_rating_defense)        



# Desem les dades a un xarray. Aquest format permet emmagatzemar matrius 3D, cosa que pandas no ho permet. A la nostra matriu tindrem dimensions (Nom de jugador, Dia de partit, Paràmetre). Això ens permet accedir a l'element que deseitgem.

# Creem una DataArray de xarray. Hi especifiquem els noms de cada dimensió
winplayed_matchdays_da = xr.DataArray(winplayed_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': winplayed_matchdays.index, 'player': winplayed_matchdays.columns})
played_matchdays_da = xr.DataArray(played_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': played_matchdays.index, 'player': played_matchdays.columns})
playedattack_matchdays_da = xr.DataArray(playedattack_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': playedattack_matchdays.index, 'player': playedattack_matchdays.columns})
playeddefense_matchdays_da = xr.DataArray(playeddefense_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': playeddefense_matchdays.index, 'player': playedattack_matchdays.columns})
winattackplayed_matchdays_da = xr.DataArray(winattackplayed_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': winattackplayed_matchdays.index, 'player': winattackplayed_matchdays.columns})
windefenseplayed_matchdays_da = xr.DataArray(windefenseplayed_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': windefenseplayed_matchdays.index, 'player': windefenseplayed_matchdays.columns})
scored_matchdays_da = xr.DataArray(scored_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': scored_matchdays.index, 'player': scored_matchdays.columns})
scoredplayed_matchdays_da = xr.DataArray(scoredplayed_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': scoredplayed_matchdays.index, 'player': scoredplayed_matchdays.columns})
scoredattack_matchdays_da = xr.DataArray(scoredattack_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': scoredattack_matchdays.index, 'player': scoredattack_matchdays.columns})
scoreddefense_matchdays_da = xr.DataArray(scoreddefense_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': scoreddefense_matchdays.index, 'player': scoreddefense_matchdays.columns})
scoredattackplayed_matchdays_da = xr.DataArray(scoredattackplayed_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': scoredplayed_matchdays.index, 'player': scoredplayed_matchdays.columns})
scoreddefenseplayed_matchdays_da = xr.DataArray(scoreddefenseplayed_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': scoredplayed_matchdays.index, 'player': scoredplayed_matchdays.columns})
received_matchdays_da = xr.DataArray(received_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': received_matchdays.index, 'player': received_matchdays.columns})
receivedplayed_matchdays_da = xr.DataArray(receivedplayed_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': receivedplayed_matchdays.index, 'player': receivedplayed_matchdays.columns})
receivedattack_matchdays_da = xr.DataArray(receivedattack_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': receivedattack_matchdays.index, 'player': receivedattack_matchdays.columns})
receiveddefense_matchdays_da = xr.DataArray(receiveddefense_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': receiveddefense_matchdays.index, 'player': receiveddefense_matchdays.columns})
receivedattackplayed_matchdays_da = xr.DataArray(receivedattackplayed_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': receivedplayed_matchdays.index, 'player': receivedplayed_matchdays.columns})
receiveddefenseplayed_matchdays_da = xr.DataArray(receiveddefenseplayed_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': receivedplayed_matchdays.index, 'player': receivedplayed_matchdays.columns})
elo_attack_matchdays_da = xr.DataArray(elo_attack_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': elo_attack_matchdays.index, 'player': elo_attack_matchdays.columns})
elo_defense_matchdays_da = xr.DataArray(elo_defense_matchdays.values, dims = ('matchday', 'player'),
                                      coords = {'matchday': elo_defense_matchdays.index, 'player': elo_defense_matchdays.columns})

# Combinem tots els DataArrays a un únic Dataset de xarray (cal que les coords siguin les mateixes per a tots)
dataset = xr.Dataset({"GamesPlayed": played_matchdays_da,
                      "PlayedAttack": playedattack_matchdays_da,
                      "PlayedDefense": playeddefense_matchdays_da,
                      "WinPlayed": winplayed_matchdays_da,
                      "WinPlayedAttack": winattackplayed_matchdays_da,
                      "WinPlayedDefense": windefenseplayed_matchdays_da,
                      "Scored": scored_matchdays_da,
                      "ScoredPlayed": scoredplayed_matchdays_da,
                      "ScoredAttack": scoredattack_matchdays_da,
                      "ScoredDefense": scoreddefense_matchdays_da,
                      "ScoredAttackPlayed": scoredattackplayed_matchdays_da,
                      "ScoredDefensePlayed": scoreddefenseplayed_matchdays_da,
                      "Received": received_matchdays_da,
                      "ReceivedPlayed": receivedplayed_matchdays_da,
                      "ReceivedAttack": receivedattack_matchdays_da,
                      "ReceivedDefense": receiveddefense_matchdays_da,
                      "ReceivedAttackPlayed": receivedattackplayed_matchdays_da,
                      "ReceivedDefensePlayed": receiveddefenseplayed_matchdays_da,
                      "ELOAttack": elo_attack_matchdays_da,
                      "ELODefense": elo_defense_matchdays_da})

# TODO: el procés de crear el DataArray a partir del DataFrame es pot automatitzar amb una funció que faci un concat al dataframe. 


# Afegim el càlcul dels índexs d'atac i de defensa en base al paràmetres que ja hem calculat:
# 
#     attack_index = goals_attack * games_attack / games_total + goals_defense * games_defense / games_total
# 
# També afegim el càlcul d'ELO total ponderat pel nombre de partits que ha jugat cada jugador a cada posició. Cal tenir en compte si el jugador no ha jugat cap partit. Utilitzem un nombre total de partits fals per fer el recompte. Per evitar divisions per 0, on hi havia un 0 al nombre de partits jugats hi posem un 1. El 0 de la divisió el farà el numerador.

filtered_games_played = (dataset['GamesPlayed']).where(dataset['GamesPlayed'] != 0, 1)

dataset['AttackIndex'] = dataset['ScoredAttack'] * dataset['PlayedAttack'] / filtered_games_played + dataset['ScoredDefense'] * dataset['PlayedDefense'] / filtered_games_played
dataset['DefenseIndex'] = dataset['ReceivedAttack'] * dataset['PlayedAttack'] / filtered_games_played + dataset['ScoredDefense'] * dataset['PlayedDefense'] / filtered_games_played

# Weighted ELO a partir del valors normalitzats min-max
normalized_ELO_attack = (dataset['ELOAttack'] - dataset['ELOAttack'].min()) / (dataset['ELOAttack'].max() - dataset['ELOAttack'].min())
normalized_ELO_defense = (dataset['ELODefense'] - dataset['ELODefense'].min()) / (dataset['ELODefense'].max() - dataset['ELODefense'].min())

dataset['WeightedELO'] = normalized_ELO_attack * dataset['PlayedAttack'] / filtered_games_played + normalized_ELO_defense * dataset['PlayedDefense'] / filtered_games_played

# Si algun jugador només ha jugat en una posició, pertorba la normalització min-max. Fem que el seu valor sigui nan
#print(dataset['WeightedELO'].max(), dataset['WeightedELO'].min())
dataset['WeightedELO'] = dataset['WeightedELO'].where(dataset['WeightedELO'] < 500)
dataset['WeightedELO']


