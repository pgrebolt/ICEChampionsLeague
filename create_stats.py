#!/usr/bin/env python
# coding: utf-8

# En aquest codi analitzarem els resultats de cada partida i escriurem les classificacions corresponents.

# In[1]:


# Per silenciar un warning vinculat amb el Jupyter Notebook
import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# In[2]:


# Importem les llibreries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr # per guardar les dades 3D
from collections import Counter


# In[3]:


# Definim tab20 com la paleta per defecte dels plots
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)


# In[4]:


# Definim una funció que afegeix un nou matchday al ataframe
def join_matchdays(master_dataframe, dict_to_join):
     # Create dataframe with results of this matchday
    matchday_results = pd.DataFrame(dict_to_join.items()).transpose().reset_index(drop=True) # la llista vertical de resultats per jugador, la passem a fila
    matchday_results.columns = matchday_results.iloc[0] # definim que els noms de la columna són els noms dels jugadors (que surten a la 1a fila)
    matchday_ratio = matchday_results.drop(matchday_results.index[0]) #esborrem la primera fila, que conté els noms del jugadors

    # Agrupem els resultats d'aquesta jornada amb els de les anteriors (columna = nom jugador; fila = matchday)
    master_dataframe = pd.concat([master_dataframe, matchday_ratio], ignore_index=True)

    return master_dataframe


# In[5]:


# Carreguem les dades
data_df = pd.read_csv('results.csv')

# Emplenem els espais en blanc amb 0
data_df = data_df.fillna(0.)


# In[6]:


data_df


# In[7]:


# Obtenim una llista amb tots els noms dels participants
players_names = np.unique(data_df[['Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4']].values.flatten())

# Llista de dies jugats
matchdays = pd.unique(data_df['D'])


# In[8]:


# Comptem quants partits ha jugat cada participant
all_players = data_df['Jugador 1'].tolist() + data_df['Jugador 2'].tolist() + data_df['Jugador 3'].tolist() + data_df['Jugador 4'].tolist()

games_count = dict(Counter(all_players))


# ### Partits jugats i victòries

# In[9]:


# En aquest dataframe hi guardem les estadístiques finals després de cada jornada
played_matchdays = pd.DataFrame(columns=players_names) # jugats total
playedattack_matchdays = pd.DataFrame(columns = players_names) # jugats atac
playeddefense_matchdays = pd.DataFrame(columns = players_names) # jugats defensa
winplayed_matchdays = pd.DataFrame(columns=players_names) # guanyats / jugats
winattackplayed_matchdays = pd.DataFrame(columns=players_names) # guanyats / jugats (atac)
windefenseplayed_matchdays = pd.DataFrame(columns=players_names) # guanyats / jugats (defensa)

for nmatchday in range(len(matchdays)):
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
# Convert to DataFrame for display
#win_counts_df = pd.concat(pd:winDataFrame(list(win_counts.items()), columns=['Player', 'WinCount'])
# played_matchdays
#win_counts_df


# ## ELO rating
# 
# El sistema ELO és un algoritme emprat en compteticions com els escacs per classificar els jugadors en funció del seu nivell. Si un jugador amb alt nivell guanya a un de baix nivell, la seva valoració no canviarà significativament. Ara bé, si és al revés, aleshores la puntuació del d'alt nivell baixarà notablement i la del de baix nivell pujarà bastant.
# 
# Sigui un jugador A amb puntuació $s_A$ i un jugador B amb puntuació $s_B$, aleshores la probabilitat que guanyi A en un enfrontament ve descrita, segons el model, com
# 
# $$P_A(s_A, s_B) = \frac{1}{1 + 10^{(s_B - s_A) /400}}$$
# 
# Si 1 denota victòria i 0 denota derrota, després d'un enfrontament entre A i B on A ha guanyat, les puntuacions s'actualitzen de la següent manera:
# 
# $$ s_A = s_A + K\cdot R_p \cdot (1 - P_A(s_A, s_B))$$
# $$ s_B = s_B + K\cdot R'_p \cdot (0 - P_B(s_B, s_A))$$
# 
# on $K = 30$ és una constant i $R_p$ és una regularització/ponderació que té en compte el rendiment del jugador a la victòria..
# 
# Pel nostre cas, considerarem un ELO en posicions ofensives i un ELO en posicions defensives. Cada jugador comenaça la competició amb 1000 punts en cada posició, i s'anirà actualitzant en funció dels seus resultats a cada partits. Per calcular l'ELO del rival al càlcul, es calcula la mitjana ponderada d'ELOs del rival. La ponderació té en compte el nombre de partits que ha jugat l'atacant i el defensor en llurs posicions. A més, la ponderació $R_p$ dependrà de quants gols hagi anotat/rebut l'atacant/defensor i de si ha guanyat o perdut el partit. Per cada jugador, el rendiment es calcula com:
# 
# $$ R_{p,i} = \frac{r_i}{\sum_k r_k} $$
# 
# on $r_i$ és el rendiment particular de cada jugador i $\sum_k r_k$ representa el rendiment total de l'equip. Així, $R_{p,i}$ està normalitzat a 1. El valor $r_i$ depèn de si el jugador és atacant o defensor:
# 
# $$ r_i (\text{atacant}) =  \mu \cdot \left(\frac{\text{Gols anotats}}{3} \right) + \lambda \cdot \left( 1- \frac{\text{Gols rebuts}}{3} \right) $$
# $$ r_i (\text{defensor}) = \mu \cdot \left( 1 - \frac{\text{Gols rebuts}}{3} \right) + \lambda \cdot \left( \frac{\text{Gols anotats}}{3} \right) $$
# 
# on $\mu = 0.7$ i $\lambda = 0.3$ són dos paràmetres arbitraris que ponderen l'activitat defensiva i ofensiva del defensor.
# 
# Ara bé, tal i com està escrit, si un defensor perd i no ha anotat cap gol tindrà $r_i = 0 \Longrightarrow R_i = 0$ i, per tant, no se li descomptaria cap punt! És per això que, en el cas que l'equip hagi perdut, el rendiment $r_i$ s'ha de recalcular per tal de penalitzar la derrota d'aquesta manera:
# 
# $$ r_i' = 1 - r_i$$
# 
# Així, el rendiment pel cas de la derrota es calcula com $R'_{p,i} = \frac{r'_i}{\sum_k r'_k}$.
# 
# Finalment, cal destacar que en cada càlcul de $r_i$ imposem un rendiment mínim de 0.1. Així, encara que un jugador obtingui una puntuació de $r_i = 0$, aquesta passarà a ser 0.1. Això evita conflictes quan es ponderi a l'hora de calcular $R_{p, i}$.
# 

# In[10]:


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
for nmatchday in range(len(matchdays)):
    matchday_df = data_df.loc[data_df['D'] == nmatchday+1].reset_index()

    for nmatch in range(matchday_df.shape[0]):
        match_df = matchday_df[matchday_df.index == nmatch]

        # Noms dels jugadors
        j1_name, j2_name, j3_name, j4_name = match_df['Jugador 1'].values[0], match_df['Jugador 2'].values[0], match_df['Jugador 3'].values[0], match_df['Jugador 4'].values[0]

        # Nombre de partits jugats com a atacant o com a defensa (el +1 serveix per facilitar el càlcul posterior i indica que en el partit actual també es guanya experiència)
        n1_played = playeddefense_matchdays[playeddefense_matchdays.index == nmatchday][j1_name].values[0] +1
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
            print(r_p3, r_visitant)
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


# In[11]:


elo_defense_matchdays


# ### Gols anotats

# In[12]:


# En aquest dataframe hi guardem les estadístiques finals després de cada jornada
scored_matchdays = pd.DataFrame(columns=players_names) # gols anotats
scoredplayed_matchdays = pd.DataFrame(columns=players_names) #gols anotats / partits jugats
scoredattack_matchdays = pd.DataFrame(columns=players_names) # gols anotats atac
scoreddefense_matchdays = pd.DataFrame(columns=players_names) # gols anotats defensa
scoredattackplayed_matchdays = pd.DataFrame(columns=players_names) # gols anotats atac / partits jugats atac
scoreddefenseplayed_matchdays = pd.DataFrame(columns=players_names) # gols anotats defensa / partits defensa

for nmatchday in range(len(matchdays)):
    # Initialize an empty dictionary to store data
    scored_counts = {}
    scoredplayed_counts = {}
    scoredattack_counts = {}
    scoreddefense_counts = {}
    scoredattackplayed_counts = {}
    scoreddefenseplayed_counts = {}

    for player in players_names: # set all initial wins to 0
        scored_counts[player] = 0
        scoredplayed_counts[player] = 0
        scoredattack_counts[player] = 0
        scoreddefense_counts[player] = 0
        scoredattackplayed_counts[player] = 0
        scoreddefenseplayed_counts[player] = 0

    # Select matchdays
    matchday_df = data_df.loc[data_df['D'] <= nmatchday+1]

    # Select the dataframe index for the last recorded game of this matchday
    # last_matchday_index = matchday_df['D'].index.max()

    # Scorded home defense
    for player in set(matchday_df['Jugador 1'].tolist()):
        scored_counts[player] = scored_counts.get(player, 0) + matchday_df['Gols 1'][matchday_df['Jugador 1'] == player].sum() # socred goals
        scoreddefense_counts[player] = scoreddefense_counts.get(player, 0) + matchday_df['Gols 1'][matchday_df['Jugador 1'] == player].sum() # socred goals
    # Scorded home attack
    for player in set(matchday_df['Jugador 2'].tolist()):
        scored_counts[player] = scored_counts.get(player, 0) + matchday_df['Gols 2'][matchday_df['Jugador 2'] == player].sum()
        scoredattack_counts[player] = scoredattack_counts.get(player, 0) + matchday_df['Gols 2'][matchday_df['Jugador 2'] == player].sum() # socred goals
    # Scorded away defense
    for player in set(matchday_df['Jugador 3'].tolist()):
        scored_counts[player] = scored_counts.get(player, 0) + matchday_df['Gols 3'][matchday_df['Jugador 3'] == player].sum()
        scoreddefense_counts[player] = scoreddefense_counts.get(player, 0) + matchday_df['Gols 3'][matchday_df['Jugador 3'] == player].sum() # socred goals
    # Scorded away attack
    for player in set(matchday_df['Jugador 4'].tolist()):
        scored_counts[player] = scored_counts.get(player, 0) + matchday_df['Gols 4'][matchday_df['Jugador 4'] == player].sum()
        scoredattack_counts[player] = scoredattack_counts.get(player, 0) + matchday_df['Gols 4'][matchday_df['Jugador 4'] == player].sum() # socred goals

    # Ratio scored / played for each player
    for player in set(matchday_df['Jugador 1'].tolist() + matchday_df['Jugador 2'].tolist() + matchday_df['Jugador 3'].tolist() + matchday_df['Jugador 4'].tolist()):   
        if played_matchdays[player].iloc[nmatchday]==0: # if denominator is 0, set value to 0
            scoredplayed_counts[player] = 0
        else: # calculate ratio if denominator is not 0
            scoredplayed_counts[player] = scored_counts.get(player, 0) / played_matchdays[player].iloc[nmatchday] # ratio scored / played
        if playedattack_matchdays[player].iloc[nmatchday] == 0:
            scoredattackplayed_counts[player] = 0
        else:
            scoredattackplayed_counts[player] = scoredattack_counts.get(player, 0) / playedattack_matchdays[player].iloc[nmatchday]
        if playeddefense_matchdays[player].iloc[nmatchday] == 0:
            scoreddefenseplayed_counts[player] = 0
        else:
            scoreddefenseplayed_counts[player] = scoreddefense_counts.get(player, 0) / playeddefense_matchdays[player].iloc[nmatchday]

    # Agrupem els resultats d'aquesta jornada amb els de les anteriors (columna = nom jugador; fila = matchday)
    scored_matchdays = join_matchdays(scored_matchdays, scored_counts)
    scoredplayed_matchdays = join_matchdays(scoredplayed_matchdays, scoredplayed_counts)
    scoredattack_matchdays = join_matchdays(scoredattack_matchdays, scoredattack_counts)
    scoreddefense_matchdays = join_matchdays(scoreddefense_matchdays, scoreddefense_counts)
    scoredattackplayed_matchdays = join_matchdays(scoredattackplayed_matchdays, scoredattackplayed_counts)
    scoreddefenseplayed_matchdays = join_matchdays(scoreddefenseplayed_matchdays, scoreddefenseplayed_counts)


# ### Gols rebuts

# In[13]:


# En aquest dataframe hi guardem les estadístiques finals després de cada jornada
received_matchdays = pd.DataFrame(columns=players_names) # gols rebuts
receivedplayed_matchdays = pd.DataFrame(columns=players_names) # gols rebuts / partit jugat
receivedattack_matchdays = pd.DataFrame(columns=players_names) # gols rebuts atac
receiveddefense_matchdays = pd.DataFrame(columns=players_names) # gols rebuts defensa
receivedattackplayed_matchdays = pd.DataFrame(columns=players_names) # gols rebuts atac / partits jugats atac
receiveddefenseplayed_matchdays = pd.DataFrame(columns=players_names) # gols rebuts defensa / partits jugats defensa

for nmatchday in range(len(matchdays)):
    # Initialize an empty dictionary to store data
    received_counts = {}
    receivedplayed_counts = {}
    receivedattack_counts = {}
    receiveddefense_counts = {}
    receivedattackplayed_counts = {}
    receiveddefenseplayed_counts = {}

    for player in players_names: # set all initial wins to 0
        received_counts[player] = 0
        receivedplayed_counts[player] = 0
        receivedattack_counts[player] = 0
        receiveddefense_counts[player] = 0
        receivedattackplayed_counts[player] = 0
        receiveddefenseplayed_counts[player] = 0

    # Select matchdays
    matchday_df = data_df.loc[data_df['D'] <= nmatchday+1]

    # Select the dataframe index for the last recorded game of this matchday
    # last_matchday_index = matchday_df['D'].index.max()

    # Received home defense
    for player in set(matchday_df['Jugador 1'].tolist()):
        received_counts[player] = scored_counts.get(player, 0) + matchday_df[['Gols 3', 'Gols 4']][matchday_df['Jugador 1'] == player].sum().sum() # socred goals
        receiveddefense_counts[player] = receiveddefense_counts.get(player, 0) + matchday_df[['Gols 3', 'Gols 4']][matchday_df['Jugador 1'] == player].sum().sum() # socred goals
    # Received home attack
    for player in set(matchday_df['Jugador 2'].tolist()):
        received_counts[player] = received_counts.get(player, 0) +  matchday_df[['Gols 3', 'Gols 4']][matchday_df['Jugador 2'] == player].sum().sum() # add to global count
        receivedattack_counts[player] = receivedattack_counts.get(player, 0) +  matchday_df[['Gols 3', 'Gols 4']][matchday_df['Jugador 2'] == player].sum().sum() # socred goals
    # Scorded away defense
    for player in set(matchday_df['Jugador 3'].tolist()):
        received_counts[player] = received_counts.get(player, 0) + matchday_df[['Gols 1', 'Gols 2']][matchday_df['Jugador 3'] == player].sum().sum() #add to global count
        receiveddefense_counts[player] = receiveddefense_counts.get(player, 0) + matchday_df[['Gols 1', 'Gols 2']][matchday_df['Jugador 3'] == player].sum().sum() # socred goals
    # Scorded away attack
    for player in set(matchday_df['Jugador 4'].tolist()):
        received_counts[player] = received_counts.get(player, 0) + matchday_df[['Gols 1', 'Gols 2']][matchday_df['Jugador 4'] == player].sum().sum()
        receivedattack_counts[player] = receivedattack_counts.get(player, 0) + matchday_df[['Gols 1', 'Gols 2']][matchday_df['Jugador 4'] == player].sum().sum() # socred goals

    # Ratio received / played for each player
    for player in set(matchday_df['Jugador 1'].tolist() + matchday_df['Jugador 2'].tolist() + matchday_df['Jugador 3'].tolist() + matchday_df['Jugador 4'].tolist()):   
        if played_matchdays[player].iloc[nmatchday]==0: # if denominator is 0, set value to 0
            receivedplayed_counts[player] = 0
        else: # calculate ratio if denominator is not 0
            receivedplayed_counts[player] = received_counts.get(player, 0) / played_matchdays[player].iloc[nmatchday] # ratio scored / played
        if playedattack_matchdays[player].iloc[nmatchday] == 0:
            receivedattackplayed_counts[player] = 0
        else:
            receivedattackplayed_counts[player] = receivedattack_counts.get(player, 0) / playedattack_matchdays[player].iloc[nmatchday]
        if playeddefense_matchdays[player].iloc[nmatchday] == 0:
            receiveddefenseplayed_counts[player] = 0
        else:
            receiveddefenseplayed_counts[player] = receiveddefense_counts.get(player, 0) / playeddefense_matchdays[player].iloc[nmatchday]

    # Agrupem els resultats d'aquesta jornada amb els de les anteriors (columna = nom jugador; fila = matchday)
    received_matchdays = join_matchdays(received_matchdays, received_counts)
    receivedplayed_matchdays = join_matchdays(receivedplayed_matchdays, receivedplayed_counts)
    receivedattack_matchdays = join_matchdays(receivedattack_matchdays, receivedattack_counts)
    receiveddefense_matchdays = join_matchdays(receiveddefense_matchdays, receiveddefense_counts)
    receivedattackplayed_matchdays = join_matchdays(receivedattackplayed_matchdays, receivedattackplayed_counts)
    receiveddefenseplayed_matchdays = join_matchdays(receiveddefenseplayed_matchdays, receiveddefenseplayed_counts)


# Desem les dades a un xarray. Aquest format permet emmagatzemar matrius 3D, cosa que pandas no ho permet. A la nostra matriu tindrem dimensions (Nom de jugador, Dia de partit, Paràmetre). Això ens permet accedir a l'element que deseitgem.

# In[14]:


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

# dataset['goals'] = goals_da # si volem afegir un nou element
dataset


# Afegim el càlcul dels índexs d'atac i de defensa en base al paràmetres que ja hem calculat:
# 
#     attack_index = goals_attack * games_attack / games_total + goals_defense * games_defense / games_total
# 
# També afegim el càlcul d'ELO total ponderat pel nombre de partits que ha jugat cada jugador a cada posició. Cal tenir en compte si el jugador no ha jugat cap partit. Utilitzem un nombre total de partits fals per fer el recompte. Per evitar divisions per 0, on hi havia un 0 al nombre de partits jugats hi posem un 1. El 0 de la divisió el farà el numerador.

# In[15]:


filtered_games_played = (dataset['GamesPlayed']).where(dataset['GamesPlayed'] != 0, 1)

dataset['AttackIndex'] = dataset['ScoredAttack'] * dataset['PlayedAttack'] / filtered_games_played + dataset['ScoredDefense'] * dataset['PlayedDefense'] / filtered_games_played
dataset['DefenseIndex'] = dataset['ReceivedAttack'] * dataset['PlayedAttack'] / filtered_games_played + dataset['ScoredDefense'] * dataset['PlayedDefense'] / filtered_games_played


# In[16]:


# Weighted ELO a partir del valors normalitzats min-max
normalized_ELO_attack = (dataset['ELOAttack'] - dataset['ELOAttack'].min()) / (dataset['ELOAttack'].max() - dataset['ELOAttack'].min())
normalized_ELO_defense = (dataset['ELODefense'] - dataset['ELODefense'].min()) / (dataset['ELODefense'].max() - dataset['ELODefense'].min())

dataset['WeightedELO'] = normalized_ELO_attack * dataset['PlayedAttack'] / filtered_games_played + normalized_ELO_defense * dataset['PlayedDefense'] / filtered_games_played

# Si algun jugador només ha jugat en una posició, pertorba la normalització min-max. Fem que el seu valor sigui nan
print(dataset['WeightedELO'].max(), dataset['WeightedELO'].min())
dataset['WeightedELO'] = dataset['WeightedELO'].where(dataset['WeightedELO'] < 500)
dataset['WeightedELO']


# In[17]:


print(dataset)


# ## Variables per desar a xarray:
# - Partits jugats
# - Gols anotats
# - Gols rebuts
# - Gols anotats atacant
# - Gols anotats defensor
# - Gols anotats atacant local
# - Gols anotats atacant visitant
# - Gols anotats defensor local
# - Gols anotats defensor visitant
# - Gols rebuts atacant
# - Gols rebuts defensor
# - Gols rebuts atacant local
# - Gols rebuts atacant visitant
# - Gols rebuts defensor local
# - Gols rebuts defensor visitant
# - Un paràmetre d'atacant (que ponderi contra qui s'està jugant)
# - Un paràmetre de defensa (que ponderi contra qui s'està defensant)
# 
# ## Una nova xarray
# - Una nova xarray on s'inclogui la freqüència d'ocurrència de cada parella. La 3a dimensió seria cada jornada, i les dues dimensions principals serien els noms dels jugadors. És la matriu amb 0 a la diagonal que teníem abans.
# - Igual que l'anterior, però amb freqüència de victòries.
# 
# ## Per pensar:
# - Intentar trobar la manera de saber com es pot saber quina parella guanya més partits contra qui.

# Desem el fitxer xarray en format netcdf4. Això ens permetrà obrir-lo amb un altre fitxer i fer-ne l'anàlisi que volguem.

# In[18]:


dataset.to_netcdf('stats.nc', mode='w')
# ds = xr.open_dataset('stats.nc', engine ='netcdf4') # si volem obrir el fitxer

