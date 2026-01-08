#!/usr/bin/env python
# coding: utf-8

# ## Extracció de freqüències

# Aquest codi pren les dades recopilades dels partits jugats i en fa un fitxer amb els rendiments per equips. Aquest fitxer és una `xarray` de dimensions `player` vs. `teammate`. És a dir, tenim informació de cada possible combinació d'equips.

import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Importem les llibreries
import numpy as np
import pandas as pd
import xarray as xr # per guardar les dades 3D

# Carreguem les dades
season = '5' # 2,3, 4, historical
if season == 'historical':
    data_df = pd.read_csv(f'../generated_files/results_{season}.csv')
else:
    data_df = pd.read_csv(f'../generated_files/results_Season_{season}.csv')

# Emplenem els espais en blanc amb 0
data_df = data_df.fillna(0.)

# Read the number of games played per each player
if season == 'historical':
    dataarray = xr.open_dataset('../generated_files/stats_historical.nc', engine='scipy')
else:
    dataarray = xr.open_dataset('../generated_files/stats.nc', engine='scipy') # cal que quadri la season d'aquest codi amb la del fitxer!

played_games = dataarray['GamesPlayed'].isel(matchday=-1)

# Obtenim una llista amb tots els noms dels participants
players_names = np.unique(data_df[['Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4']].values.flatten())

# Llista de dies jugats
if season == 'historical':
    matchdays = pd.unique(data_df['Total_D'])
else:
    matchdays = pd.unique(data_df['D'])

# Crearem les matrius de `xarray` a partir de `pd.DataFrame()`. Per crear els dataframes, iniciarem els diccionaris buits on hi guardarem les dades per cada jugador. Això serà una de les dimensions de la matriu. Després, per cada partit, escriurem al diccionari el paràmetre corresponent a l'alineació. (ESCRIURE MILLOR AIXÒ)

# DataFrame with the number of team occurrences
mates_df = pd.DataFrame(columns = players_names) # number of games played with each mate
matesplayed_df = pd.DataFrame(columns = players_names) # number of games played with each mate divided by number of games playe by player
winmates_df = pd.DataFrame(columns = players_names) # number of games won by each mate
mates_attack_df = pd.DataFrame(columns = players_names) # number of games played with the mate playing as defender (player in attack)
winmates_attack_df = pd.DataFrame(columns = players_names) # number of games won with the mate playing as defender (player in attack)
closematches_df = pd.DataFrame(columns = players_names) # number of games where the game ended 2-3 or 3-2 (won or lost)
closewins_df = pd.DataFrame(columns = players_names) # number of games won by 1 goal (3-2 or 2-3)
closewinsplayed_df = pd.DataFrame(columns = players_names) # number of games won by 1 goal divided by total number of close matches

# Possible lineups
lineups = [['Jugador 1', 'Jugador 2'], ['Jugador 2', 'Jugador 1'], ['Jugador 3', 'Jugador 4'], ['Jugador 4', 'Jugador 3']]

for player in players_names: # for eack player
    # Initialize a dictionary where the number of team-ups will be stored
    mate_count = {}
    mateplayed_count = {} # number of games played with mate divided by total games played
    winmate_count = {}
    mates_attack_count = {}
    winmates_attack_count = {}
    closematches_count = {}
    closewins_count = {}
    closewinsplayed_count = {}

    # Comptadors a 0
    for mate in players_names:
        mate_count[mate] = 0 # set counter to 0
        mateplayed_count[mate] = 0
        winmate_count[mate] = 0
        mates_attack_count[mate] = 0
        winmates_attack_count[mate] = 0
        closematches_count[mate] = 0
        closewins_count[mate] = 0
        closewinsplayed_count[mate] = 0

    # Calculate number of playing counts and victories
    for lineup in lineups:
        lineup_df = data_df[data_df[lineup[0]] == player] # pick games when player plays in position lineup[0]
        teammate_list = lineup_df[lineup[1]] # pick teammate name when player plays in position lineup[1]
        teammate_list_counts = teammate_list.value_counts() # count how many team occurences (teammate in position lineup[1])

        for mate in teammate_list_counts.keys():
            close_victories = 0 # initialize close victories with this mate and this lineup
            close_losses = 0

            mate_count[mate] = mate_count.get(mate, 0) + teammate_list_counts.loc[mate] # store the teammate join number

            mask_mate = (teammate_list == mate) # mask selecting the games that involve 'mate'

            # For the overall games (regardless of position)
            if lineup in [lineups[0], lineups[1]]: # pick lineups playing as local
                number_victories = (lineup_df['Local'][mask_mate] > lineup_df['Visitant'][mask_mate]).sum() # calculate number of victories

                if number_victories: # if the game was won
                    winmate_count[mate] = winmate_count.get(mate, 0) + number_victories # add number of victories

                    if lineup == lineups[1]: # if the game was won with player playing as attacker
                        winmates_attack_count[mate] = winmates_attack_count.get(mate, 0) + number_victories

                close_victories += ((lineup_df['Local'][mask_mate] - lineup_df['Visitant'][mask_mate]) == 1).sum() # calculate number of close victories (3-2)
                close_losses += ((lineup_df['Visitant'][mask_mate] - lineup_df['Local'][mask_mate]) == 1).sum() # calculate number of close losses (2-3)

                # Això si volem distingir victòries com a local/visitant, o bé atacant/defensor
                #closewins_count[mate] = closewins_count.get(mate, 0) + close_victories # store close victories with this mate
                #closematches_count[mate] = closematches_count.get(mate, 0) + (close_victories + close_losses) # store close victories and losses with this mate

                if lineup == lineups[1]: # if the game lineup had the player as attacker
                    mates_attack_count[mate] = mates_attack_count.get(mate, 0) + teammate_list_counts.loc[mate] # games won being attacker, with mate being defender

            elif lineup in [lineups[2], lineups[3]]: # pick lineups playing as visitor    
                if lineup == lineups[3]: # if the game lineup had the player as attacker
                    mates_attack_count[mate] = mates_attack_count.get(mate, 0) + teammate_list_counts.loc[mate] # store occurrences
                number_victories = (lineup_df['Local'][mask_mate] < lineup_df['Visitant'][mask_mate]).sum() # calculate number of victories
                if number_victories: # if the game was won
                    winmate_count[mate] = winmate_count.get(mate, 0) + number_victories # add number of victories

                    if lineup == lineups[3]: # if the game was won with player playing as attacker
                        winmates_attack_count[mate] = winmates_attack_count.get(mate, 0) + number_victories

                close_victories += ((lineup_df['Visitant'][mask_mate] - lineup_df['Local'][mask_mate]) == 1).sum() # calculate number of close victories (3-2)
                close_losses += ((lineup_df['Local'][mask_mate] - lineup_df['Visitant'][mask_mate]) == 1).sum() # calculate number of close losses (2-3)

            closewins_count[mate] = closewins_count.get(mate, 0) + close_victories # store close victories with this mate
            closematches_count[mate] = closematches_count.get(mate, 0) + (close_victories + close_losses) # store close victories and losses with this mate

            # If we take into acount player position
            #if lineup in [lineups[0], lineups[2]]: # pick lineups playing as defender
 #               number_victories = lineup_df[


    # Divide count by number of games played by player
    for mate in mate_count.keys():
        # si no hi ha hagut close matches, marquem els valors a -99
        if closematches_count[mate] == 0:
            closematches_count[mate] = np.nan
            closewins_count[mate] = np.nan
            closewinsplayed_count[mate] = np.nan
        if mate_count[mate] == 0: # si no s'ha jugat cap partit amb mate
            mate_count[mate] = np.nan
            mateplayed_count[mate] = np.nan
            winmate_count[mate] = np.nan

        if played_games.sel(player = player).values == 0:
            continue
        else:
            mateplayed_count[mate] = mate_count.get(mate, 0) / played_games.sel(player = player).values
            if closematches_count.get(mate, 0) == 0:
                continue
            else:
                closewinsplayed_count[mate] = closewins_count.get(mate, 0) / closematches_count.get(mate, 0)
    # Append this player result to the overall property dataframe
    mates_df = pd.concat([mates_df, pd.DataFrame([mate_count])])
    matesplayed_df = pd.concat([matesplayed_df, pd.DataFrame([mateplayed_count])])
    winmates_df = pd.concat([winmates_df, pd.DataFrame([winmate_count])])
    mates_attack_df = pd.concat([mates_attack_df, pd.DataFrame([mates_attack_count])])
    winmates_attack_df = pd.concat([winmates_attack_df, pd.DataFrame([winmates_attack_count])])
    closematches_df = pd.concat([closematches_df, pd.DataFrame([closematches_count])])
    closewins_df = pd.concat([closewins_df, pd.DataFrame([closewins_count])])
    closewinsplayed_df = pd.concat([closewinsplayed_df, pd.DataFrame([closewinsplayed_count])])

# Transpose to match dimesnions with xarray dimensions (x: player, y: teammate)
matesplayed_df = matesplayed_df.transpose() # necessary if matrix is not symmetric
#closematches_df = closematches_df.transpose()
#closewins_df = closewins_df.transpose()

# Set index to teammate name
mates_df = mates_df.set_index(players_names)
winmates_df = winmates_df.set_index(players_names)
mates_attack_df = mates_attack_df.set_index(players_names)
winmates_attack_df = winmates_attack_df.set_index(players_names)
closematches_df = closematches_df.set_index(players_names)
closewins_df = closewins_df.set_index(players_names)
closewinsplayed_df = closewinsplayed_df.set_index(players_names)

# Convert to float (problems with nan)
#closematches_df = closematches_df.astype(int) # no sé per què em sortia error si no faig aquesta línia
closewins_df = closewins_df.astype(float)
closewinsplayed_df = closewinsplayed_df.astype(float)

# Create win / played ratio for each team (substitute 0 in the denominator by NaN, then recover 0 in the result)
winmatesplayed_df = winmates_df.div(mates_df.replace(0, pd.NA))

# Amb un codi similar, creem una matriu que descriurà els paràmetres cara a cara dels jugadors. Per exemple, quants gols ha marcat cada atacant a cada defensor.

lu = data_df[data_df['Jugador 1'] == 'Pau']
group = lu.groupby(by=['Jugador 1']) # agrupem per defensor
defense_rival_count = group['Jugador 4'].value_counts() # comptem quantes vegades s'ha enfrontat al jugador rival jugant com a defensor rival
#for a in defense_rival_count.index:
#    print(a, a[0], a[1])
#print(defense_rival_count.index)
#print(defense_rival_count['Pau'].loc['Víctor'])
#for opponent_name in defense_rival_count.index:
#    count = defense_rival_count[opponent_name]
#    print(opponent_name, count)

# DataFrame with the number of team occurrences
receivedgoals_attack_defense_df = pd.DataFrame(columns = players_names) # number of goals conceded by each attacking player while playing on defense
receivedgoals_defense_defense_df = pd.DataFrame(columns = players_names)  # number of goals conceded by each defensing player while playing on defense
games_defense_defense_df = pd.DataFrame(columns = players_names)  # number of games by each defensing player while playing on defense
games_attack_defense_df = pd.DataFrame(columns = players_names)  # number of games by each attacking player while playing on defense

# Possible lineups
lineups_defense = [['Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4'], ['Jugador 3', 'Jugador 4', 'Jugador 1', 'Jugador 2']] # (defensa-atac)-(defensa-atac)

for player in players_names: # for eack player
    # Initialize a dictionary where the number of team-ups will be stored
    receivedgoals_attack_defense_count = {}
    receivedgoals_defense_defense_count = {}
    games_defense_defense_count = {}
    games_attack_defense_count = {}

    # Comptadors a 0
    for opponent in players_names:
        receivedgoals_attack_defense_count[opponent] = 0 # set counter to 0
        receivedgoals_defense_defense_count[opponent] = 0
        games_defense_defense_count[opponent] = 0
        games_attack_defense_count[opponent] = 0

    # Calculate number of playing counts and victories
    for lineup in lineups_defense:
        lineup_df = data_df[data_df[lineup[0]] == player] # pick games when player plays in position lineup[0] (as defender)
        #opponent_attack = lineup_df[lineup[3]] # pick list of opponents playing in attack
        #opponent_defense = lineup_df[lineup[2]] # pick list of opponent playing in defense

        # Aquesta comanda ens permet agrupar per dia i jugador en defensa. Després, fem la suma dels gols anotats.
        #group_by_defender = lineup_df.groupby(by=['D', lineup[3]]).sum() # agrupem per dia
        group_by_attack_defender = lineup_df.groupby(by=[lineup[3]]).sum() # sumem per tots els dies (attack-jugador rival; defender-jugador player)
        group_by_defender_defender = lineup_df.groupby(by=[lineup[2]]).sum()
        for opponent_name_attack_defense in group_by_attack_defender.index:
            opponent_attack_goals = group_by_attack_defender['Gols '+lineup[3][-1]].loc[opponent_name_attack_defense]
            receivedgoals_attack_defense_count[opponent_name_attack_defense] = receivedgoals_attack_defense_count.get(opponent_name_attack_defense, 0) + opponent_attack_goals # desem els gols que ha fet aquest atacant
        for opponent_name_defense_defense in group_by_defender_defender.index:
            opponent_defense_goals = group_by_defender_defender['Gols '+lineup[3][-1]].loc[opponent_name_defense_defense]
            receivedgoals_defense_defense_count[opponent_name_defense_defense] = receivedgoals_defense_defense_count.get(opponent_name_defense_defense, 0) + opponent_defense_goals # desem els gols que ha fet aquest atacant

        ## Ara comptarem en quants partits hi ha hagut cada combinació de jugadors
        group = lineup_df.groupby(by=[lineup[0]]) # agrupem per defensor
        defense_rival_count = group[lineup[2]].value_counts() # comptem quantes vegades s'ha enfrontat al jugador rival jugant com a defensor rival
        attack_rival_count = group[lineup[3]].value_counts() # comptem quantes vegades s'ha enfrontat al jugador rival jugant com a atacant rival

        # Guardem els recomptes
        for opponent_name_defense in defense_rival_count.index:
            if isinstance(defense_rival_count.index, pd.MultiIndex):
                opponent = opponent_name_defense[1]
                count = defense_rival_count[player].loc[opponent]
            else:
                # crec que aquí no hi entra mai, que ja està bé
                opponent = opponent_name_defense #si és pd.Index - només hi ha un partit en aquesta posició anotat
                count = defense_rival_count[opponent]

            games_defense_defense_count[opponent] = games_defense_defense_count.get(opponent, 0) + count

        for opponent_name_attack in attack_rival_count.index:
            if isinstance(attack_rival_count.index, pd.MultiIndex):
                opponent = opponent_name_attack[1]
                count = attack_rival_count[player].loc[opponent]
            else:
                opponent = opponent_name_attack #si és pd.Index - només hi ha un partit en aquesta posició anotat
                count = attack_rival_count[opponent]

            games_attack_defense_count[opponent] = games_attack_defense_count.get(opponent, 0) + count


    # Append this player result to the overall property dataframe
    receivedgoals_attack_defense_df = pd.concat([receivedgoals_attack_defense_df, pd.DataFrame([receivedgoals_attack_defense_count])])
    receivedgoals_defense_defense_df = pd.concat([receivedgoals_defense_defense_df, pd.DataFrame([receivedgoals_defense_defense_count])])
    games_defense_defense_df = pd.concat([games_defense_defense_df, pd.DataFrame([games_defense_defense_count])])
    games_attack_defense_df = pd.concat([games_attack_defense_df, pd.DataFrame([games_attack_defense_count])])

receivedgoals_attack_defense_df.index = players_names # l'horitzontal és el jugador atacant. en vertical hi ha d'haver el nom del jugador que defensa
receivedgoals_defense_defense_df.index = players_names # l'horitzontal és el jugador defensor rival. en vertical hi ha el nom del jugador que estem mirant, que està en defensa
games_defense_defense_df.index = players_names # ha de ser una matriu simètrica amb 0 a la diagonal
games_attack_defense_df.index = players_names # no ha de ser una matriu simètria, però sí que ha de tenir 0 a la diagonal

# Convertim els valors dels dataframes a floats
receivedgoals_attack_defense_df = receivedgoals_attack_defense_df.astype(float)
receivedgoals_defense_defense_df = receivedgoals_defense_defense_df.astype(float)
games_defense_defense_df = games_defense_defense_df.astype(float)
games_attack_defense_df = games_attack_defense_df.astype(float)

# TODO: AMPLIAR A QUAN player ÉS ATACANT I MIRAR QUÈ FAN ELS ALTRES JUGADORS (ATACANT I DEFENSOR RIVALS)

# Fem les ràtios entre gols i partits
receivedgoals_games_attack_defense_df = receivedgoals_attack_defense_df.div(games_attack_defense_df)
receivedgoals_games_defense_defense_df = receivedgoals_defense_defense_df.div(games_defense_defense_df)

# Sumem independentment de la posició del rival
games_defense_df = games_attack_defense_df + games_defense_defense_df # partits jugats contra el rival per cada jugador en defensa
receivedgoals_defense_df = receivedgoals_attack_defense_df + receivedgoals_defense_defense_df # gols rebuts contra el rival per cada jugador en defensa
receivedgoals_games_defense_df = receivedgoals_defense_df.div(games_defense_df) # ràtio de gols rebuts per partits jugats contra el rival per cada jugador en defensa


# Adaptem les matrius que hem creat a xarray dataframes.


# Creem una DataArray de xarray. Hi especifiquem els noms de cada dimensió
mates_da = xr.DataArray(mates_df.values, dims = ('teammate', 'player'),
                                      coords = {'teammate': mates_df.index, 'player': mates_df.columns})
matesplayed_da = xr.DataArray(matesplayed_df.values, dims = ('teammate', 'player'),
                                      coords = {'teammate': mates_df.index, 'player': mates_df.columns})
winmates_da = xr.DataArray(winmates_df.values, dims = ('teammate', 'player'),
                                      coords = {'teammate': winmates_df.index, 'player': winmates_df.columns})
winmatesplayed_da = xr.DataArray(winmatesplayed_df.values, dims = ('teammate', 'player'),
                                      coords = {'teammate': winmatesplayed_df.index, 'player': winmatesplayed_df.columns})
closewins_da = xr.DataArray(closewins_df.values, dims = ('teammate', 'player'),
                                      coords = {'teammate': winmatesplayed_df.index, 'player': winmatesplayed_df.columns})
closematches_da = xr.DataArray(closematches_df.values, dims = ('teammate', 'player'),
                                      coords = {'teammate': closematches_df.index, 'player': closematches_df.columns})
closewinsplayed_da = xr.DataArray(closewinsplayed_df.values, dims = ('teammate', 'player'),
                                      coords = {'teammate': closewinsplayed_df.index, 'player': closewinsplayed_df.columns})
receivedgoals_attack_defense_da = xr.DataArray(receivedgoals_attack_defense_df.values, dims = ('defender', 'attacker_rival'),
                                      coords = {'defender': receivedgoals_attack_defense_df.index, 'attacker_rival': receivedgoals_attack_defense_df.columns})
receivedgoals_defense_defense_da = xr.DataArray(receivedgoals_defense_defense_df.values, dims = ('defender', 'defender_rival'),
                                      coords = {'defender': receivedgoals_games_defense_defense_df.index, 'defender_rival': receivedgoals_games_defense_defense_df.columns})
receivedgoals_games_attack_defense_da = xr.DataArray(receivedgoals_games_attack_defense_df.values, dims = ('defender', 'attacker_rival'),
                                      coords = {'defender': receivedgoals_games_attack_defense_df.index, 'attacker_rival': receivedgoals_games_attack_defense_df.columns})
receivedgoals_games_defense_defense_da = xr.DataArray(receivedgoals_games_defense_defense_df.values, dims = ('defender', 'defender_rival'),
                                      coords = {'defender': receivedgoals_games_defense_defense_df.index, 'defender_rival': receivedgoals_games_defense_defense_df.columns})
receivedgoals_defense_da = xr.DataArray(receivedgoals_defense_df.values, dims = ('defender', 'rival'),
                                      coords = {'defender': receivedgoals_defense_df.index, 'rival': receivedgoals_defense_df.columns})
games_defense_da = xr.DataArray(games_defense_df.values, dims = ('defender', 'rival'),
                                      coords = {'defender': games_defense_df.index, 'rival': games_defense_df.columns})
receivedgoals_games_defense_da = xr.DataArray(receivedgoals_games_defense_df.values, dims = ('defender', 'rival'),
                                      coords = {'defender': receivedgoals_games_defense_df.index, 'rival': receivedgoals_games_defense_df.columns})
# Combinem tots els DataArrays a un únic Dataset de xarray (cal que les coords siguin les mateixes per a tots)
dataset = xr.Dataset({"Teammates": mates_da,
                      "TeammatesPlayed": matesplayed_da,
                      "TeammatesWins": winmates_da,
                      "TeammatesWinsPlayed": winmatesplayed_da,
                      "CloseWins": closewins_da,
                      "CloseMatches": closematches_da,
                      "CloseWinsPlayed": closewinsplayed_da,
                      "ReceivedGoalsAttackDefense": receivedgoals_attack_defense_da,
                      "ReceivedGoalsDefenseDefense": receivedgoals_defense_defense_da,
                      "ReceivedGoalsGamesAttackDefense": receivedgoals_games_attack_defense_da,
                      "ReceivedGoalsGamesDefenseDefense": receivedgoals_games_defense_defense_da,
                      "ReceivedGoalsDefense": receivedgoals_defense_da,
                      "GamesDefense": games_defense_da,
                      "ReceivedGoalsGamesDefense": receivedgoals_games_defense_da})

# TODO: el procés de crear el DataArray a partir del DataFrame es pot automatitzar amb una funció que faci un concat al dataframe. 

# Save dataset
if season == 'historical':
    dataset.to_netcdf('../generated_files/teammates_historical.nc', mode='w')
else:
    dataset.to_netcdf('../generated_files/teammates.nc', mode='w')

print("Frequencies saved successfully!")



