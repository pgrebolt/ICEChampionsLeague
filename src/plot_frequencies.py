#!/usr/bin/env python
# coding: utf-8
## Frequencies plotter"
"In this notebook we load the stored data of frequencies between teams and plot them. The basic structure of the code is to read and then to plot."
"First, we import the required libraries."
import asyncio
import sys
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

"Read the stored NetCDF4 file, where the data is stored."
# Llegim les dades
# Carreguem les dades
season = '5' # 2,3, 4, historical
if season == 'historical':
    dataarray = xr.open_dataset(f'../generated_files/teammates_{season}.nc', engine='scipy') # fitxer amb les dades de les estadístiques
else:
    dataarray = xr.open_dataset(f'../generated_files/teammates_Season_{season}.nc', engine='scipy')

"Store the list of all the players."
# List of all player names (ordered)"
players_names = dataarray.player.values

"Define a function that will plot the heatmap of each parameter."
# Function to plot the heatmaps
def heatmap_frequencies(matrix, ax, integer=False, cmap = 'YlGn'):
    # Plot heatmap
    if cmap == 'RdYlGn':
        ax.imshow(matrix, cmap = cmap, aspect = 'equal', vmin = 0, vmax = 1)
    else:
        ax.imshow(matrix, cmap = cmap, aspect = 'equal')

    # Show the values at each cell",
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i,j] # value at that cell
            if np.isnan(value):
                continue
            else:
                #value = int(value) if int==True else value # convert value to int if needed
                if integer==True: # if values are float
                    ax.text(j, i, "{:d}".format(int(value)), ha='center', va='center', fontsize=7, color="black")
                else: # if values are other (int)
                    ax.text(j, i, "{:.2f}".format(value), ha='center', va='center', fontsize=7, color="black")

"Plotting. First, select the parameters to plot, and store their corresponding numpy arrays (matrices). We filter the 0 values in the ratio between wins and played games for plotting purposes. Then, initialize the figure and plot each colormap using the previouly-defined function."

# Read the parameters to plot
teammates = dataarray['Teammates'].values # number of games played with each teammate
teammates_wins = dataarray['TeammatesWins'].values # number of games won with each teammate
teammatesplayed = dataarray['TeammatesPlayed'].values # percentage of games played with each mate wrt the total games played by player
teammatesplayed_wins = dataarray['TeammatesWinsPlayed'].values # percentage of games played with each mate wrt the total games played by player
teammatesclosematches = dataarray['CloseMatches'].values # number of close matches played with each teammate
teammatesclosewins_played = dataarray['CloseWinsPlayed'].values # relative number of close matches won with each teammate
receivedgoals_games_attack_defense = dataarray['ReceivedGoalsGamesAttackDefense'].values # received goals per game by each attacker when playing on defense
receivedgoals_games_defense_defense = dataarray['ReceivedGoalsGamesDefenseDefense'].values # received goals per game by each defender when playing on defense
receivedgoals_games_defense = dataarray['ReceivedGoalsGamesDefense'].values # received goals per game by each player when playing on defense

# Convert values to int while keeping NaNs in selected parameters
teammates = np.where(np.isnan(teammates), np.nan, teammates.astype(int))
teammates_wins = np.where(np.isnan(teammates_wins), np.nan, teammates_wins.astype(int))
teammatesclosematches = np.where(np.isnan(teammatesclosematches), np.nan, teammatesclosematches.astype(int))

# Initialize figure
fig, axs = plt.subplots(figsize=(16,16*5/2), ncols = 2, nrows = 5)
plt.subplots_adjust(wspace=0.)
axs = axs.flatten()

# Plot each parameter
heatmap_frequencies(teammates, axs[0], integer=True)
heatmap_frequencies(teammates_wins, axs[1], integer = True)
heatmap_frequencies(teammatesplayed, axs[2])
heatmap_frequencies(teammatesplayed_wins, axs[3], cmap = 'RdYlGn') # la parella més efectiva és la que té aquest valor més alt
heatmap_frequencies(teammatesclosematches, axs[4], integer=True) # la parella més efectiva és la que té aquest valor més alt
heatmap_frequencies(teammatesclosewins_played, axs[5], cmap = 'RdYlGn') # la parella més efectiva és la que té aquest valor més alt
heatmap_frequencies(receivedgoals_games_attack_defense, axs[6], cmap='RdYlGn_r') # TODO: CANVIAR COLORMAP
heatmap_frequencies(receivedgoals_games_defense_defense, axs[7], cmap='RdYlGn_r')
heatmap_frequencies(receivedgoals_games_defense, axs[8], cmap='RdYlGn_r')

# Add tick labels
axs[0].set_title("Games played")
axs[1].set_title("Games won")
axs[2].set_title("Games played with mate / Total games played by player")
axs[3].set_title("Games won with mate / Total games played with mate")
axs[4].set_title("Close games")
axs[5].set_title("Close wins / Close games")
axs[6].set_title("Goals scored ratio to each defender (attacker)")
axs[7].set_title("Goals scored ratio to each defender (defender)")
axs[8].set_title("Goals scored ratio to each defender (any position)")

for i in range(9):
    axs[i].set_xticks(ticks=np.arange(len(players_names)))
    axs[i].set_xticklabels(players_names, rotation=90, fontsize=8)
    axs[i].set_yticks(ticks=np.arange(len(players_names)))
    axs[i].set_yticklabels(players_names, rotation=0, fontsize=8)

    if i < 6:
        ylabel = "Team mate"
        xlabel = "Player"
    else:
        ylabel = "Defender"
        if i == 6:
            xlabel = "Rival attacker"
        elif i == 7:
            xlabel = "Rival defender"
        elif i == 8:
            xlabel = "Rival (any position)"

    axs[i].set_ylabel(ylabel)
    axs[i].set_xlabel(xlabel)
plt.savefig('../results/frequencies.png', dpi=300, bbox_inches='tight')
dataarray.close()
