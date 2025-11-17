#!/usr/bin/env python
# coding: utf-8

# Aquest codi llegeix els fitxers amb les dades de les estadístiques dels jugadors i en treu les classificacions. Les classificacions les desa en un fitxer markdown.

# In[1]:


import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# In[2]:


#!pip install numpy==1.25.2
#!pip uninstall xarray scipy netCDF4
#!pip install xarray scipy netCDF4
#!pip install pybin11 --upgrade


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd


# In[4]:


# Definim tab20 com la paleta per defecte dels plots
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)


# In[5]:


# Llegim les dades
dataarray = xr.open_dataset('stats.nc', engine='scipy')
dataarray


# In[6]:


dataarray['WeightedELO']


# In[7]:


# Create a dataaray with the coordinates of the dimension to remove ('player')
minimum_games = 0 # minimum games for player to have played to be considered
last_games_played = dataarray['GamesPlayed'].isel(matchday=-1) # list of the number of games played by each player
mask = xr.DataArray(last_games_played > minimum_games, dims = 'player', coords = {'player':dataarray.player})

# Filter out players
dataarray = dataarray.where(mask, drop=True)


# In[8]:


# Extreiem els noms dels jugadors i les jornades
players_names = dataarray['player'].astype(str).values # noms dels jugadors
matchdays = dataarray['matchday'] # array de números de jornades


# In[9]:


# Paràmetres que volem posar a la taula, per ordre d'aparició
parameters = ['WinPlayed', 'ScoredPlayed', 'ELOAttack', 'ELODefense', 'WeightedELO']


# In[10]:


# Creem un fitxer on hi desarem les taules
md_file = open('results/standings.md', 'w')

# Escrivim cada classificació al fitxer
for parameter in parameters:
    # Extreiem els valors de l'última jornada
    values = dataarray[parameter].isel(matchday = -1).values

    # Ordenem de major a menor
    values_sorted_idx = np.argsort(values)[::-1] # índexs d'ordre (revertim per fer de major a menor)
    values_sorted = values[values_sorted_idx] # ordenem
    values_sorted = np.round(values_sorted, 2) # arrondonim els valors a 2 xifres decimals

    players_names_sorted = players_names[values_sorted_idx] # ordenem els noms dels jugadors

    #Posem els resultats a una taula (DataFrame)
    standings = pd.DataFrame(np.array([players_names_sorted, values_sorted]).T, index = np.arange(1, players_names_sorted.shape[0]+1), columns = ['Player', parameter])
    #standings = standings.style.set_caption(parameters) # afegim títol al dataframe

    # Guardem la taula en Markdown
    standings_md = standings.to_markdown()

    # Escrivim la taula al fitxer
    md_file.write(standings_md)
    md_file.write('\n\n')

md_file.close()

