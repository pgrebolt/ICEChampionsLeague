import requests
import pandas as pd
import numpy as np

# Your sheet's info
sheet_id = "1pN_5Ulttdi5cmj1nlE_vzwZG4n8n_prxm6blmtu5L2U"  # replace with your actual Sheet ID
gid_dict = {'Season 2': '1546699619',
            'Season 3': '1943224824',
            'Season 4': '145438050',
            'Season 5': '2141771608'} # diccionari amb els gid de cada temporada

#gid = "145438050" #Season 4
# gid = "1943224824"  # default is usually 0; change if needed for other tabs
# gid = "1546699619" # Season 2

# Columns to keep
columns = ['D', 'Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4', 'Gols 1', 'Gols 2', 'Gols 3', 'Gols 4', 'Local', 'Visitant', 'Guanyador']

# Columns to convert to int
int_columns = ['D', 'Gols 1', 'Gols 2', 'Gols 3', 'Gols 4', 'Local', 'Visitant']

# Create an empty DataFrame to hold historical results
historical_results = pd.DataFrame()

matchday_offset = 0  # Initialize matchday offset

# Loop through each season and download the corresponding CSV
for season, gid in gid_dict.items():
    # Download URL
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    # Save as CSV
    response = requests.get(csv_url)
    filename = '../generated_files/results_' + season.replace(" ", "_") + '.csv'
    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"CSV downloaded successfully for {season}!")

    # Read the CSV file and remove unnecessary rows
    df = pd.read_csv(filename, usecols=columns)
    df = df.dropna(subset=[df.columns[1]]) # treiem les files innecessàries

    # Passem els floats a int (les columnes numèriques)
    df = df.fillna(-999) # emplenem els nan (espais buits que denoten que el jugador no ha marcat gol) amb -999
    df[int_columns] = df[int_columns].astype(int) # convertim els floats a int
    df = df.replace(-999, 0) # canviem els -999 per 0

    # Guardem el csv un altre cop
    df.to_csv(filename, index=False, sep=',')

    ## Afegir al dataframe històric
    # Afegim una columna de temporada
    df['Season'] = season.split()[-1]  # Només el número de temporada

    # Columna amb el dia de partit des de l'inici del recompte (diferents temporades)
    df['Total_D'] = df['D'] + matchday_offset
    df = df[['Total_D'] + [c for c in df.columns if c != 'Total_D']]  # Reorder columns to have 'Total_D' first (passarà a ser 2a)
    df = df[['Season'] + [c for c in df.columns if c != 'Season']]  # Reorder columns to have 'Season' first

    # Actualitzem el matchday_offset per a la següent temporada
    matchday_offset = matchday_offset + df['D'].iloc[-1]

    # Afegim els resultats de la Season al dataframe històric
    historical_results = pd.concat([historical_results, df], ignore_index=True)

# Guardem els resultats històrics
historical_results.to_csv('../generated_files/results_historical.csv', index=False, sep=',')
