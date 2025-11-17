import requests
import pandas as pd
import numpy as np

# Your sheet's info
sheet_id = "1pN_5Ulttdi5cmj1nlE_vzwZG4n8n_prxm6blmtu5L2U"  # replace with your actual Sheet ID
gid = "145438050"
# gid = "1943224824"  # default is usually 0; change if needed for other tabs
# gid = "1546699619" # Season 2

# Download URL
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

# Save as CSV
response = requests.get(csv_url)
with open("results.csv", "wb") as f:
    f.write(response.content)

print("CSV downloaded successfully!")

columns = ['D', 'Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4', 'Gols 1', 'Gols 2', 'Gols 3', 'Gols 4', 'Local', 'Visitant', 'Guanyador']

# Read the CSV file and remove unnecessary rows
df = pd.read_csv("results.csv", usecols=columns)
df = df.dropna(subset=[df.columns[1]]) # treiem les files innecessàries

# Passem els floats a int (les columnes numèriques)
df = df.fillna(-999) # emplenem els nan (espais buits que denoten que el jugador no ha marcat gol) amb -999
int_columns = ['D', 'Gols 1', 'Gols 2', 'Gols 3', 'Gols 4', 'Local', 'Visitant']
df[int_columns] = df[int_columns].astype(int) # convertim els floats a int
df = df.replace(-999, 0) # canviem els -999 per 0

# Guardem el csv un altre cop
df.to_csv("results.csv", index=False, sep=',')
