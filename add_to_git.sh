#!/bin/bash

# Add to git
git add src\create_standings.ipynb
git add src\create_stats.ipynb 
git add src\create_teamfrequencies.ipynb
git add src\plot_frequencies.ipynb
git add src\plot_stats.ipynb
git add results/ELO_stats.png
git add results.csv
git add results/frequencies.png
git add results/standings.md
git add results/winplayed_stats.html
git add results/winplayed_stats.png
git add stats.nc
git add teammates.nc

## Extract matchday number
file="results.csv"
last_line=$(tail -n 1 "$file")
matchday="${last_line:0:2}"
echo "Last matchday number is $matchday"

# Commit changes to git
git commit -m "Updated to Matchday $matchday"

read -p "Press Enter to exit.."