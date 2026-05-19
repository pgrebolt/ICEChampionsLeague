#!/bin/bash

# Add to git
git add results/Season*/frequencies_Season_*.png
git add results/Season*/ELO_stats_Season_*.png
git add results/Season*/scatter_plotsSeason_*.png
git add results/standings.md
git add results/winplayed_stats.html
git add results/Season*/winplayed_statsSeason_*.html
git add results/Season*/winplayed_stats_Season_*.png
git add generated_files/results_Season_*.csv
git add generated_files/stats_Season_*.nc
git add generated_files/teammates_Season_*.nc
git add generated_files/results_historical.csv
git add generated_files/stats_historical.nc
git add generated_files/teammates_historical.nc
git add results/historical/


## Extract matchday number
file="generated_files/results_Season_6.csv"
last_line=$(tail -n 1 "$file")
matchday="${last_line:0:2}"
echo "Last matchday number is $matchday"

# Commit changes to git
git commit -m "Updated to Matchday $matchday"

read -p "Press Enter to exit.."