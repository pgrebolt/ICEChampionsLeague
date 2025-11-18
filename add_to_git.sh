#!/bin/bash

# Add to git
git add results/ELO_stats.png
git add results/frequencies.png
git add results/standings.md
git add results/winplayed_stats.html
git add results/winplayed_stats.png
git add generated_files/results.csv
git add generated_files/stats.nc
git add generated_files/teammates.nc

## Extract matchday number
file="generated_files/results.csv"
last_line=$(tail -n 1 "$file")
matchday="${last_line:0:2}"
echo "Last matchday number is $matchday"

# Commit changes to git
git commit -m "Updated to Matchday $matchday"

read -p "Press Enter to exit.."