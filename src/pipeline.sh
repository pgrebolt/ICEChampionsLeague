#!/bin/bash

# Run this pipeline from the command line as: pipeline.sh -s 6
# where '6' is the season number. If no season is provided, it defaults to 'historical'.

### PARSER
# Default value
season="historical"
#Parse command-line options
while getopts "s:" opt; do
  case $opt in
    s) season="$OPTARG" ;;
  esac
done

echo "Running pipeline for season: $season"

.venv\Scripts\activate

py download_results.py

py create_stats_match.py --season "$season"
py create_teamfrequencies.py --season "$season"
py create_standings.py --season "$season"
py plot_stats.py --season "$season"
py plot_frequencies.py --season "$season"

deactivate

read -p "Press Enter to exit.."
