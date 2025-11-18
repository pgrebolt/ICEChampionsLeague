#!/bin/bash

.venv\Scripts\activate

py download_results.py

py create_stats.py
py create_teamfrequencies.py
py create_standings.py
py plot_stats.py
py plot_frequencies.py

deactivate

read -p "Press Enter to exit.."
