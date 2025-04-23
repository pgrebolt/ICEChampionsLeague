#!/bin/bash

py download_results.py

jupyter nbconvert --to notebook --execute --inplace create_stats.ipynb
jupyter nbconvert --to notebook --execute --inplace create_teamfrequencies.ipynb
jupyter nbconvert --to notebook --execute --inplace create_standings.ipynb
jupyter nbconvert --to notebook --execute --inplace plot_stats.ipynb
jupyter nbconvert --to notebook --execute --inplace plot_frequencies.ipynb

