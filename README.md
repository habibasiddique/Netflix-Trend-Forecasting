# AUSPiFY Task 5 (Advanced): Netflix Trend Forecasting Model

## What it does
Loads Netflix titles data, counts titles by release year, compares Linear Regression and Random Forest, evaluates them using a chronological holdout split, and forecasts the next five years.

## Run in VS Code
1. Extract this ZIP and open the `Netflix_Trend_Forecasting_Task5` folder in VS Code.
2. Open Terminal and run:
   `python -m venv .venv`
   `.venv\\Scripts\\Activate.ps1`
   `pip install -r requirements.txt`
3. Download the Netflix Movies and TV Shows CSV from https://www.kaggle.com/datasets/shivamb/netflix-shows
4. Put `netflix_titles.csv` inside `data/`.
5. Run `python main.py`.

If the real CSV is absent, the script uses `data/demo_netflix_titles.csv` for testing. Demo data is illustrative only; use the real dataset for assignment results.

## Outputs
The script creates historical trend, model evaluation, and future forecast charts plus CSV metrics in `outputs/`.

## Limitation
This is an exploratory forecast based on release-year patterns, not a guarantee of Netflix's future decisions.
