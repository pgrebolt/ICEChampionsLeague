#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import preprocessing
import joblib
from sklearn.datasets import (
  make_classification,
  make_regression
)
from sklearn.ensemble import (
  BaggingClassifier,
  BaggingRegressor,
  RandomForestClassifier,
  RandomForestRegressor,
  StackingClassifier,
  StackingRegressor
)
from sklearn.linear_model import (
  LogisticRegression,
  LinearRegression
)
from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  roc_auc_score,
  confusion_matrix,
  mean_squared_error,
  r2_score,
  mean_absolute_error
)
from sklearn.model_selection import (
  train_test_split,
  cross_val_score,
  KFold,
  StratifiedKFold
)
from sklearn.neighbors import (
  KNeighborsClassifier,
  KNeighborsRegressor
)
from sklearn.preprocessing import (
  StandardScaler,
  LabelEncoder
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# set visualisation style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# set random seed for reproducibility
RND_STATE = None
np.random.seed(RND_STATE)

''' Aquest codi utilitza un model XGBoost per predir els resultats dels partits a partir de les estadístiques dels jugadors. '''

# Per poder entrenar el model, ens cal donar-li informació de qui està jugant (amb quins punts forts i punts febles) i quin és el resultat. Per tant, hem de llegir el fitxer amb les estadístiques de cada jugador i també el fitxer amb el registre històric de resultats.

# Llegim l'històric de partits i resultats
matches_df = pd.read_csv('../generated_files/results_historical.csv')

# Llegim les estadístiques de cada jugador
stats_xr = xr.open_dataset('../generated_files/stats_historical.nc', engine='scipy')
print("Individual available parameters:")
print(list(stats_xr.keys()))

# Llegim les estadístiques creuades (com un jugador es comporta contra un altre)
frequencies_xr = xr.open_dataset('../generated_files/teammates_historical.nc', engine='scipy')
print("Team available parameters:")
print(list(frequencies_xr.keys()))

# Construïm una matriu amb els paràmetres dels jugadors a pista (matriu `X`) i una matriu amb el resultat (matriu `y`). La matriu `y` l'extraiem directament de `matches_df`. La matriu `X` la construïm a partir dels jugadors que són a pista a cada partit (de `matches_df`) i les estadístiques de `stats_xr`.

# Ens cal codificar els resultats. Els possibles resultats són 6: '3-0', '3-1', '3-2', '0-3', '1-3', '2-3'. Perquè el model pugui treballar bé, passarem aquests possibles valors a codis (0, 1, 2, 3, 4, 5), a partir dels quals en farem l'entrenament. Per usos futurs, també codificarem el guanyador (local/vistant).

encoder_winning = preprocessing.LabelEncoder() # codificador d'etiquetes ('Local'/'Visitant') a nombres (0/1)
Results_winning = encoder_winning.fit_transform(matches_df['Guanyador'].drop(0).values.astype(str)) # Array amb tots els resultats que farem servir per l'entrenament del model (0='Local', 1='Visitant')
# hem tret el primer partit perquè es faran servir les estadístiques del partit anterior per predir cada resultat, i per al primer partit no hi ha dades prèvies

# El mateix que abans, però per tots els possibles resultats
# Llista amb tots els resultats en format unificat (6 resultats possibles)
Scores = [str(matches_df['Local'].iloc[i])+'-'+str(matches_df['Visitant'].iloc[i]) for i in range(matches_df.shape[0])]
Scores = np.array(Scores[1:]) # hem tret el primer partit perquè es faran servir les estadístiques del partit anterior per predir cada resultat, i per al primer partit no hi ha dades prèvies

# Codifiquem els resultats (0-3: 0, 1-3: 1, ..., 3-2:5)
encoder_scores = preprocessing.LabelEncoder()
Scores_training = encoder_scores.fit_transform(Scores.astype(str))

# Desem l'StandardScaler() per fer-lo servir en un altre programa
joblib.dump(encoder_scores, '../generated_files/encoder_scores.pkl')

# Codifiquem el nom de cada jugador
encoder_names = preprocessing.LabelEncoder()
players_names = np.unique(matches_df[['Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4']].values.flatten())
player_codes = encoder_names.fit_transform(players_names)
player_codes_dict = {players_names[i]: player_codes[i] for i in range(len(players_names))}

# Seguidament construïm la matriu `X` que conté les dades dels jugadors al camp. Primer definim quins paràmetres tenim en compte pels atacants i pels defensors. Després, construïm la matriu on, fila per fila, hi ha tots els paràmetres dels jugadors.

# Paràmetres que considerem al model, en funció de si el jugador és atacant o defensor
#considered_stats_defense = ['WinDefensePlayed', 'ScoredDefensePlayed', 'ReceivedDefensePlayed', 'WinPlayedMatchday']
#considered_stats_attack = ['WinAttackPlayed', 'ScoredAttackPlayed', 'ReceivedAttackPlayed', 'WinPlayedMatchday']
#considered_stats_teams = ['ELOAttackDifference', 'ELODefenseDifference', 'ELOAttackDefenseDifference',
#                          'ELODefenseAttackDifference', 'CloseWinsLocal',
#                          'CloseWinsVisitant', 'ReceivedGoalsDDLocal', 'ReceivedGoalsDDVisitant',
#                          'ReceivedGoalsADLocal', 'ReceivedGoalsADVisitant', 'WinsLocal', 'WinsVisitant']
considered_stats_defense = ['WinDefensePlayed', 'WinPlayedMatchday', 'ReceivedDefensePlayed']
considered_stats_attack = ['WinAttackPlayed', 'WinPlayedMatchday', 'ScoredAttackPlayed']# afegir Goals_Differential_per_Game
considered_stats_teams = ['ELOAttackDefenseDifference',
                          'ELODefenseAttackDifference',
                          'ELODifference',
                          'WinsLocal', 'WinsVisitant', 'CloseWinsPlayedLocal', 'CloseWinsPlayedVisitant']
print("\nChosen parameters for defense:", considered_stats_defense)
print("Chosen parameters for attack:", considered_stats_attack)
print("Chosen parameters for teams:", considered_stats_teams)
print()

## Creem el training set
# Dataframe on hi desem tots els paràmetres d'avaluació de cada jugador
columns = [stat_def+'1' for stat_def in considered_stats_defense] + [stat_att+'2' for stat_att in considered_stats_attack] +\
            [stat_def+'3' for stat_def in considered_stats_defense] + [stat_att+'4' for stat_att in considered_stats_attack] # noms de les columnes
feature_names = columns + considered_stats_teams
print(len(feature_names))
Stats_training = pd.DataFrame(columns = feature_names)
#print(Stats_training.columns)
#Stats_training = pd.DataFrame(columns = ['ELODiffAttack'])

for match in range(1, matches_df.shape[0]): # per cada partit disputat
#for match in range(int(0.2*matches_df.shape[0])+1, matches_df.shape[0]+1): # per cada partit disputat. Treiem els primers partits que no representen l'ELO real dels jugadors
    match_df = matches_df.iloc[match-1] # triem les dades d'aquest partit

    #matchday = match_df['Total_D']-1 # número de matchday
    # Llista on hi desarem els valors des les estadístiques de cada jugador que hi ha al camp, amb el mateix ordre que `columns`
    stats_match = []
    for player in match_df[['Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4']]:
        if (player == match_df['Jugador 1']) or (player == match_df['Jugador 3']): # defensors
            # Triem les estadístiques dels jugadors en aquest partit
            player_stats = stats_xr.sel(match=match, player=player)[considered_stats_defense].to_array().values
        elif (player == match_df['Jugador 2']) or (player == match_df['Jugador 4']): # atacants
            # Triem les estadístiques dels jugadors en aquest partit
            player_stats = stats_xr.sel(match=match, player=player)[considered_stats_attack].to_array().values
        stats_match = stats_match + list(player_stats) # adjuntem les estadístiques del jugador a les dades d'aquest partit

    # Afegim a la llista paràmetres del partit, creuant els dos equips
    elo_attack_difference = (stats_xr.sel(match=match, player=match_df['Jugador 2'])['ELOAttack'].values.item() -
                             stats_xr.sel(match=match, player=match_df['Jugador 4'])['ELOAttack'].values.item())
    elo_defense_difference = (stats_xr.sel(match=match, player=match_df['Jugador 1'])['ELODefense'].values.item() -
                             stats_xr.sel(match=match, player=match_df['Jugador 3'])['ELODefense'].values.item())
    elo_difference = ( (stats_xr.sel(match=match, player=match_df['Jugador 1']))['ELODefense'].values.item() +
                       (stats_xr.sel(match=match, player=match_df['Jugador 2']))['ELOAttack'].values.item() -
                       ((stats_xr.sel(match=match, player=match_df['Jugador 3']))['ELODefense'].values.item() +
                       (stats_xr.sel(match=match, player=match_df['Jugador 4']))['ELOAttack'].values.item() ))
    elo_attackh_defensea_difference = (stats_xr.sel(match=match, player=match_df['Jugador 2'])['ELOAttack'].values.item() -
                              stats_xr.sel(match=match, player=match_df['Jugador 3'])['ELODefense'].values.item()) # diferència ELO atacant-defensor rivals
    elo_defenseh_attacka_difference = (stats_xr.sel(match=match, player=match_df['Jugador 1'])['ELODefense'].values.item() -
                              stats_xr.sel(match=match, player=match_df['Jugador 4'])['ELOAttack'].values.item())
    close_wins_local = frequencies_xr.sel(teammate=match_df['Jugador 1'], player = match_df['Jugador 2'])['CloseWinsPlayed'].values.item()
    close_wins_visitant = frequencies_xr.sel(teammate=match_df['Jugador 3'], player = match_df['Jugador 4'])['CloseWinsPlayed'].values.item()
    receivedgoals_defense_defense_local = frequencies_xr.sel(defender = match_df['Jugador 1'], defender_rival = match_df['Jugador 3'])['ReceivedGoalsGamesDefenseDefense'].values.item()
    receivedgoals_defense_defense_visitant = frequencies_xr.sel(defender = match_df['Jugador 3'], defender_rival = match_df['Jugador 1'])['ReceivedGoalsGamesDefenseDefense'].values.item()
    receivedgoals_attack_defense_local = frequencies_xr.sel(defender = match_df['Jugador 1'], attacker_rival = match_df['Jugador 4'])['ReceivedGoalsGamesAttackDefense'].values.item()
    receivedgoals_attack_defense_visitant = frequencies_xr.sel(defender = match_df['Jugador 3'], attacker_rival = match_df['Jugador 2'])['ReceivedGoalsGamesAttackDefense'].values.item()
    team_wins_local = frequencies_xr.sel(teammate = match_df['Jugador 1'], player = match_df['Jugador 2'])['TeammatesWinsPlayed'].values.item()
    team_wins_visitant = frequencies_xr.sel(teammate = match_df['Jugador 3'], player = match_df['Jugador 4'])['TeammatesWinsPlayed'].values.item()

    # Per CloseWinsPlayed, fem que el valor sigui 0.5 si l'original és NaN, que vol dir que no han jugat mai un Close Match junts
    if np.isnan(close_wins_local):
      close_wins_local = 0.5
    if np.isnan(close_wins_visitant):
        close_wins_visitant = 0.5

#    stats_match = stats_match + [elo_attack_difference, elo_defense_difference, elo_attackh_defensea_difference, elo_defenseh_attacka_difference,
#                                 close_wins_local, close_wins_visitant,
#                                 receivedgoals_defense_defense_local, receivedgoals_defense_defense_visitant,
#                                 receivedgoals_attack_defense_local, receivedgoals_attack_defense_visitant,
#                                 team_wins_local, team_wins_visitant]
    stats_match = stats_match + [elo_attackh_defensea_difference, elo_defenseh_attacka_difference,
                                     elo_difference, team_wins_local, team_wins_visitant, close_wins_local, close_wins_visitant]
    # Afegim el codi numèric de cada jugador
    #player_codes_match = [player_codes_dict[match_df['Jugador 1']], player_codes_dict[match_df['Jugador 2']],
    #                      player_codes_dict[match_df['Jugador 3']], player_codes_dict[match_df['Jugador 4']]]
    #stats_match = player_codes_match + stats_match

    # Desem la llista d'estadístiques d'aquest partit
    Stats_training.loc[len(Stats_training)] = stats_match

Stats_training.to_csv('../generated_files/stats_training.csv', index=False) # desem per poder obrir més tard

print('Saved stats training')

#############################################################################
### Analysis and prediction from https://github.com/NunonuN/ml-playground ###
#############################################################################

title = "RAW DATA SET"
lenti = len(title)
print("="*lenti)
print(title)
print("-"*lenti + '\n')
print("Features")
print("-"*8)
print(f"Shape: {Stats_training.shape}\n")
print(Stats_training.head())
print(f"\nTarget")
print("-"*6)
print(f"{Results_winning[:5] = }")
print(f"{Results_winning[-5:] = }")
print('='*80)

# Retallem els datasets. Deixem fora els primers partits per aconseguir ELOs estables per cada jugador
threshold = 0.3 # descartem el primer X% de partits per entrenar el model
Results_winning = Results_winning[int(threshold*len(Results_winning)):]
Stats_training = Stats_training[int(threshold*Stats_training.shape[0]):]

print("-"*6)
print(f"\nCroppping first {threshold*100} % of points")
print("-"*6)

# Ara ja tenim les dades `X` per entrenar el model. Abans d'entrenar-lo, estandaritzem els valors.
# #Això és un pas comú en IA quan es tenen paràmetres amb diferents escales de valors.
# standardise features
scaler_class = preprocessing.StandardScaler()
X_class = scaler_class.fit_transform(Stats_training)

df_class = pd.DataFrame(X_class, columns= feature_names)
df_class['Local/Visitant'] = Results_winning

title = "MATCHES DATA SET"
lenti = len(title)
print("-" * lenti)
print(title)
print("-"*lenti + '\n')
print(f"Shape: {df_class.shape}")
print(f"\nClass distribution:")
print(df_class['Local/Visitant'].value_counts())
print(f"\nFirst few rows:")
print(df_class.head())

# Desem l'StandardScaler() per fer-lo servir en un altre programa
joblib.dump(scaler_class, '../generated_files/scaler.pkl')

# split data into training and testing sets
X_class_trn, X_class_tst, y_class_trn, y_class_tst = (
  train_test_split(
    X_class,
    Results_winning,
    test_size=0.15,
    random_state=RND_STATE
  )
)

print(f"Training set size: {X_class_trn.shape[0]}")
print(f"Testing set size: {X_class_tst.shape[0]}")
print(np.argwhere(np.isnan(X_class_trn))) # check for NaN values in training set
print(np.argwhere(np.isnan(X_class_tst))) # check for NaN values in training set

## Decision Tree Classifier
dt_class = DecisionTreeClassifier(
  max_depth=4, # 4, (6)
  min_samples_leaf=4, # 4, (6)
  min_samples_split=10, # 10, (20)
  random_state=RND_STATE
)
dt_class.fit(X_class_trn, y_class_trn)
y_pred_dt_class = dt_class.predict(X_class_tst)

title = "Game winning Classification – Decision Tree"
lenti = len(title)
print(title)
print("-"*lenti)
print(f"Accuracy: {accuracy_score(y_class_tst, y_pred_dt_class):.4f}")
print(f"F1-Score: {f1_score(y_class_tst, y_pred_dt_class):.4f}")


## Bagging Classifier
bagging_class = BaggingClassifier(
  estimator=DecisionTreeClassifier(
    max_depth=10,
    random_state=RND_STATE
  ),
  n_estimators=10, # 50
  max_features=1.0,
  max_samples=1.0,
  random_state=RND_STATE,
  # n_jobs=None
  n_jobs=-1
)
bagging_class.fit(X_class_trn, y_class_trn)
y_pred_bagging = bagging_class.predict(X_class_tst)

title = "Game winning Classification – Bagging Classifier"
lenti = len(title)
print(title)
print("-"*lenti)
print(f"Accuracy: {accuracy_score(y_class_tst, y_pred_bagging):.4f}")
print(f"F1-Score: {f1_score(y_class_tst, y_pred_bagging):.4f}")

## Random Forest Classifier
rf_class = RandomForestClassifier(
  n_estimators=200,
  max_depth=10,
  max_features='sqrt',
  min_samples_split=8, # 5
  random_state=RND_STATE,
  n_jobs=-1
)
rf_class.fit(X_class_trn, y_class_trn)
y_pred_rf = rf_class.predict(X_class_tst)
y_pred_rf_prob = rf_class.predict_proba(X_class_tst)[:, 1]

title = "Game winning Classification – Random Forest"
lenti = len(title)
print(title)
print("-"*lenti)
print(f"Accuracy: {accuracy_score(y_class_tst, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_class_tst, y_pred_rf):.4f}")

## XGBoost Classifier
xgb_class = xgb.XGBClassifier(
  n_estimators=150,
  max_depth=4,            # shallow trees (boosting builds depth via ensembles)
  learning_rate=0.05,     # slow, steady learning
  subsample=0.8,          # 90% of data per tree
  colsample_bytree=0.7,   # 70% of features per tree
  reg_lambda=1.0,         # moderate regularisation
  random_state=RND_STATE,
  n_jobs=-1,
  verbosity=0
)
xgb_class.fit(
  X_class_trn, y_class_trn,
  eval_set=[(X_class_tst, y_class_tst)],
  verbose=False
)
y_pred_xgb = xgb_class.predict(X_class_tst)
y_pred_xgb_prob = xgb_class.predict_proba(X_class_tst)[:, 1]

title = "Game winning Classification – XGBoost"
lenti = len(title)
print(title)
print("-"*lenti)
print(f"Accuracy: {accuracy_score(y_class_tst, y_pred_xgb):.4f}")
print(f"F1-Score: {f1_score(y_class_tst, y_pred_xgb):.4f}")

## Stacking Classifier
# define base learners (diverse models)
base_learners_class = [
  # tree-based
  ('dt', DecisionTreeClassifier(
    max_depth=8,
    random_state=RND_STATE
  )),
  # margin-based
  ('svm', SVC(
    kernel='rbf',
    probability=True,
    random_state=RND_STATE
  )),
  # instance-based
  ('knn', KNeighborsClassifier(n_neighbors=5))
]

# define meta-learner
meta_learner_class = LogisticRegression(
  max_iter=1000,
  random_state=RND_STATE
)

# create stacking classifier
stacking_class = StackingClassifier(
  estimators=base_learners_class,
  final_estimator=meta_learner_class,
  cv=5
)
stacking_class.fit(X_class_trn, y_class_trn)
y_pred_stacking = stacking_class.predict(X_class_tst)

title = "Game winning Classification – Stacking Classifier"
lenti = len(title)
print(title)
print("-"*lenti)
print(f"Accuracy: {accuracy_score(y_class_tst, y_pred_stacking):.4f}")
print(f"F1-Score: {f1_score(y_class_tst, y_pred_stacking):.4f}")


### Summary


# compile results for all models
results_class = {
  'Model': [
    'Decision Tree',
    'Bagging',
    'Random Forest',
    'XGBoost',
    'Stacking'
  ],
  'Accuracy': [
    accuracy_score(y_class_tst, y_pred_dt_class),
    accuracy_score(y_class_tst, y_pred_bagging),
    accuracy_score(y_class_tst, y_pred_rf),
    accuracy_score(y_class_tst, y_pred_xgb),
    accuracy_score(y_class_tst, y_pred_stacking)
  ],
  'Precision': [
    precision_score(y_class_tst, y_pred_dt_class),
    precision_score(y_class_tst, y_pred_bagging),
    precision_score(y_class_tst, y_pred_rf),
    precision_score(y_class_tst, y_pred_xgb),
    precision_score(y_class_tst, y_pred_stacking)
  ],
  'Recall': [
    recall_score(y_class_tst, y_pred_dt_class),
    recall_score(y_class_tst, y_pred_bagging),
    recall_score(y_class_tst, y_pred_rf),
    recall_score(y_class_tst, y_pred_xgb),
    recall_score(y_class_tst, y_pred_stacking)
  ],
  'F1-Score': [
    f1_score(y_class_tst, y_pred_dt_class),
    f1_score(y_class_tst, y_pred_bagging),
    f1_score(y_class_tst, y_pred_rf),
    f1_score(y_class_tst, y_pred_xgb),
    f1_score(y_class_tst, y_pred_stacking)
  ]
}

df_class_res = pd.DataFrame(results_class)
print('\n' + "="*70)
print("Classification Results Summary")
print("-"*70)
print(df_class_res.to_string(index=False))
print(
  f"\nBest Model (F1-Score): {df_class_res.loc[df_class_res['F1-Score'].idxmax(), 'Model']}"
)

## Visualization of results
# plot performance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# bar plot of metrics
ax = axes[0]
x = np.arange(len(df_class_res))
width = 0.2

ax.bar(
  x - 1.5*width,
  df_class_res['Accuracy'],
  width,
  label='Accuracy',
  alpha=0.8
)
ax.bar(
  x - 0.5*width,
  df_class_res['Precision'],
  width,
  label='Precision',
  alpha=0.8
)
ax.bar(
  x + 0.5*width,
  df_class_res['Recall'],
  width,
  label='Recall',
  alpha=0.8
)
ax.bar(
  x + 1.5*width,
  df_class_res['F1-Score'],
  width,
  label='F1-Score',
  alpha=0.8
)

ax.set_ylabel('Score')
ax.set_title('Classification Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(
  df_class_res['Model'],
  rotation=45,
  ha='right'
)
ax.legend()
ax.grid(axis='y', alpha=0.4)
ax.tick_params(axis='x', which='both', gridOn=False)

# heatmap of metrics
ax = axes[1]
metrics_data = df_class_res[[
  'Accuracy',
  'Precision',
  'Recall',
  'F1-Score'
]].set_index(
  df_class_res['Model']
)

sns.heatmap(
  metrics_data.T,
  annot=True,
  fmt='.3f',
  cmap='YlGn',
  ax=ax,
  cbar_kws={'label': 'Score'}
)
ax.set_title('Metrics Heatmap')
ax.set_xlabel('Model')
ax.set_ylabel('Metric')

plt.tight_layout()
plt.show()

print("\nVisualisations generated successfully.")

## Feature analysis plot
# extract feature importance from tree-based models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# random Forest feature importance
rf_importance = rf_class.feature_importances_
sorted_idx = np.argsort(rf_importance)[::-1]

ax = axes[0]
ax.barh(
  [feature_names[i] for i in sorted_idx],
  rf_importance[sorted_idx],
  alpha=0.8
)
ax.set_xlabel('Importance')
ax.set_title('Random Forest feature importance')
ax.invert_yaxis()

# XGBoost feature importance
xgb_importance = xgb_class.feature_importances_
sorted_idx_xgb = np.argsort(xgb_importance)[::-1]

ax = axes[1]
ax.barh(
  [feature_names[i] for i in sorted_idx_xgb],
  xgb_importance[sorted_idx_xgb],
  alpha=0.8,
  color='orange'
)
ax.set_xlabel('Importance')
ax.set_title('XGBoost feature importance')
ax.invert_yaxis()

plt.tight_layout()
plt.show()

print("Feature importance analysis complete.")


## 5-fold cross validation
cv = StratifiedKFold(
  n_splits=5,
  shuffle=True,
  random_state=RND_STATE,
)

models_class_cv = {
  'Decision Tree': DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=4,
    min_samples_split=10,
    random_state=RND_STATE,
  ),
  'Bagging': BaggingClassifier(
    estimator=DecisionTreeClassifier(
      max_depth=10,
      random_state=RND_STATE
    ),
    n_estimators=10,
    max_features=1.0,
    max_samples=1.0,
    random_state=RND_STATE,
    n_jobs=-1
  ),
  'Random Forest': RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_features='sqrt',
    min_samples_split=8,
    random_state=RND_STATE,
    n_jobs=-1,
  ),
  'XGBoost': xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    random_state=RND_STATE,
    verbosity=0,
  ),
}

title = "5-fold CV (F1-score, classification)".upper()
lentit = len(title)
print(title)
print('-' * lentit)
for name, model in models_class_cv.items():
  scores = cross_val_score(
    model,
    X_class,
    Results_winning,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
  )
  print(
    f"{name:15s} "
    f"mean = {scores.mean():.3f} "
    f"± {scores.std():.3f}"
  )

