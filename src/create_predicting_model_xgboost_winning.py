#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
# ## Endeví de resultats
# 
# Aquest codi inclou un model de Deep Learning per predir els resultats d'un partit a partir de les dades històriques de partits.

import numpy as np
import pandas as pd
import xarray as xr

import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

''' Aquest codi utilitza un model XGBoost per predir els resultats dels partits a partir de les estadístiques dels jugadors. '''

# Per poder entrenar el model, ens cal donar-li informació de qui està jugant (amb quins punts forts i punts febles) i quin és el resultat. Per tant, hem de llegir el fitxer amb les estadístiques de cada jugador i també el fitxer amb el registre històric de resultats.

# Llegim l'històric de partits i resultats
matches_df = pd.read_csv('../generated_files/results_historical.csv')

# Llegim les estadístiques de cada jugador
stats_xr = xr.open_dataset('../generated_files/stats_historical.nc', engine='scipy')

# Llegim les estadístiques creuades (com un jugador es comporta contra un altre)
frequencies_xr = xr.open_dataset('../generated_files/teammates_historical.nc', engine='scipy')

# Construïm una matriu amb els paràmetres dels jugadors a pista (matriu `X`) i una matriu amb el resultat (matriu `y`). La matriu `y` l'extraiem directament de `matches_df`. La matriu `X` la construïm a partir dels jugadors que són a pista a cada partit (de `matches_df`) i les estadístiques de `stats_xr`.

# Ens cal codificar els resultats. Els possibles resultats són 6: '3-0', '3-1', '3-2', '0-3', '1-3', '2-3'. Perquè el model pugui treballar bé, passarem aquests possibles valors a codis (0, 1, 2, 3, 4, 5), a partir dels quals en farem l'entrenament. Per usos futurs, també codificarem el guanyador (local/vistant).

encoder_winning = preprocessing.LabelEncoder() # codificador d'etiquetes ('Local'/'Visitant') a nombres (0/1)
Results_winning_training = encoder_winning.fit_transform(matches_df['Guanyador'].values.astype(str)) # Array amb tots els resultats que farem servir per l'entrenament del model (0='Local', 1='Visitant')

# El mateix que abans, però per tots els possibles resultats
# Llista amb tots els resultats en format unificat (6 resultats possibles)
Scores = [str(matches_df['Local'].iloc[i])+'-'+str(matches_df['Visitant'].iloc[i]) for i in range(matches_df.shape[0])]
Scores = np.array(Scores)

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
considered_stats_defense = ['WinDefensePlayed', 'WinPlayedMatchday']
considered_stats_attack = ['WinAttackPlayed', 'WinPlayedMatchday']
considered_stats_teams = ['ELOAttackDefenseDifference',
                          'ELODefenseAttackDifference',
                          'WinsLocal', 'WinsVisitant']

## Creem el training set
# Dataframe on hi desem tots els paràmetres d'avaluació de cada jugador
columns = [stat_def+'1' for stat_def in considered_stats_defense] + [stat_att+'2' for stat_att in considered_stats_attack] +\
            [stat_def+'3' for stat_def in considered_stats_defense] + [stat_att+'4' for stat_att in considered_stats_attack] # noms de les columnes
Stats_training = pd.DataFrame(columns = columns + considered_stats_teams)
#Stats_training = pd.DataFrame(columns = ['ELODiffAttack'])

#for nmatch in matches_df['Total_D']: # per cada partit disputat
for match in range(1, matches_df.shape[0]+1): # per cada partit disputat
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
#    stats_match = stats_match + [elo_attack_difference, elo_defense_difference, elo_attackh_defensea_difference, elo_defenseh_attacka_difference,
#                                 close_wins_local, close_wins_visitant,
#                                 receivedgoals_defense_defense_local, receivedgoals_defense_defense_visitant,
#                                 receivedgoals_attack_defense_local, receivedgoals_attack_defense_visitant,
#                                 team_wins_local, team_wins_visitant]
    stats_match = stats_match + [elo_attackh_defensea_difference, elo_defenseh_attacka_difference,
                                     team_wins_local, team_wins_visitant]

    # Afegim el codi numèric de cada jugador
    #player_codes_match = [player_codes_dict[match_df['Jugador 1']], player_codes_dict[match_df['Jugador 2']],
    #                      player_codes_dict[match_df['Jugador 3']], player_codes_dict[match_df['Jugador 4']]]
    #stats_match = player_codes_match + stats_match

    # Desem la llista d'estadístiques d'aquest partit
    Stats_training.loc[len(Stats_training)] = stats_match

Stats_training.to_csv('../generated_files/stats_training.csv', index=False) # desem per poder obrir més tard

print('Saved stats training')

# Ara ja tenim les dades `X` per entrenar el model. Abans d'entrenar-lo, estandaritzem els valors.
# #Això és un pas comú en IA quan es tenen paràmetres amb diferents escales de valors.
scaler = preprocessing.StandardScaler()
Stats_training_stand = scaler.fit(Stats_training).transform(Stats_training.astype(float))

# Desem l'StandardScaler() per fer-lo servir en un altre programa
joblib.dump(scaler, '../generated_files/scaler.pkl')

# Separem tots els partits que tenim en una mostra d'entrenament (train) i una mostra de test.
# Això ens permetrà avaluar el rendiment del nostre model. En aquest cas, de la mostra total un 10% serà de test i el 90% restant serà per entrenar el model.

X_train, X_test, y_winning_train, y_winning_test, y_score_train, y_score_test = (
    train_test_split(Stats_training_stand, Results_winning_training, Scores_training, test_size=0.2))

# Second split: from the training set, carve out a validation set
X_train, X_val, y_winning_train, y_winning_val = train_test_split(
    X_train, y_winning_train, test_size=0.1, random_state=42, stratify=y_winning_train
)

# Passem els valors y de codi numèric (0, 1, 2...) a resultat real (3-0, 3-1, 3-2...)
#y_score_train_labels = encoder_scores.inverse_transform(y_score_train)
#y_score_test_labels = encoder_scores.inverse_transform(y_score_test)

#y_winning_train_labels = encoder_winning.inverse_transform(y_winning_train) # igual per l'equip guanyador (0,1 -> 'Local','Visitant')
#y_winning_val_labels = encoder_scores.inverse_transform(y_winning_val)
#y_winning_test_labels = encoder_winning.inverse_transform(y_winning_test)

# ## Definició i entrenament del model
# 2) compute scale_pos_weight for binary imbalance (if binary)
if len(np.unique(Results_winning_training)) == 2:
    negatives = (Results_winning_training == 0).sum()
    positives = (Results_winning_training == 1).sum()
    scale_pos_weight = negatives / max(1, positives)
else:
    scale_pos_weight = 1.0

# 3) pipeline + XGB classifier
y = Results_winning_training
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        objective="binary:logistic" if len(np.unique(y)) == 2 else "multi:softprob",
        num_class=None if len(np.unique(y)) == 2 else len(np.unique(y)),
        verbosity=0,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    ))
])

# 4) hyperparameter distributions
param_dist = {
    "clf__n_estimators": [100, 300, 600, 1000],
    "clf__max_depth": [3, 5, 7, 9],
    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "clf__subsample": [0.6, 0.8, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
    "clf__reg_alpha": [0, 0.5, 1, 5],
    "clf__reg_lambda": [1, 3, 5, 10]
}

# 5) Randomized search with stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=40, scoring="accuracy", cv=cv,
    n_jobs=-1, random_state=42, verbose=1
)
search.fit(X_train, y_winning_train)

# 6) Evaluate on test set
best = search.best_estimator_
y_pred = best.predict(X_test)
print("Best params:", search.best_params_)
print(classification_report(y_winning_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_winning_test, y_pred)
plt.savefig('../results/Confusion_Matrix_Test_winning.png', dpi=300, bbox_inches='tight')
plt.clf()

# 7) Save model and scaler
joblib.dump(best, '../generated_files/xgb_winning_best_pipeline.joblib')

# 8) SHAP summary (explain top features)
feature_names = Stats_training.columns.tolist()
X_shap = pd.DataFrame(best.named_steps["scaler"].transform(X_train), columns=feature_names)
explainer = shap.Explainer(best.named_steps["clf"], X_shap)
shap_exp = explainer(X_shap)  # Explanation object

# Plot and save (feature names will appear because X_shap is a DataFrame)
shap.summary_plot(shap_exp.values, X_shap, show=False)
plt.savefig('../results/shap_summary_winning.png', dpi=300, bbox_inches='tight')
plt.clf()

#pred_labels = probs.argmax(axis=0)
#print(probs, pred_labels)
#results = encoder_scores.inverse_transform(pred_labels)

# Prediccions dels marcadors
# Conjunt de paràmetres que avaluen el model
#print(classification_report(y_winning_test, pred_labels))

# En funció del resultat predit, trobem quin equip es prediu que guanya
#winner_labels = []
#for i in range(len(results)):
#    local, visitant = results[i].split('-')
#    if local > visitant:
#        winner_label = 'Local'
#    else:
#        winner_label = 'Visitant'
#        winner_labels.append(winner_label)

#winner_labels = np.array(winner_labels)
#winner_labels_encoded = encoder_winning.transform(winner_labels)

# Precisió de la predicció. Comparem els marcadors predits (score_pred_class) i els marcadors reals (y_score_test)
#accuracy = model.score(X_test, y_winning_test)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Imprimim la matriu de confusió, que ens indica els punts forts i punts febles del model
#ConfusionMatrixDisplay.from_predictions(y_winning_test, pred_labels)
#plt.savefig('../results/Confusion_Matrix_Winning_Test.png', dpi=300, bbox_inches='tight')
#plt.clf()

# Precisió de la predicció. Comparem els marcadors predits (score_pred_class) i els marcadors reals (y_score_test)
#print("Accuracy in winning team:", accuracy_score(y_winning_train, winner_labels_encoded) * 100, "%")


# Imprimim la matriu de confusió, que ens indica els punts forts i punts febles del model
#ConfusionMatrixDisplay.from_predictions(encoder_winning.inverse_transform(y_winning_train), winner_labels)
#plt.savefig('../results/Confusion_Matrix_Winning_Train.png', dpi=300, bbox_inches='tight')
#plt.clf()

# Desem el model i els pesos
#model.save_model('../generated_files/xgboost_model.json')

