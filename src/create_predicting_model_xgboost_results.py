#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import joblib
import shap
from shap import Explainer

from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  roc_auc_score,
  confusion_matrix,
  mean_squared_error,
  r2_score,
  mean_absolute_error,
  classification_report,
  ConfusionMatrixDisplay,
  RocCurveDisplay
)
from sklearn.model_selection import (
  train_test_split,
  cross_val_score,
  KFold,
  StratifiedKFold
)

from sklearn.preprocessing import (
  StandardScaler,
  LabelEncoder
)

import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# set visualisation style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# set random seed for reproducibility
RND_STATE = 42 #None
np.random.seed(RND_STATE)

''' Aquest codi utilitza un model XGBoost per predir els resultats dels partits a partir de les estadístiques dels jugadors. '''

# Per poder entrenar el model, ens cal donar-li informació de qui està jugant (amb quins punts forts i punts febles) i quin és el resultat. Per tant, hem de llegir el fitxer amb les estadístiques de cada jugador i també el fitxer amb el registre històric de resultats.

###############################################################
############## Llegir les dades ###############################
###############################################################

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

###############################################################
############# Crear el dataset amb les dades ##################
###############################################################

# Construïm una matriu amb els paràmetres dels jugadors a pista (matriu `X`) i una matriu amb el resultat (matriu `y`). La matriu `y` l'extraiem directament de `matches_df`. La matriu `X` la construïm a partir dels jugadors que són a pista a cada partit (de `matches_df`) i les estadístiques de `stats_xr`.

# Ens cal codificar els resultats. Els possibles resultats són 6: '3-0', '3-1', '3-2', '0-3', '1-3', '2-3'. Perquè el model pugui treballar bé, passarem aquests possibles valors a codis (0, 1, 2, 3, 4, 5), a partir dels quals en farem l'entrenament. Per usos futurs, també codificarem el guanyador (local/vistant).
# Llista amb tots els resultats en format unificat (6 resultats possibles)
Scores = [str(matches_df['Local'].iloc[i])+'-'+str(matches_df['Visitant'].iloc[i]) for i in range(matches_df.shape[0])]
Scores = np.array(Scores[1:]) # hem tret el primer partit perquè es faran servir les estadístiques del partit anterior per predir cada resultat, i per al primer partit no hi ha dades prèvies

# Codifiquem els resultats (0-3: 0, 1-3: 1, ..., 3-2:5)
encoder_scores = preprocessing.LabelEncoder()
Scores_coded = encoder_scores.fit_transform(Scores.astype(str))

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
considered_stats_defense = ['WinDefensePlayed', 'ReceivedDefensePlayed']
considered_stats_attack = ['WinAttackPlayed', 'ScoredAttackPlayed']
considered_stats_defense, considered_stats_attack = [], []
considered_stats_teams = ['ELOAttackDefenseDifference',
                          'ELODefenseAttackDifference',
                          'ELODifference',
                          'WeightedELODifference',
                          'WinsDifference', 'CloseWinsDifference',
                          'NeatGoalsAttackDifference', 'NeatGoalsDefenseDifference',
                          'WinsMatchdayDifference',
                          'ReceivedGoalsDefenseDefenseDifference', 'ReceivedGoalsAttackDefenseDifference']
print("\nChosen parameters for defense:", considered_stats_defense)
print("Chosen parameters for attack:", considered_stats_attack)
print("Chosen parameters for teams:", considered_stats_teams)
print()

## Creem el training set
# Dataframe on hi desem tots els paràmetres d'avaluació de cada jugador
columns = [stat_def+'1' for stat_def in considered_stats_defense] + [stat_att+'2' for stat_att in considered_stats_attack] +\
            [stat_def+'3' for stat_def in considered_stats_defense] + [stat_att+'4' for stat_att in considered_stats_attack] # noms de les columnes
feature_names = columns + considered_stats_teams
Stats_training = pd.DataFrame(columns = feature_names)
#print(Stats_training.columns)
#Stats_training = pd.DataFrame(columns = ['ELODiffAttack'])

def calculate_differential_head2head(xarr, stat, match_dataframe, match_number, player_local = 'Jugador 1', player_visitant = 'Jugador 3'):
    '''
    El calcula la diferència d'un paràmetre entre dos jugadors que juguen a la mateixa posició però en equips contraris
    :param xarr: xarray amb les estadístiques dels jugadors
    :param stat: estatística que volem comparar
    :param match_dataframe: dades del partit, on hi ha els noms dels jugadors
    :param match_number: número del partit que volem analitzar
    :return: diferència del paràmetre entre els dos jugadors
    '''
    values_local = xarr.sel(match=match_number, player=match_dataframe[player_local])[stat].values.item()
    values_visitant = xarr.sel(match=match, player=match_dataframe[player_visitant])[stat].values.item()

    differential = values_local - values_visitant
    return differential


def calculate_differential_opositeposition(xarr, stat_local, stat_visitant, match_dataframe, match_number, player_local='Jugador 1',
                                     player_visitant='Jugador 3'):
  '''
  El calcula la diferència d'un paràmetre entre dos jugadors que juguen en posicions oposades (atacant vs defensor)
  :param xarr: xarray amb les estadístiques dels jugadors
  :param stat: estatística que volem comparar
  :param match_dataframe: dades del partit, on hi ha els noms dels jugadors
  :param match_number: número del partit que volem analitzar
  :return: diferència del paràmetre entre els dos jugadors
  '''
  values_local = xarr.sel(match=match_number, player=match_dataframe[player_local])[stat_local].values.item()
  values_visitant = xarr.sel(match=match, player=match_dataframe[player_visitant])[stat_visitant].values.item()

  differential = values_local - values_visitant
  return differential


for match in range(1, matches_df.shape[0]): # per cada partit disputat
#for match in range(int(0.2*matches_df.shape[0])+1, matches_df.shape[0]+1): # per cada partit disputat. Treiem els primers partits que no representen l'ELO real dels jugadors
    match_df = matches_df.iloc[match-1] # triem les dades d'aquest partit

    #matchday = match_df['Total_D']-1 # número de matchday
    # Llista on hi desarem els valors des les estadístiques de cada jugador que hi ha al camp, amb el mateix ordre que `columns`
    stats_match = []
    #REMOVED INDIVIDUAL PERFORMANCE FACTORS
    #for player in match_df[['Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4']]:
    #    if (player == match_df['Jugador 1']) or (player == match_df['Jugador 3']): # defensors
    #        # Triem les estadístiques dels jugadors en aquest partit
    #        player_stats = stats_xr.sel(match=match, player=player)[considered_stats_defense].to_array().values
    #    elif (player == match_df['Jugador 2']) or (player == match_df['Jugador 4']): # atacants
    #        # Triem les estadístiques dels jugadors en aquest partit
    #        player_stats = stats_xr.sel(match=match, player=player)[considered_stats_attack].to_array().values
    #    stats_match = stats_match + list(player_stats) # adjuntem les estadístiques del jugador a les dades d'aquest partit

    # --- Paràmetres individuals dels jugadors ---
    # ELO
    elo_attack_difference = calculate_differential_head2head(stats_xr, 'ELOAttack', match_df, match, player_local='Jugador 2', player_visitant='Jugador 4')
    elo_defense_difference = calculate_differential_head2head(stats_xr, 'ELODefense', match_df, match, player_local='Jugador 1', player_visitant='Jugador 3')

    elo_difference = elo_attack_difference + elo_defense_difference

    weighted_elo_difference = ( (stats_xr.sel(match=match, player=match_df['Jugador 1']))['WeightedELO'].values.item() +
                       (stats_xr.sel(match=match, player=match_df['Jugador 2']))['WeightedELO'].values.item() -
                       ((stats_xr.sel(match=match, player=match_df['Jugador 3']))['WeightedELO'].values.item() +
                       (stats_xr.sel(match=match, player=match_df['Jugador 4']))['WeightedELO'].values.item() ))

    # Neat goals
    neatgoals_attack_difference = calculate_differential_head2head(stats_xr, 'NeatGoalsAttackPlayed', match_df, match, player_local='Jugador 2', player_visitant='Jugador 4')
    neatgoals_defense_difference = calculate_differential_head2head(stats_xr, 'NeatGoalsDefensePlayed', match_df, match, player_local='Jugador 1', player_visitant='Jugador 3')

    # Nombre de partits guanyats en aquest dia
    winsmatchday_difference = ((stats_xr.sel(match=match, player=match_df['Jugador 1']))['WinPlayedMatchday'].values.item() +
                           (stats_xr.sel(match=match, player=match_df['Jugador 2']))['WinPlayedMatchday'].values.item() -
                           ((stats_xr.sel(match=match, player=match_df['Jugador 3']))['WinPlayedMatchday'].values.item() +
                            (stats_xr.sel(match=match, player=match_df['Jugador 4']))['WinPlayedMatchday'].values.item()))

    # Diferència d'ELO entre l'atacant d'un equip i el defensor de l'altre
    elo_attackh_defensea_difference = calculate_differential_opositeposition(stats_xr, 'ELOAttack', 'ELODefense', match_df, match, player_local='Jugador 2', player_visitant='Jugador 3')
    elo_defenseh_attacka_difference = calculate_differential_opositeposition(stats_xr, 'ELODefense', 'ELOAttack', match_df, match, player_local='Jugador 1', player_visitant='Jugador 4')

    # --- Paràmetres d'equip ---
    # Victòries en partits ajustats
    close_wins_local = frequencies_xr.sel(teammate=match_df['Jugador 1'], player = match_df['Jugador 2'])['CloseWinsPlayed'].values.item()
    close_wins_visitant = frequencies_xr.sel(teammate=match_df['Jugador 3'], player = match_df['Jugador 4'])['CloseWinsPlayed'].values.item()
    if np.isnan(close_wins_local): # Per CloseWinsPlayed, fem que el valor sigui 0.5 si l'original és NaN, que vol dir que no han jugat mai un Close Match junts
      close_wins_local = 0.5
    if np.isnan(close_wins_visitant):
        close_wins_visitant = 0.5
    close_wins_difference = close_wins_local - close_wins_visitant

    # Gols rebuts d'un defensor contra l'altre defensor i contra l'atacant rival
    # 'defender' rep els gols de 'defender_rival' o 'attacker_rival'
    receivedgoals_defense_defense_local = frequencies_xr.sel(defender = match_df['Jugador 1'], defender_rival = match_df['Jugador 3'])['ReceivedGoalsGamesDefenseDefense'].values.item()
    receivedgoals_defense_defense_visitant = frequencies_xr.sel(defender = match_df['Jugador 3'], defender_rival = match_df['Jugador 1'])['ReceivedGoalsGamesDefenseDefense'].values.item()
    receivedgoals_attack_defense_local = frequencies_xr.sel(defender = match_df['Jugador 1'], attacker_rival = match_df['Jugador 4'])['ReceivedGoalsGamesAttackDefense'].values.item()
    receivedgoals_attack_defense_visitant = frequencies_xr.sel(defender = match_df['Jugador 3'], attacker_rival = match_df['Jugador 2'])['ReceivedGoalsGamesAttackDefense'].values.item()
    receivedgoals_defense_defense_difference = receivedgoals_defense_defense_local - receivedgoals_defense_defense_visitant
    receivedgoals_attack_defense_difference = receivedgoals_attack_defense_local - receivedgoals_attack_defense_visitant

    # Nombre de partits guanyats per cada equip
    team_wins_local = frequencies_xr.sel(teammate = match_df['Jugador 1'], player = match_df['Jugador 2'])['TeammatesWinsPlayed'].values.item()
    team_wins_visitant = frequencies_xr.sel(teammate = match_df['Jugador 3'], player = match_df['Jugador 4'])['TeammatesWinsPlayed'].values.item()
    team_wins_difference = team_wins_local - team_wins_visitant

    # --- Agrupació de dades ---
    # Agrupem totes les dades del partit en una sola llista, que serà la fila de la matriu `X` corresponent a aquest partit. L'ordre dels paràmetres ha de ser el mateix que el de les columnes de `Stats_training`
    stats_match = stats_match + [elo_attackh_defensea_difference, elo_defenseh_attacka_difference,
                                     elo_difference, weighted_elo_difference, team_wins_difference, close_wins_difference,
                                 neatgoals_attack_difference, neatgoals_defense_difference, winsmatchday_difference,
                                 receivedgoals_defense_defense_difference, receivedgoals_attack_defense_difference]

    # Afegim el codi numèric de cada jugador
    #player_codes_match = [player_codes_dict[match_df['Jugador 1']], player_codes_dict[match_df['Jugador 2']],
    #                      player_codes_dict[match_df['Jugador 3']], player_codes_dict[match_df['Jugador 4']]]
    #stats_match = player_codes_match + stats_match

    # --- Desament de les dades ---
    # Desem la llista d'estadístiques d'aquest partit
    Stats_training.loc[len(Stats_training)] = stats_match

Stats_training.to_csv('../generated_files/stats_training.csv', index=False) # desem per poder obrir més tard

print('Saved stats training')

###############################################################
############## Preparació de les dades ########################
###############################################################
# Inspirat per https://github.com/NunonuN/ml-playground

title = "RAW DATA SET"
lenti = len(title)
print("="*lenti)
print(title)
print("-"*lenti + '\n')
print("Features")
print("-"*8)
print(f"Shape: {Stats_training.shape}\n")
print(Stats_training.head())
print(f"\nClass distribution:")
print(np.unique(Scores_coded, return_counts=True))
print(f"\nTarget")
print("-"*6)
print(f"{Scores_coded[:5] = }")
print(f"{Scores_coded[-5:] = }")
print('='*80)

# Retallem els datasets. Deixem fora els primers partits per aconseguir ELOs estables per cada jugador
threshold = 0.3 # descartem el primer X% de partits per entrenar el model
Scores_coded = Scores_coded[int(threshold*len(Scores_coded)):]
Stats_training = Stats_training[int(threshold*Stats_training.shape[0]):]

print("-"*6)
print(f"\nCroppping first {threshold*100} % of points")
print("-"*6)

###############################################################
################### Anàlisi PCA ###############################
###############################################################

# No PCA

###############################################################
################# Train / Test split ##########################
###############################################################

# split data into training and testing sets
X_class_trn, X_class_tst, y_class_trn, y_class_tst = (
  train_test_split(
    Stats_training,
    Scores_coded,
    test_size=0.15,
    random_state=RND_STATE
  )
)

print(f"Training set size: {X_class_trn.shape[0]}")
print(f"Testing set size: {X_class_tst.shape[0]}")

###############################################################
############## Estandaritzem els valors #######################
###############################################################
# Cal estandaritzar després de fer l'split, per evitar que la informació del test set es filtrés al train set.
# Per això, primer fem el split i després estandaritzem els valors.

scaler_class = preprocessing.StandardScaler() # define scaler
X_class_trn = scaler_class.fit_transform(X_class_trn) # fit scaler to training data and transform it
X_class_tst = scaler_class.transform(X_class_tst) # transform test data using the same scaler fitted on training data

# Desem l'StandardScaler() per fer-lo servir en un altre programa
joblib.dump(scaler_class, '../generated_files/scaler.pkl')

###############################################################
############## Entrenament del model ##########################
###############################################################

# ## Definició i entrenament del model
print("Searching for best hyperparameters...")
## Grid search for hyperparameter tuning (uncomment to run)
parameters = {
    # 1. Profunditat de l'arbre
    # Amb 15 variables, els arbres necessiten una mica més de marge per trobar combinacions,
    # però sense passar-se per evitar overfitting. 3, 5 i 7 acostumen a ser els valors clau.
    'max_depth': [3, 5, 7],

    # 2. Nombre d'arbres i taxa d'aprenentatge
    # Com que per a cada "estimator" en realitat es creen 6 arbres, posem valors moderats
    # perquè el GridSearch no es torni etern.
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],

    # 3. Control de sobreajust (Regularització)
    # 'min_child_weight' és crucial en multiclasse. Valors més alts (ex: 3 o 5) fan que l'arbre
    # sigui més conservador i no crei branques per a classes que tinguin molt pocs exemples.
    'min_child_weight': [1, 3, 5],

    # 4. Mostreig aleatori (Subsample)
    # Obliga a cada arbre a entrenar-se amb només un % de les files i un % de les 15 variables.
    # Això ajuda moltíssim a fer el model més robust i genèric entre les 6 classes.
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb = xgb.XGBClassifier()
clf = GridSearchCV(xgb, parameters, cv=5)
clf.fit(X_class_trn, y_class_trn)
print("Best parameters found: ", clf.best_params_)
xgb_class = clf.best_estimator_
y_pred_xgb = xgb_class.predict(X_class_tst)
y_pred_xgb_prob = xgb_class.predict_proba(X_class_tst)[:, 1]

###############################################################
################# Informe del model ###########################
###############################################################

title = "Game winning Classification – XGBoost"
lenti = len(title)
print(title)
print("-"*lenti)
print(f"Accuracy: {accuracy_score(y_class_tst, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_class_tst, y_pred_xgb, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_class_tst, y_pred_xgb, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_class_tst, y_pred_xgb, average='weighted'):.4f}")

###############################################################
################### Matriu de confusió ########################
###############################################################

y_pred = xgb_class.predict(X_class_tst)
print("Best params:", clf.best_params_)
print(classification_report(y_class_tst, y_pred_xgb))
cm_disp = ConfusionMatrixDisplay.from_predictions(y_class_tst, y_pred)
plt.savefig('../results/ML/Confusion_Matrix_Test_scores.png', dpi=300, bbox_inches='tight')
plt.clf()

###############################################################
#################### SHAP summary #############################
###############################################################

try:
    #X_shap = pd.DataFrame(scaler_class.transform(X_class_tst), columns=feature_names)
    X_shap = pd.DataFrame(X_class_tst, columns=feature_names)
except ValueError: # estem fent servir PCA
    X_shap = X_class_tst
explainer = shap.TreeExplainer(xgb_class)
shap_values = explainer(X_shap)

# Plot and save (feature names will appear because X_shap is a DataFrame)
shap.summary_plot(shap_values, X_shap, show=False)
plt.savefig('../results/ML/shap_summary_scores.png', dpi=300, bbox_inches='tight')
plt.clf()

###############################################################
#################### ROC-AUC curve #############################
###############################################################

# No és possible per classificacions no-binàries

###############################################################


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

