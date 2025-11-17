#!/usr/bin/env python
# coding: utf-8

# ## Endeví de resultats
# 
# Aquest codi inclou un model de Deep Learning per predir els resultats d'un partit a partir de les dades històriques de partits.

# In[1]:


import numpy as np
import pandas as pd
import xarray as xr

import joblib

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Per poder entrenar el model, ens cal donar-li informació de qui està jugant (amb quins punts forts i punts febles) i quin és el resultat. Per tant, hem de llegir el fitxer amb les estadístiques de cada jugador i també el fitxer amb el registre històric de resultats.

# In[2]:


# Llegim l'històric de partits i resultats
matches_df = pd.read_csv('results.csv')
matches_df


# In[3]:


# Llegim les estadístiques de cada jugador
stats_xr = xr.open_dataset('stats.nc', engine='scipy')
stats_xr


# Construïm una matriu amb els paràmetres dels jugadors a pista (matriu `X`) i una matriu amb el resultat (matriu `y`). La matriu `y` l'extraiem directament de `matches_df`. La matriu `X` la construïm a partir dels jugadors que són a pista a cada partit (de `matches_df`) i les estadístiques de `stats_xr`.
# 
# Ens cal codificar els resultats. Els possibles resultats són 6: '3-0', '3-1', '3-2', '0-3', '1-3', '2-3'. Perquè el model pugui treballar bé, passarem aquests possibles valors a codis (0, 1, 2, 3, 4, 5), a partir dels quals en farem l'entrenament. Per usos futurs, també codificarem el guanyador (local/vistant).

# In[4]:


encoder_winning = preprocessing.LabelEncoder() # codificador d'etiquetes ('Local'/'Visitant') a nombres (0/1)
Results_winning_training = encoder_winning.fit_transform(matches_df['Guanyador'].values.astype(str))

# Array amb tots els resultats que farem servir per l'entrenament del model (0='Local', 1='Visitant')
Results_winning_training


# In[5]:


# El mateix que abans, però per tots els possibles resultats
# Llista amb tots els resultats en format unificat (6 resultats possibles)
Scores = [str(matches_df['Local'].iloc[i])+'-'+str(matches_df['Visitant'].iloc[i]) for i in range(matches_df.shape[0])]
Scores = np.array(Scores)

# Codifiquem els resultats (0-3: 0, 1-3: 1, ..., 3-2:5)
encoder_scores = preprocessing.LabelEncoder()
Scores_training = encoder_scores.fit_transform(Scores.astype(str))

# Desem l'StandardScaler() per fer-lo servir en un altre programa
joblib.dump(encoder_scores, 'encoder_scores.pkl')

Scores_training


# Seguidament construïm la matriu `X` que conté les dades dels jugadors al camp. Primer definim quins paràmetres tenim en compte pels atacants i pels defensors. Després, construïm la matriu on, fila per fila, hi ha tots els paràmetres dels jugadors.

# In[6]:


# Paràmetres que considerem al model, en funció de si el jugador és atacant o defensor
considered_stats_defense = ['GamesPlayed', 'WeightedELO', 'PlayedDefense', 'WinPlayedDefense', 'ScoredDefensePlayed', 'ReceivedDefensePlayed', 'ELODefense', 'DefenseIndex']
considered_stats_attack = ['GamesPlayed', 'WeightedELO', 'PlayedAttack', 'WinPlayedAttack', 'ScoredAttackPlayed', 'ReceivedAttackPlayed', 'ELOAttack', 'AttackIndex']

# Dataframe on hi desem tots els paràmetres d'avaluació de cada jugador
columns = [stat_def+'1' for stat_def in considered_stats_defense] + [stat_att+'2' for stat_att in considered_stats_attack] +\
            [stat_def+'3' for stat_def in considered_stats_defense] + [stat_att+'4' for stat_att in considered_stats_attack] # noms de les columnes
Stats_training = pd.DataFrame(columns = columns)

for nmatch in range(len(matches_df['D'])): # per cada partit disputat
    match_df = matches_df.iloc[nmatch] # triem les dades d'aquest partit

    matchday = match_df['D']-1 # número de matchday

    # Llista on hi desarem els valors des les estadístiques de cada jugador que hi ha al camp, amb el mateix ordre que `columns`
    stats_match = []
    for player in match_df[['Jugador 1', 'Jugador 2', 'Jugador 3', 'Jugador 4']]:
        if (player == match_df['Jugador 1']) or (player == match_df['Jugador 3']): # defensors
            # Triem les estadístiques dels jugadors en aquest partit
            player_stats = stats_xr.sel(matchday=matchday, player=player)[considered_stats_defense].to_array().values
        elif (player == match_df['Jugador 2']) or (player == match_df['Jugador 4']): # atacants
            # Triem les estadístiques dels jugadors en aquest partit
            player_stats = stats_xr.sel(matchday=matchday, player=player)[considered_stats_attack].to_array().values
        stats_match = stats_match + list(player_stats) # adjuntem les estadístiques del jugador a les dades d'aquest partit
    # Desem la llista d'estadístiques d'aquest partit
    Stats_training.loc[len(Stats_training)] = stats_match

Stats_training


# Ara ja tenim les dades `X` per entrenar el model. Abans d'entrenar-lo, estandaritzem els valors. Això és un pas comú en IA quan es tenen paràmetres amb diferents escales de valors.

# In[7]:


scaler = preprocessing.StandardScaler()
Stats_training_stand = scaler.fit(Stats_training).transform(Stats_training.astype(float))


# In[8]:


# Desem l'StandardScaler() per fer-lo servir en un altre programa
joblib.dump(scaler, 'scaler.pkl')


# Separem tots els partits que tenim en una mostra d'entrenament (train) i una mostra de test. Això ens permetrà avaluar el rendiment del nostre model. En aquest cas, de la mostra total un 10% serà de test i el 90% restant serà per entrenar el model.

# In[9]:


X_train, X_test, y_winning_train, y_winning_test, y_score_train, y_score_test = train_test_split(Stats_training_stand, Results_winning_training, Scores_training, test_size=0.2) # random_state=42


# In[10]:


# Passem els valors y al codi numèric
y_score_train_labels = encoder_scores.inverse_transform(y_score_train)
y_score_test_labels = encoder_scores.inverse_transform(y_score_test)

y_winning_train_labels = encoder_winning.inverse_transform(y_winning_train)
y_winning_test_labels = encoder_winning.inverse_transform(y_winning_test)


# ## Definició i entrenament del model

# El model de xarxa neuronal (Deep Learning) el crearem amb Keras, dins de TensorFlow. Primer crearem el model, l'entrenarem i després l'avaluarem.

# #### Model de predicció de l'equip guanyador

# Aquest model és una xarxa sequencial (una capa darrere l'altra) que acaba amb una funció d'activació softmax. Aquesta funció d'activació la interpretem com la probabilitat que ocorri cada possible resultat (3-0, 3-1, 3-2...). L'arquitectura del model (quines capes té i amb quina dimensionalitat) és de tria arbitrària.

# In[11]:


# Definim el model keras
input_layer = layers.Input(shape = (X_train.shape[1],)) # capa d'input
x = layers.Dense(32, activation = 'relu')(input_layer) # capes internes
x = layers.Dense(16, activation = 'relu')(x)
x = layers.Dense(16, activation = 'relu')(x)
x = layers.Dense(8, activation = 'relu')(x)
x = layers.Dense(4, activation = 'relu')(x)
x = layers.Dense(2, activation = 'relu')(x)

score_output = layers.Dense(6, activation  = 'softmax', name='score_output')(x) # probabilitat associada a cada resultat

# Agrupem totes les capes
model = models.Model(inputs=input_layer, outputs=[score_output])

# Resum del model
model.summary()


# In[12]:


# Compilem el model. Definim quina funció de cost fem servir, l'optimitzador i les mètriques que fem servir per avaluar-lo
model.compile(optimizer='adam', loss={
            'score_output': 'sparse_categorical_crossentropy'
            },
              metrics={
                  'score_output': ['accuracy']
              }
             )


# In[13]:


# Entrenem el model. Definim quantes èpoques fem servir i la mida batch
model.fit(X_train, y_score_train,
          validation_data = (X_test, y_score_test),
          epochs=500, batch_size = 10, verbose=False) # PER FER: DEFINIR UN CALLBACK QUE ATURI L'ENTRENAMENT FINS UQE S'ARRIBI A UN LOSS DETERMINAT


# ### Avaluació del model

# A continuació es fan una sèrie d'avaluacions per comprovar que el model funciona correctament. Numèricament, extraurem la precisió del model. Visualment, dibuixarem les matrius de confusió, que ens indiquen quins són els punts forts i els punts febles del model.

# In[14]:


# Avaluem el model amb els valors de test (extracte de les mètriques del model)
results = model.evaluate(
    X_test,
    y_score_test)

# Desem els resultats de les mètriques de cada paràmetre avaluat (marcador i equip guanyador) com un diccionari
metrics = dict(zip(model.metrics_names, results))

print("Total loss:", metrics['loss'])
print("Compile_metrics", metrics['compile_metrics'])
#print("Winner accuracy:", metrics['winner_output_accuracy'])
#print("Score accuracy:", metrics['score_output_accuracy'])


# A continuació treurem les llistes de valors que ens permeten avaluar-ne la precisió.

# In[15]:


# Treiem la predicció per cada cas d'input (amb els valors train)
score_pred = model.predict(X_train)
score_pred_class = score_pred.argmax(axis = 1) # triem quin és el resultat que és la probabilitat més alta

# Passem el codi numèric a etiquetes amb els marcadors dels partits
score_pred_labels = encoder_scores.inverse_transform(score_pred_class)
#predictions


# In[16]:


# Conjunt de paràmetres que avaluen el model
print(classification_report(y_score_train, score_pred_class))


# In[17]:


# En funció del resultat predit, trobem quin equip es prediu que guanya
winner_labels = []
for i in range(len(score_pred_labels)):
    local, visitant = score_pred_labels[i].split('-')
    if local > visitant:
        winner_label = 'Local'
    else:
        winner_label = 'Visitant'
    winner_labels.append(winner_label)

winner_labels = np.array(winner_labels)
winner_labels_encoded = encoder_winning.transform(winner_labels)
winner_labels_encoded


# In[18]:


# Precisió de la predicció. Comparem els marcadors predits (score_pred_class) i els marcadors reals (y_score_test)
print("Accuracy in score:", accuracy_score(y_score_train, score_pred_class) * 100, "%")


# In[19]:


# Imprimim la matriu de confusió, que ens indica els punts forts i punts febles del model
ConfusionMatrixDisplay.from_predictions(y_score_train_labels, score_pred_labels)


# In[20]:


# Precisió de la predicció. Comparem els marcadors predits (score_pred_class) i els marcadors reals (y_score_test)
print("Accuracy in winning team:", accuracy_score(y_winning_train, winner_labels_encoded) * 100, "%")


# In[21]:


# Imprimim la matriu de confusió, que ens indica els punts forts i punts febles del model
ConfusionMatrixDisplay.from_predictions(encoder_winning.inverse_transform(y_winning_train), winner_labels)


# In[22]:


# Desem el model i els pesos
model.save('sequential.keras')

