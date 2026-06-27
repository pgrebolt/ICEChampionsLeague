# ICE Champions League
Aquest repositori inclou tots els codis per recopilar i analitzar els resultats de la ICE Champions League.
La millor manera d'executar el programa és descarregar el repositori amb `git clone` i aleshores executar el fitxer `pipeline.sh` perquè es descarreguin les dades del Google Sheets i els facin tots els plots.
Per si és necessari, s'inclou el fitxer `requirements.txt` amb els paquets necessaris.

## Com executar el programa
Fes un `git clone` del repositori per descarregar tots els fitxers a l'ordinador. Des de la terminal, navega fins
al directori on has descarregat el repositori i, dins la carpeta `src\`, executa el fitxer `pipeline.sh` amb el següent comandament:

```bash
    pipeline.sh -s [Season]
```

on `[Season]` és el número de temporada que vols analitzar. Per exemple, si vols analitzar la temporada 5, el comandament seria:

```bash
    pipeline.sh -s 5
```

Si no especifiques cap temporada, el programa analitzarà totes les temporades disponibles i farà el càlcul històric de resultats i estadísiques.

Si `pipeline.sh` no s'executa correctament, potser és perquè l'alias de `python` no està configurat. En aquest cas, obre `pipeline.sh` i canvia `python` per l'alias que tinguis configurat al teu ordinador.

El funcionament modular de `pipeline.sh` és el següent:
1. Descarrega les dades de la temporada especificada a través de Google Sheets i desa el fitxer `.csv` corresponent.
2. Calcula les estadístiques individuals de cada jugador i desa el fitxer `.nc` corresponent.
3. Calcula les estadístiques per parelles i desa el fitxer `.nc` corresponent.
4. Dibuixa els plots de les estadístiques individuals i desa els fitxers `.png` corresponents.
5. Dibuixa els plots de les estadístiques per equips i desa els fitxers `.png` corresponents.

Opcionalment, també s'inclou un fitxer a partir del qual es crea un model de predicció amb Machine Learning.

## How to Run the Program

Run a `git clone` of the repository to download all the files to your computer. From the terminal, navigate to the directory where you downloaded the repository and, inside the `src/` folder, run the `pipeline.sh` file with the following command:

```bash
pipeline.sh -s [Season]
```

where `[Season]` is the season number you want to analyze. For example, if you want to analyze season 5, the command would be:

```bash
pipeline.sh -s 5
```

If you do not specify a season, the program will analyze all available seasons and compute the historical results and statistics.

If `pipeline.sh` does not run correctly, it may be because the `python` alias is not configured. In that case, open `pipeline.sh` and replace `python` with the alias configured on your computer.

The modular workflow of `pipeline.sh` is as follows:
1. Download the data for the specified season through Google Sheets and save them as `.csv`files.
2. Calculate the individual statistics for each player and save the corresponding `.nc` file.
3. Calculate the pair statistics and save the corresponding `.nc` file.
4. Generate plots for the individual statistics and save the corresponding `.png` files.
5. Generate plots for the team statistics and save the corresponding `.png` files.

Optionally, the project also includes a file used to create a Machine Learning prediction model.

## Enllaços als outputs
[Històric de resultats](results/historical/)
[Inferència bayesiana](results/Bayesian_prediction/)

[Season 6](results/Season6/)
[Season 5](results/Season5/)
[Season 4](results/Season4/)
[Season 3](results/Season3/)
[Season 2](results/Season2/)

## Sistema ELO
El sistema ELO és l'emprat per classificar el nivell de cada jugador. ## ELO rating

## ELO rating

El sistema ELO és un algoritme emprat en competicions com els escacs per classificar els jugadors en funció del seu nivell. Si un jugador amb alt nivell guanya a un de baix nivell, la seva valoració no canviarà significativament. Ara bé, si és al revés, aleshores la puntuació del d'alt nivell baixarà notablement i la del de baix nivell pujarà bastant.

Sigui un jugador A amb puntuació $s_A$ i un jugador B amb puntuació $s_B$, aleshores la probabilitat que guanyi A en un enfrontament ve descrita, segons el model, com

$$P_A(s_A, s_B) = \frac{1}{1 + 10^{(s_B - s_A) /400}}$$

Si 1 denota victòria i 0 denota derrota, després d'un enfrontament entre A i B on A ha guanyat, les puntuacions s'actualitzen de la següent manera:

$$ s_A = s_A + K\cdot R_p \cdot (1 - P_A(s_A, s_B))$$
$$ s_B = s_B + K\cdot R'_p \cdot (0 - P_B(s_B, s_A))$$

on $K = 30$ és una constant i $R_p$ és una regularització/ponderació que té en compte el rendiment del jugador a la victòria..

Pel nostre cas, considerarem un ELO en posicions ofensives i un ELO en posicions defensives. Cada jugador comenaça la competició amb 1000 punts en cada posició, i s'anirà actualitzant en funció dels seus resultats a cada partits. Per calcular l'ELO del rival al càlcul, es calcula la mitjana ponderada d'ELOs del rival. La ponderació té en compte el nombre de partits que ha jugat l'atacant i el defensor en llurs posicions. A més, la ponderació $R_p$ dependrà de quants gols hagi anotat/rebut l'atacant/defensor i de si ha guanyat o perdut el partit. Per cada jugador, el rendiment es calcula com:

$$ R_{p,i} = \frac{r_i}{\sum_k r_k} $$

on $r_i$ és el rendiment particular de cada jugador i $\sum_k r_k$ representa el rendiment total de l'equip. Així, $R_{p,i}$ està normalitzat a 1. El valor $r_i$ depèn de si el jugador és atacant o defensor:

$$ r_i (\text{atacant}) =  \mu \cdot \left(\frac{\text{Gols anotats}}{3} \right) + \lambda \cdot \left( 1- \frac{\text{Gols rebuts}}{3} \right) $$
$$ r_i (\text{defensor}) = \mu \cdot \left( 1 - \frac{\text{Gols rebuts}}{3} \right) + \lambda \cdot \left( \frac{\text{Gols anotats}}{3} \right) $$

on $\mu = 0.7$ i $\lambda = 0.3$ són dos paràmetres arbitraris que ponderen l'activitat defensiva i ofensiva del defensor.

Ara bé, tal i com està escrit, si un defensor perd i no ha anotat cap gol tindrà $r_i = 0 \Longrightarrow R_i = 0$ i, per tant, no se li descomptaria cap punt! És per això que, en el cas que l'equip hagi perdut, el rendiment $r_i$ s'ha de recalcular per tal de penalitzar la derrota d'aquesta manera:

$$ r_i' = 1 - r_i$$

Així, el rendiment pel cas de la derrota es calcula com $R'_{p,i} = \frac{r'_i}{\sum_k r'_k}$.

Finalment, cal destacar que en cada càlcul de $r_i$ imposem un rendiment mínim de 0.1. Així, encara que un jugador obtingui una puntuació de $r_i = 0$, aquesta passarà a ser 0.1. Això evita conflictes quan es ponderi a l'hora de calcular $R_{p, i}$.

## Resultats de la 1a temporada
Victòries / Partit : Víctor

Gols / Partit: Pau

Millor parella: Víctor i Pedro

Pitjor parella: Simone i Pedro