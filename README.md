# ICE Champions League
Aquest repositori inclou tots els codis per recopilar i analitzar els resultats de la ICE Champions League. La millor manera d'executar el programa és descarregar el repositori i aleshores executar el fitxer `pipeline.sh` perquè es descarreguin les dades del Google Sheets i els facin tots els plots.

Canvieu el que cregueu oportú!

## Enllaços als outputs
# Season 3
[Last standings table](results/standings.md)

[ELO plot over matchdays](results/ELO_stats.png)

[Stats plot over matchdays](results/winplayed_stats.png)

[Team stats plot](results/frequencies.png)

# Season 2
[Last standings table](results/Season2/standings.md)

[Stats plot over matchdays](results/Season2/winplayed_stats.png)

[Team stats plot](results/Season2/frequencies.png)

## Sistema ELO
El sistema ELO és l'emprat per classificar el nivell de cada jugador. ## ELO rating

## ELO rating

El sistema ELO és un algoritme emprat en compteticions com els escacs per classificar els jugadors en funció del seu nivell. Si un jugador amb alt nivell guanya a un de baix nivell, la seva valoració no canviarà significativament. Ara bé, si és al revés, aleshores la puntuació del d'alt nivell baixarà notablement i la del de baix nivell pujarà bastant.

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