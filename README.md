# Employee Attrition: previsione dell'abbandono aziendale

Classificazione binaria (`Left` / `Stayed`) su un dataset sintetico di dipendenti. Il progetto confronta nove modelli con cross-validation, corregge le probabilità con le prior reali, sceglie la soglia di classificazione con un vincolo sulla specificità e valuta il modello finale su un test set separato.

Questo repository contiene il **porting in Python (scikit-learn)** di un progetto sviluppato in origine in **R con caret** per il corso di Machine Learning dell'Università degli Studi di Milano-Bicocca (A.A. 2025/2026).

## Crediti

Il progetto originale in R è stato svolto in gruppo da Mattia Sorgia, Giacomo Regazzoni e Lorenzo Cavallari. Il porting in Python è di Mattia Sorgia.

## Risultati in breve

- **Modello scelto: Gradient Boosting (GBM)**, con AUC 0.851 sul validation e **0.853 sul test**.
- Rete neurale, logistica, Lasso e LDA restano vicini (AUC validation tra 0.841 e 0.844). Decision Tree e Naive Bayes sono i più deboli.
- **Soglia finale: 0.088** (sulla scala delle probabilità corrette), scelta massimizzando la sensibilità con vincolo di specificità ≥ 0.60. Sul validation: sensibilità 0.884, specificità 0.601, accuratezza 0.736.
- Sul test con la soglia finale: **sensibility = 0.817 | specificity = 0.701 | accuracy = 0.756**.
- La replica in Python riproduce la soglia dello script R (0.088 contro 0.089) e quasi la stessa confusion matrix (vedi sotto).

## Dataset

- Fonte: [Employee Attrition Dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset) (Stealth Technologies, Kaggle, 2024). I dati sono **sintetici**.
- `train.csv`: 59.598 righe, divise con split stratificato 80/20 in 47.678 di training e 11.920 di validation (seed 42). `test.csv`: 14.900 righe.
- 22 predittori (7 numerici e 15 categorici). La colonna `Employee ID` è scartata perché è un identificatore. Nessun valore mancante.
- Target `Attrition`: nel train 52,5% `Stayed` e 47,5% `Left`.

I file CSV non sono inclusi: scaricali da Kaggle e mettili nella cartella principale.

## Struttura del repository

```
common.py          dati, split, analisi quantitativa preliminare, Cramér's V, set di colonne A/B per il training di modelli che necessitano di model selection, funzione run() per far partire i modelli
File dei modelli:
model_tree.py      model_nb, model_logistic, model_lasso,
model_rf.py        model_knn, model_lda, model_nnet, model_gbm
compare.py         confronto dei modelli in CV (tabella e boxplot)
roc.py             curve ROC sul validation
lift.py            correzione dei prior e curve gain/lift del GBM
soglia.py          studio della soglia sul GBM
score_test.py      scoring finale su test.csv
models/            modelli e probabilità salvati con joblib (non versionati)
output/            grafici e predizioni
```

Ogni modello ha il proprio file e viene salvato su disco, così non serve riaddestrarlo a ogni esecuzione. `common.py` rigenera i grafici dell'analisi esplorativa a ogni import.

## Come eseguire

```bash
pip install -r requirements.txt
python model_gbm.py     # e gli altri model_*.py
python roc.py
python lift.py
python soglia.py
python score_test.py
```

## Metodologia

1. **Split holdout 80/20** stratificato. Il validation non entra mai nell'addestramento né nella cross-validation.
2. **Metrica: AUC-ROC**, calcolata con CV a 10 fold stratificata, con gli stessi fold per tutti i modelli.
3. **Analisi esplorativa.**
   - Correlazioni tra le numeriche: l'unica coppia rilevante è Age con Years at Company (0.54). Nessuna coppia supera 0.8.
   - Cramér's V tra ogni categorica e il target. Associazioni forti (> 0.15): Job Level (0.315), Marital Status (0.284), Remote Work (0.219), Work-Life Balance (0.196). Associazioni deboli (< 0.05): Company Size, Innovation Opportunities, Job Role, Leadership Opportunities, Employee Recognition.
4. **Due set di variabili.** `trainA`: tutti i 22 predittori. `trainB`: 17 predittori, senza le 5 categoriche deboli, per i modelli più sensibili alla dimensionalità.
5. **Preprocessing dentro la pipeline**, quindi appreso solo sui dati di training di ogni fold: scaling delle numeriche, one-hot delle categoriche e rimozione delle dummy quasi costanti.

### Modelli

| Modello | Set | Tuning | AUC CV (R) | AUC validation (Python) |
|---|---|---|---|---|
| Decision Tree | A | `ccp_alpha`, 25 valori da 0.00001 a 0.048 | 0.787 | 0.788 |
| Naive Bayes | A | nessuno | 0.822 | 0.814 |
| Regressione logistica | A | nessuno | 0.840 | 0.842 |
| Lasso (solo Python) | A | `C`, 20 valori derivati da λ da 0.001 a 0.096 | n.d. | 0.842 |
| Random Forest | A | `max_features`, 3 valori | 0.830 | 0.837 |
| KNN | B | k da 25 a 69, scelto su un campione di 10.000 righe | 0.821 | 0.820 |
| LDA | B | nessuno | 0.840 | 0.841 |
| Rete neurale (1 strato nascosto) | B | 7/10/15/20 neuroni × regolarizzazione 0.005-0.05 | 0.843 | 0.844 |
| GBM | A | 18 combinazioni: alberi, profondità, learning rate | 0.851 | 0.851 |


## Prior reali e correzione delle probabilità

Il dataset è quasi bilanciato (47,5% `Left`), ma le prior reali sono state stimate intorno a **85% `Stayed` e 15% `Left`** [Ravio. Employee Attrition Rate Report. 2025]. Le probabilità del modello sono corrette con le prior reali:

```
p_corretta = p · w_L / (p · w_L + (1 − p) · w_S)
w_L = 0,15 / quota di Left nel train    w_S = 0,85 / quota di Stayed nel train
```

La correzione conserva l'ordine delle osservazioni, quindi **non cambia ROC, gain e lift**. Cambia la scala delle probabilità, e per questo la soglia (0.088) è molto più bassa di 0.5.

## Curve gain e lift

Il GBM ha lift cumulato ≈ 2.06 nel primo decile (2.05 in R). Il lift è calcolato sul validation, dove i `Left` sono il 47,5%, quindi è relativo a quella quota e non al 15% della popolazione reale.

## Scelta della soglia

Si è supposto che un dipendente che se ne va senza preavviso sia un errore più costoso di uno che rimane pur essendo segnalato come a rischio. Per questo la soglia massimizza la **sensibilità**. Il vincolo sulla specificità serve a evitare la soluzione banale di classificare tutti come `Left`.

| Regola (validation, GBM) | Soglia | Sensibilità | Specificità | Accuratezza |
|---|---|---|---|---|
| **Specificità ≥ 0.60 (finale)** | 0.088 | 0.884 | 0.601 | 0.736 |
| Specificità ≥ 0.70 | 0.125 | 0.810 | 0.700 | 0.752 |
| Media geometrica di sens. e spec. | 0.147 | 0.770 | 0.747 | 0.758 |

Soglia selezionata per massimizzare la **sensibilità**

**Soglia finale:**

| | Soglia | Veri Stayed | Falsi Left | Falsi Stayed | Veri Left |
|---|---|---|---|---|---|
 0.088 | 3758 | 2494 | 656 | 5012 |

## Risultati sul test set

- Quota prevista Left: **0.543**
- AUC del GBM: **0.853**.
- Con la soglia finale: 
[[5518 2350]
[1287 5745]]
sens = 0.817 | spec = 0.701 | acc = 0.756.

Le predizioni sono salvate in `output/predizioni_test.csv`.

## Limiti

- Il dataset è sintetico e le prior reali sono una stima, non un dato osservato.
- La soglia è scelta e valutata sullo stesso validation, quindi le metriche su quel set sono leggermente ottimistiche. Il test set, non usato per nessuna scelta, è il riferimento.

## Possibili estensioni

- Integrare informazioni economiche aziendali (costo di sostituzione, costo dell'intervento) per trasformare la scelta della soglia in una decisione orientata al profitto.
- Scegliere la soglia con una matrice dei costi esplicita, invece di un vincolo sulla specificità.
