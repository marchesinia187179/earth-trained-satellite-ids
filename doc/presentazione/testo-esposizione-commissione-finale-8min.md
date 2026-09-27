# Testo per l'esposizione della prova finale

**Durata obiettivo:** circa **8 minuti complessivi**, includendo brevi pause e cambio slide.

---

## Slide 1 — Titolo

Buongiorno a tutti. Il lavoro che presento si intitola **“Analisi della Trasferibilità di NIDS basati su Machine Learning al Dominio Satellitare”** ed è stato svolto con il professor Mirco Marchetti e l'ingegner Dimitri Galli.

L'obiettivo è studiare quanto un sistema di rilevamento delle intrusioni, addestrato su traffico terrestre, mantenga le proprie capacità quando viene trasferito verso distribuzioni rappresentative di scenari terrestre-spaziali.

## Slide 2 — Introduzione

Il punto di partenza è un **Network Intrusion Detection System**, o NIDS, basato su apprendimento supervisionato. In una valutazione convenzionale, il modello viene addestrato e testato su dati appartenenti allo stesso dominio.

Il problema emerge quando il dominio operativo cambia. Nel passaggio da una rete terrestre a uno scenario integrato terrestre-spaziale, la distribuzione delle feature del traffico può differire da quella osservata durante il training: si verifica quindi un **domain shift**.

Di conseguenza, un classificatore può ottenere prestazioni elevate sul dominio sorgente e perdere sensibilità su una distribuzione diversa. Le sole prestazioni *in-domain* non sono quindi sufficienti: la robustezza deve essere verificata esplicitamente in condizioni **cross-domain**.

## Slide 3 — Problem Statement

Il problema di ricerca consiste quindi nel valutare la **generalizzazione cross-domain** di modelli addestrati sul dominio sorgente terrestre e applicati a distribuzioni target terrestre-spaziali.

L'analisi è organizzata attorno a quattro domande: la trasferibilità diretta in presenza di domain shift; l'effetto di una conoscenza target molto limitata; il confronto tra modelli aggregati e rilevatori biclasse specializzati; infine, l'effetto dell'architettura del classificatore su sensibilità, specificità e stabilità.

## Slide 4 — Contributi

Il contributo principale non consiste nella proposta di un nuovo algoritmo, ma nella realizzazione di un **framework sistematico di valutazione cross-domain**.

Il framework combina una rappresentazione comune dei dati, un training controllato in cui varia la quantità di conoscenza target e una cross-evaluation su benchmark eterogenei. In questo modo è possibile osservare come cambia la risposta del NIDS al variare sia dei dati di addestramento sia dell'architettura utilizzata.

A questa pipeline sono affiancati **PCA e KDE** per caratterizzare il domain shift, **SHAP** per analizzare la struttura decisionale, le metriche **TPR, TNR, F1 e Average Precision**, e il **test t di Welch** come analisi inferenziale esplorativa dei confronti multi-seed.

## Slide 5 — Metodologia: Preprocessing e Dataset

Come dominio sorgente è stato utilizzato **UNSW-NB15**, mentre il dominio target deriva da **STIN**, nelle componenti TER20 e SAT20.

Poiché i dataset utilizzano feature differenti, un operatore di mappatura e pulizia li riconduce a uno **spazio condiviso di 16 feature**, preservandone la coerenza semantica e dimensionale.

I benchmark sono costruiti imponendo un rapporto di **10 a 1 tra traffico benigno e malevolo** e una partizione **80/20 tra training e test**, ottenendo 5 benchmark aggregati e 11 biclasse.

Sono poi confrontati tre regimi. Nel **Source-only** la conoscenza è esclusivamente sorgente. In **INJECTION** vengono sostituiti soltanto 29 dei 9.300 campioni malevoli con osservazioni target, circa lo **0,31%**. Nei modelli **Hybrid**, invece, la componente malevola è interamente target. Si passa quindi da conoscenza target assente, a sparsa, fino a dominante.

## Slide 6 — Metodologia: Modelli e Validazione

Sono state valutate tre famiglie di classificatori ad albero: **Decision Tree**, **Random Forest** e **Histogram-based Gradient Boosting**.

Ogni configurazione è stata valutata su **10 seed**, mantenendo invariati i benchmark e modificando la componente stocastica dei classificatori. Per ogni replica vengono calcolate le metriche e costruita la matrice di cross-evaluation; successivamente si analizzano media e dispersione delle prestazioni.

La lettura privilegia **TPR e TNR**, cioè rispettivamente la capacità di riconoscere il traffico malevolo e quello nominale. F1 e Average Precision forniscono informazioni complementari, mentre il test t di Welch supporta il confronto tra le distribuzioni multi-seed mantenendo un carattere esplorativo.

## Slide 7 — Risultati: Effetto della Conoscenza Target Sparsa

Il primo risultato rilevante riguarda la conoscenza target sparsa.

La slide sintetizza l'intera cross-evaluation. **INJECTION** raggiunge una TPR media di circa **0,89 per DT, 0,80 per HGB e 0,90 per RF**, risultando sensibilmente superiore sia alla sintesi Source-only sia a quella Hybrid.

Separando il solo dominio target, la TPR media di INJECTION raggiunge circa **0,97 per DT, 0,83 per HGB e 0,99 per RF**. Il miglioramento rispetto ai modelli Source-only è quindi marcato.

Questo incremento non è accompagnato da una perdita sostanziale sul dominio sorgente: la risposta media sui benchmark NB15 rimane quasi invariata. I modelli Hybrid mostrano invece il comportamento opposto: sono molto efficaci sul target, ma perdono quasi completamente la capacità di riconoscere gli attacchi sorgente.

Nelle condizioni analizzate, una quota target estremamente ridotta risulta quindi associata a un compromesso più uniforme tra **adaptation**, cioè acquisizione di conoscenza sul nuovo dominio, e **retention**, cioè conservazione della conoscenza sorgente.

## Slide 8 — Risultati: Generalizzazione e Specializzazione

Il secondo risultato riguarda la granularità della componente malevola.

Sia nel dominio sorgente sia nel target, i modelli **aggregati** mostrano mediamente una TPR superiore ai modelli biclasse quando la valutazione comprende distribuzioni differenti da quella usata nel training.

Gli specialistici possono ottenere sensibilità elevate sulla singola famiglia appresa, ma risultano più dipendenti dalla corrispondenza tra training e test. L'aggregazione di più famiglie d'attacco è invece associata a una risposta più uniforme.

Sono comunque presenti eccezioni e trasferimenti locali tra specifiche famiglie. Il risultato evidenzia quindi un compromesso tra **specializzazione locale** e **ampiezza della generalizzazione**, senza indicare un approccio universalmente migliore.

## Slide 9 — Conclusioni: Risposte alle Domande Sperimentali

Le quattro domande di ricerca possono essere sintetizzate così.

Per la **RQ1**, la trasferibilità diretta dei modelli Source-only è limitata e dipendente dalla distribuzione target: una buona prestazione sul sorgente non garantisce robustezza cross-domain.

Per la **RQ2**, INJECTION mostra che una conoscenza target molto sparsa può essere associata a una maggiore generalizzazione, preservando sostanzialmente la risposta sorgente. Lo 0,31% non rappresenta però una quantità minima o ottimale.

Per la **RQ3**, i modelli aggregati presentano una generalizzazione generalmente più uniforme, mentre gli specialistici dipendono maggiormente dalla famiglia d'attacco.

Per la **RQ4**, l'architettura modula sensibilità, specificità e stabilità. Nella configurazione INJECTION, Random Forest presenta il profilo predittivo complessivamente più favorevole tra quelli misurati; HGB privilegia maggiormente la specificità e DT mostra un comportamento intermedio. Il risultato rimane però riferito alle configurazioni sperimentali adottate e non identifica un'architettura universalmente ottimale.

## Slide 10 — Takeaways e Sviluppi Futuri

Concludo con tre messaggi principali.

Primo: **la prestazione in-domain non garantisce robustezza cross-domain**.

Secondo: **una conoscenza target estremamente limitata può favorire la generalizzazione preservando il sorgente**, nelle condizioni investigate.

Terzo: la robustezza è un compromesso tra **adaptation e retention**, non la massimizzazione della prestazione su un singolo dominio.

Gli sviluppi futuri principali sono estendere il domain shift anche al traffico benigno e ad ulteriori dataset target, studiare sistematicamente la quantità di campioni INJECTION e il trade-off decisionale, e integrare misure di latenza, memoria, risorse e consumo energetico per valutarne l'applicabilità in deployment.

In sintesi, un NIDS destinato a infrastrutture eterogenee dovrebbe essere valutato non soltanto per quanto apprende un nuovo dominio, ma soprattutto per **quanto riesce ad adattarsi a nuove distribuzioni senza perdere la conoscenza già acquisita**.

Grazie per l'attenzione.
