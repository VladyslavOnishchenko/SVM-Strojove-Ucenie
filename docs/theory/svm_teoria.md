# Metóda podporných vektorov (SVM)

## Čo je klasifikácia

Klasifikácia je jednou zo základných úloh strojového učenia. Jej cieľom je priradiť každý vstupný príklad do jednej z vopred definovaných kategórií. Napríklad e-mailový filter rozhoduje, či je správa spam alebo nie. Lekársky systém na základe výsledkov vyšetrení určí, či je pacient zdravý alebo chorý. Botanický klasifikátor rozozná druh kvetu podľa rozmerov jeho okvetných lístkov. Vo všetkých týchto prípadoch existuje konečná množina tried a úlohou modelu je naučiť sa, do ktorej triedy každý vstup patrí.

Algoritmus sa učí z trénovacej množiny — sady príkladov, pri ktorých vieme správne zaradenie. Z týchto príkladov extrahuje vzory a pravidlá, ktoré potom aplikuje na nové, dosiaľ nevidené vstupy. Čím lepšie model zachytí skutočné zákonitosti v dátach (a nie len náhodné šumy v trénovacej množine), tým presnejší bude pri klasifikácii nových príkladov.

## Princíp SVM

Metóda podporných vektorov hľadá rozdeľovaciu hranicu — hyperpriamu alebo hyperplochu — ktorá oddeľuje jednotlivé triedy v priestore príznakov. Pre dvojrozmerný prípad si to možno predstaviť ako priamku, ktorá rozdeľuje body dvoch tried na rovine. V trojrozmernom priestore je to rovina, v ešte vyšších dimenziách hovoríme o hyperrovine.

Čo odlišuje SVM od iných klasifikačných metód, je spôsob, akým túto hranicu vyberá. Existuje spravidla nekonečne mnoho hraníc, ktoré by dáta správne rozdelili. SVM vyberie tú, ktorá maximalizuje margin — vzdialenosť medzi hranicou a najbližšími bodmi každej triedy. Tieto najbližšie body sa nazývajú podporné vektory (support vectors) a práve oni určujú polohu a orientáciu hranice. Všetky ostatné trénovacie príklady, ktoré sú ďalej od hranice, na výsledok nemajú vplyv — keby sme ich zo trénovacej množiny vymazali, model by zostal rovnaký.

Prečo je väčší margin lepší? Model s maximálnym marginom má väčšiu rezervu a lepšie sa zovšeobecňuje na nové dáta. Intuitívne: ak leží hranica tesne vedľa niekoľkých bodov, malá zmena vstupných dát ju môže ľahko posunúť. Hranica uprostred voľného priestoru je robustnejšia voči variabilite nových príkladov.

V reálnych dátach triedy často nie sú perfektne separovateľné — niektoré body jednej triedy zasahujú do oblasti druhej triedy. SVM toto rieši tzv. mäkkým marginom (soft margin): model môže niektoré body misklasifikovať alebo nechať vo vnútri marginu, ale platí za to pokutu. Rovnováhu medzi šírkou marginu a počtom chýb riadi parameter C, ktorý je popísaný v sekcii o hyperparametroch.

## Jadrové funkcie (kernels)

Lineárna hyperrovina nestačí vždy na oddelenie tried — v mnohých reálnych problémoch je vzťah medzi príznakmi a triedami nelineárny. Jadrové funkcie (kernels) riešia tento problém tým, že implicitne transformujú vstupné dáta do vyššierozmerného priestoru, kde sú triedy separovateľné lineárne. Pozoruhodné je, že SVM to robí bez explicitného výpočtu tejto transformácie — stačí vypočítať skalárne súčiny v pôvodnom priestore pomocou jadrovej funkcie.

### Lineárne jadro (linear)

Lineárne jadro nevykonáva žiadnu transformáciu — hľadá priamu hyperpriamku v pôvodnom priestore príznakov. Je vhodné vtedy, keď sú triedy lineárne separovateľné, alebo keď je počet príznakov veľmi vysoký (napríklad pri klasifikácii textu, kde príznaky sú výskyty slov). Je najrýchlejším a najjednoduchším kernelom, ľahko interpretovateľným, a pri vhodných dátach dosahuje výsledky porovnateľné so zložitejšími jadrami.

### RBF jadro (Radial Basis Function)

RBF jadro je predvolenou voľbou v scikit-learn aj v tomto projekte a vo väčšine prípadov prináša najlepšie výsledky. Meria podobnosť dvoch bodov ako klesajúcu funkciu ich vzdialenosti — blízke body sú si podobné, vzdialené nie. Vďaka tomu dokáže modelovať nelineárne, krivočiare rozhodovacie hranice ľubovoľného tvaru. RBF je dobrou prvou voľbou vždy, keď nevieme, aký je tvar hranice medzi triedami. Citlivosť na vzdialenosť riadi parameter gamma.

### Polynomiálne jadro (poly)

Polynomiálne jadro modeluje interakcie medzi príznakmi pomocou polynómov zadaného stupňa. Je vhodné pre dáta, kde sú dôležité kombinácie viacerých príznakov naraz — napríklad pri rozpoznávaní obrazov, kde záleží na vzájomnej polohe pixelov. V porovnaní s RBF je pomalšie a citlivejšie na voľbu stupňa polynómu. Pri nevhodnom nastavení ľahko vedie k preučeniu.

### Sigmoidné jadro (sigmoid)

Sigmoidné jadro je inšpirované aktivačnou funkciou neurónových sietí. V praxi sa používa zriedkavo, pretože nesplňuje matematické podmienky platného jadra pre všetky kombinácie parametrov. Jeho výsledky sú väčšinou horšie ako RBF a línia medzi ním a jednoduchou neurónovou sieťou je nejasná. Pre väčšinu problémov je RBF alebo lineárne jadro lepšou voľbou.

## Hyperparametre

### Parameter C (regularizácia)

Parameter C riadi kompromis medzi dvoma protichodnými cieľmi: na jednej strane snahou o čo najširší margin, na druhej strane snahou o čo najmenej chybne klasifikovaných trénovacích bodov. Malá hodnota C (napríklad 0,01) uprednostňuje širší margin a toleruje viac chýb na trénovacích dátach — model je menej citlivý na odľahlé body a šum, ale môže byť príliš jednoduchý (underfitting). Veľká hodnota C (napríklad 100) nutí model správne klasifikovať každý trénovací bod, margin sa zužuje a model sa prispôsobuje aj šumovým príkladom, čo vedie k preučeniu (overfitting).

Dobrá analógia: malé C je ako tolerantný učiteľ, ktorý akceptuje, že žiak občas urobí chybu, a radšej ho naučí všeobecné princípy. Veľké C je ako prísny učiteľ, ktorý trestá každú chybu a žiak sa naučí naspamäť všetky príklady zo skriptá — ale na skúške so novými príkladmi zlyhá. Optimálna hodnota C závisí od dát a zvyčajne sa nastavuje pomocou krížovej validácie.

### Parameter gamma (pre RBF/poly/sigmoid)

Gamma určuje, ako ďaleko siaha vplyv jedného trénovacieho bodu pri výpočte rozhodovacích hraníc. Malá hodnota gamma znamená, že každý bod ovplyvňuje veľkú oblasť priestoru — výsledná hranica je hladká a jednoduchá. Veľká gamma obmedzuje vplyv každého bodu na jeho bezprostredné okolie — hranica je komplikovanejšia, sleduje tvar trénovacích dát veľmi tesne a ľahko dochádza k preučeniu.

Analógia: malá gamma je ako verejná mienka v celom meste — každý občan ovplyvňuje celkový náhľad mesta na nejakú tému. Veľká gamma je ako susedské klebety — každý ovplyvňuje len tých, čo bývajú hneď vedľa. Scikit-learn ponúka dva automatické režimy: `scale` (gamma = 1 / (počet príznakov * rozptyl dát)) a `auto` (gamma = 1 / počet príznakov). V praxi `scale` väčšinou funguje lepšie a je predvolenou hodnotou.

### Automatické ladenie (GridSearchCV)

Manuálne nastavenie hyperparametrov si vyžaduje skúsenosti a je časovo náročné. GridSearchCV systematicky vyskúša všetky kombinácie zadaných hodnôt C, gamma a kernelu a každú kombináciu vyhodnotí pomocou krížovej validácie. Na konci vyberie kombináciu s najvyšším priemerným skóre. Táto metóda zaručuje, že nevynecháme dobrú kombináciu parametrov, no za cenu výpočtového času — pri veľkej mriežke parametrov a veľkých dátach môže ladenie trvať minúty až hodiny. Pre väčšinu bežných datasetov je však automatické ladenie praktickým riešením, ktoré prekoná ručné nastavenie.

## Krížová validácia

Krížová validácia (k-fold cross-validation) je technika spoľahlivého odhadu výkonu modelu. Trénovacia množina sa rozdelí na k rovnako veľkých častí. Model sa natrénuje k-krát: vždy na k-1 častiach a vyhodnotí sa na zvyšnej časti, ktorá slúži ako validačná množina. Výsledné skóre sa spriemeruje cez všetkých k iterácií. Napríklad pri 5-fold CV sa model natrénuje a vyhodnotí päťkrát, zakaždým na inej pätine dát ako validácii.

Výhoda oproti jednoduchému rozdeleniu trénovacia/testovacia množina je spoľahlivosť — jedno náhodné rozdelenie môže byť šťastné alebo nešťastné, záleží od toho, ktoré príklady padnú do testovacieho setu. CV eliminuje túto náhodnosť tým, že každý príklad sa raz ocitne v validačnej množine. V tomto projekte sa CV aplikuje na trénovaciu množinu na hodnotenie modelu počas trénovania, zatiaľ čo separátna testovacia množina (odložená pred začiatkom trénovania) poskytuje finálne, nestranné vyhodnotenie výkonu.

## Predspracovanie dát (preprocessing)

SVM je obzvlášť citlivé na škálovanie príznakov, pretože rozhoduje na základe vzdialeností v priestore príznakov. Ak jeden príznak nadobúda hodnoty v rozsahu 0 až 10 000 (napríklad príjem v eurách) a iný príznak hodnoty 0 až 1 (napríklad binárny príznak), prvý príznak bude dominovať pri výpočte vzdialeností a model ho bude považovať za oveľa dôležitejší — nie preto, že naozaj dôležitejší je, ale len kvôli rozdielnej škále. StandardScaler transformuje každý číselný príznak tak, aby mal priemer 0 a rozptyl 1, čím sa všetky príznaky dostanú na porovnateľnú škálu.

V tomto projekte sú preprocessing a SVM zapojené do jedného sklearn Pipeline. To zaručuje, že parametre škálovania (priemer a rozptyl) sa naučia výhradne z trénovacích dát a rovnaká transformácia sa automaticky aplikuje pri predikcii nových vstupov — tým sa vylúči riziko úniku informácií z testovacích dát do trénovania (data leakage). Pre rôzne typy stĺpcov sa používajú rôzne transformácie: číselné stĺpce prechádzajú cez `StandardScaler`, kategoriálne stĺpce cez `OneHotEncoder` (každá kategória dostane vlastný binárny stĺpec) a binárne stĺpce cez `OrdinalEncoder` (hodnoty sa namapujú na číselné kódy).

## Hodnotenie modelu

Najjednoduchšou metrikou je presnosť (accuracy) — podiel správne klasifikovaných príkladov zo všetkých. Je intuitívna a ľahko interpretovateľná, no môže byť zavádzajúca pri nevyvážených triedach. Ak 95 % e-mailov nie je spam, model ktorý označí všetky správy za „nie spam" dosiahne 95 % presnosť — no je úplne zbytočný. Preto sú dôležité aj podrobnejšie metriky. Precision (presnosť triedy) udáva, koľko percent z príkladov označených ako pozitívnych naozaj pozitívnych je — meria spoľahlivosť pozitívnych predikcií. Recall (úplnosť) udáva, koľko percent skutočných pozitívnych príkladov model zachytil — meria, nakoľko model „nič nepremešká". F1 skóre je harmonický priemer precision a recall a vyrovnáva oba aspekty do jediného čísla.

Matica zámen (confusion matrix) poskytuje úplný obraz výkonu modelu. Je to tabuľka, kde riadky zodpovedajú skutočným triedam a stĺpce predikovaným triedam. Diagonála obsahuje správne klasifikované príklady, mimo diagonály sú chyby — môžeme vidieť, ktoré triedy si model pletie navzájom. Pre celkové zhrnutie sa používajú dva typy priemerov: macro avg spriemeruje metriky rovnomerne cez všetky triedy bez ohľadu na ich veľkosť, weighted avg zohľadňuje veľkosť každej triedy — pri nevyvážených dátach je weighted avg spravidla informatívnejší.

## Použité nástroje

**scikit-learn** — hlavná knižnica strojového učenia, poskytuje triedy `SVC`, `Pipeline`, `ColumnTransformer`, `GridSearchCV` a `cross_val_score`, ktoré tvoria jadro tohto projektu.

**pandas** — knižnica na spracovanie tabuľkových dát; používa sa na načítanie CSV súborov, manipuláciu s DataFrame a prípravu vstupov pre sklearn.

**numpy** — knižnica pre numerické operácie s poľami; zabezpečuje výpočty skóre, generovanie predikčnej mriežky pre vizualizáciu a ďalšie maticové operácie.

## Literatúra

Cortes, C., Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.

Pedregosa, F., Varoquaux, G., Gramfort, A. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

scikit-learn developers (2024). Support Vector Machines — scikit-learn documentation. Dostupné na: https://scikit-learn.org/stable/modules/svm.html
