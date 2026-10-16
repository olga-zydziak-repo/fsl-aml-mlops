# SAML-D: typologie AML jako zadanie few-shot w pipelinie MLOps

Raport z całego łańcucha na SAML-D: od okien kont (v0), przez falsyfikację cech sieciowych (v1), przejście na przypadki, freeze i kampanię 5-foldową, po active learning i explainability w pipelinie. Każdy krok miał kryterium ustalone przed uruchomieniem. Wyniki są opisane razem z tym, czego nie dowodzą.

## Najważniejsze

**Active learning oszczędza około połowy etykiet.** Model sam wybiera, który przypadek analityk ma oznaczyć jako następny (strategia `margin`: przypadek najbliżej granicy dwóch prototypów). Trafność, którą wybór losowy osiąga przy 24 etykietach (0,834), `margin` osiąga przy 11. Kryterium ustalone przed uruchomieniem przeszło: średnia przewaga nad losowym wyborem w budżecie 8–16 etykiet to +0,042, 95% CI [+0,022; +0,061]. Przewaga jest dodatnia na każdym z pięciu foldów. Ten sam protokół działa teraz jako komponent pipeline'u. Na foldzie 3 daje 10 etykiet zamiast 24 i trafność 0,925 vs 0,831 przy 24 etykietach.

**Pętla z analitykiem działa na prawdziwym foldzie.** Cztery rundy po 8 odpowiedzi na foldzie 3 podniosły trafność wzorców typologii z 80,1% do 92,8% bez żadnego trenowania. Bramka rundy odrzuciła jedną z nich, bo jedna typologia się w niej załamała (Layered_Fan_In z 80,5% do 48,8%), choć średnia spadła tylko o 1,7 pp. Czwarta runda wystartowała sama: wrzucenie pliku do `oznaczone/` uruchomiło pipeline rundy przez funkcję Cloud Run, bez ręcznego startu.

**Kolejka do oznaczenia trafia w błędy modelu i ma dokładne wyjaśnienie.** W przebiegu pipeline'u każdy przypadek z kolejki był w danej chwili źle klasyfikowany. Prawie wszystkie dotyczyły tej samej pomyłki: Layered_Fan_In vs Gather-Scatter. Klasyfikacja do najbliższego prototypu w surowych cechach rozkłada się na wkłady cech bez przybliżeń. Analityk widzi więc nie tylko, który przypadek oznaczyć, ale też które cechy przeciągnęły go w złą stronę.

**Uczona metryka nie bije losowej projekcji, a bramka to wychwyciła.** Kampania 5-foldowa: wszystkie 5 modeli odrzucone. Uczenie metryki (meta-learning ProtoNet) dodaje około 5 pp na dwóch foldach, odejmuje 6–11 pp na dwóch, na jednym nic nie zmienia. Średni lift wobec nietrenowanego enkodera to −0,012, CI [−0,100; +0,076]. Wysoka trafność bezwzględna (0,66–0,91 w zadaniu 4-way, przy szansie 0,25) bierze się z cech przypadku, nie z uczenia. Pipeline to pokazał, bo próg liczy wartość uczenia ponad podłogę, a nie samą trafność.

**Zakres twierdzeń.** To jest przypisanie typologii do już wykrytego przypadku (etap analityka, typologia w zgłoszeniu), a nie wykrywanie prania. Dane są syntetyczne (generator SAML-D), więc wyniki mierzą rozdzielność typologii tego generatora.

## 1. Dane i zadanie

SAML-D to syntetyczny zbiór transakcji: 9 504 852 transakcji z okresu 2022-10-07 → 2023-08-23, 12 kolumn, 28 typologii (normalne i podejrzane). Kolumna `Is_laundering` oznacza transakcje podejrzane, a `Laundering_type` ich typologię.

Ujęcie few-shot: pojawia się typologia, której model nie widział w treningu, i jest dla niej kilka oznaczonych przykładów. Pytanie brzmi, czy na tej podstawie nowe przypadki trafią do właściwej typologii. Foldy są rozłączne klasowo: typologie testowe nigdy nie występują w treningu.

## 2. Łańcuch

### 2.1 v0: okno konta (konto × tydzień)

Jednostką było okno jednego konta w jednym tygodniu: 5 210 998 okien, około 30 cech agregujących. Freeze objął 58 569 okien, 26 cech i 27 klas w 5 foldach. Test każdego foldu to 3 typologie podejrzane i 2 normalne.

Przed uczeniem była bramka zero: nearest-centroid 5-way 5-shot bez uczenia, ze standaryzacją liczoną tylko na klasach treningowych. Wynik to 0,556 ±0,014 na samych typologiach podejrzanych i 0,627 ±0,014 na wszystkich, przy szansie 0,20 i kryterium śmierci 0,35. Agregaty niosą więc informację o typologii.

Podłoga z losowego, nietrenowanego MLP wyszła 0,619 na walidacji, czyli tyle co nearest-centroid. Stąd zasada stosowana dalej: próg bramki musi mierzyć, ile dodaje uczenie metryki ponad samą reprezentację, a nie samą trafność.

Pierwszy przebieg z bramką (fold 0) dał 76,80% na teście i lowerCI 0,7629 przy progu 0,80. Model odrzucono, a raport odrzucenia i wpis w Experiments powstały tak jak przy każdym innym zakończeniu.

Explain v0 wskazał, gdzie leży problem. Słabe typologie to pary różniące się topologią przepływu między kontami: Layered_Fan_In vs Fan_In, Stacked Bipartite vs Bipartite, Cycle oraz Behavioural_Change_1 vs 2. Najczęstsze pomyłki to pary Scatter-Gather/Layered_Fan_Out i Deposit-Send/Gather-Scatter. Okno jednego konta nie widzi kształtu sieci, w której to konto uczestniczy.

### 2.2 v1: cechy sieciowe i historyczne konta (sfalsyfikowane)

Hipoteza: do okna konta brakuje informacji o kontrahentach i historii. v1 dodał cechy sieciowe i historyczne liczone tylko wstecz, bez etykiet. Ocena odbyła się na pulach walidacyjnych, w parach z v0 na tych samych epizodach.

Kryterium miało cztery gałęzie ustalone przed uruchomieniem:
- wzrost średniego recall sześciu słabych klas o ≥ 0,10 → freeze v1;
- wzrost tylko w sondach (nearest-centroid, RF) → poprawka w modelu;
- wzrost częściowy 0,05–0,10 → uzasadnienie dla GNN;
- brak wzrostu → sygnatura nie należy do pojedynczego konta, przejście na klasyfikację przypadków.

Wynik trafił w ostatnią gałąź (liczby w `samld_v1_proto.ipynb`). To był negatywny wynik, który wskazał następny krok: nie dokładać cech do konta, tylko zmienić jednostkę.

### 2.3 Przypadek jako jednostka (PASS)

**Definicja.** Przypadek to spójna składowa transakcji z `Is_laundering = 1`. Dwie transakcje należą do jednego przypadku, jeśli dzielą konto i dzieli je ≤ 7 dni, łańcuchowo. `Laundering_type` nie wyznacza granic, jest tylko etykietą (typ dominujący, wymagana czystość > 0,5).

**Grupy kont.** Przypadki, które w całym okresie dzielą jakiekolwiek konto, tworzą grupę `grp`. Support i query losowane są z rozłącznych grup, żeby ten sam pracz nie był po obu stronach epizodu.

**Cechy (31, bez etykiet).** Siedem grup:
- rozmiar i czas: liczba transakcji i kont, czas trwania, dni aktywne, transakcje na dzień, średni odstęp;
- role kont: udział źródeł, ujść i pośredników, maksymalny udział wyjść i wejść jednego konta, zmienność stopni;
- struktura: powtarzalność krawędzi, gęstość, głębokość warstw, cykl, wzajemność, dwudzielność;
- przepływ: przepływ przez pośredników, przekazanie dalej w ciągu doby, koncentracja ujścia i źródła;
- kwoty: suma, średnia, zmienność, max/mediana, udział okrągłych kwot i kwot tuż pod 10 000;
- instrumenty: gotówka, przelewy transgraniczne, zmiana waluty.

**Uczciwe ujęcie.** Granicę przypadku wyznacza flaga podejrzanej transakcji. To jest więc przypisanie typologii do wykrytego przypadku, a nie wykrywanie. Ujęcie jest zapisane w `schema.json` freeze'u.

**Kryterium główne: pary.** Liczono trafność 2-way 5-shot na trzech parach, które v0 myli: Bipartite/Stacked Bipartite, Fan_In/Layered_Fan_In i Cycle/Layered_Fan_In. ProtoNet był trenowany bez obu klas pary. PASS wymagał przyrostu przypadki − okna ≥ 0,10.

Pierwotnie kryterium było recallem w pulach 5-klasowych. Suchy bieg na danych syntetycznych pokazał, że partnerzy mylonej pary mogą trafić do różnych pul i wtedy test nie mierzy ich rozdzielania. Kryterium zmieniono, zanim cokolwiek uruchomiono na SAML-D. Zmiana jest opisana w notebooku.

**Wynik:**
- pary strukturalne, ProtoNet: 0,640 → 0,830, czyli +0,191 → PASS;
- na tych samych parach nearest-centroid +0,223 i RF +0,242;
- pule 5-way (nagłówek porównywalny z v0): ProtoNet 0,585 → 0,822, RF 0,609 → 0,889, nearest-centroid 0,555 → 0,844.

Już tutaj było widać to, co potwierdziła kampania: proste sondy na surowych cechach są co najmniej tak dobre jak ProtoNet.

**Kontrola mechanizmu (para BC1/BC2).** Topologia nie powinna rozdzielać tej pary, bo obie typologie to zmiana zachowania jednego konta. ProtoNet zyskał +0,133, pod flagą 0,15. RF zyskał jednak +0,34. Część zysku z przejścia na przypadki pochodzi więc z odcięcia normalnej aktywności konta przez granicę flagi, a nie tylko z topologii. Ten efekt rozcieńczenia jest realny i zostaje w raporcie.

**Kontrola podpisu generatora.** Same cechy rozmiaru (liczba transakcji, kont, czas trwania) dają na parach 0,644, a pełne cechy w tych samych sondach 0,856. Różnica jest dużo większa od progu 0,03, więc wynik nie sprowadza się do tego, że generator robi typologie różnej wielkości.

### 2.4 Freeze `samld-case`

Ze wszystkich transakcji 9873 jest podejrzanych i tworzą one 1902 przypadki. Największy przypadek ma 30 transakcji, czyli 0,3% podejrzanych. Warunek STOP (> 10%, czyli łączenie skleja osobne przypadki) nie zadziałał. Przetrwało wszystkie 17 typologii podejrzanych. SHA freeze'u to `9e47d115aeed…`.

Foldy to bloki po 4 typologie (zadanie 4-way). 17 nie dzieli się przez 4, więc pozostała typologia dostała dodatkowy fold, dopełniony pierwszymi typologiami z tej samej seedowanej permutacji. Bez tego Stacked Bipartite nigdy nie byłaby testowana. Wyszło to przy weryfikacji, przed treningiem, i freeze powtórzono.

Typologie testowe:
- fold 0: Bipartite, Cash_Withdrawal, Layered_Fan_Out, Over-Invoicing;
- fold 1: Behavioural_Change_1, Cycle, Deposit-Send, Fan_Out;
- fold 2: Fan_In, Scatter-Gather, Single_large, Structuring;
- fold 3: Behavioural_Change_2, Gather-Scatter, Layered_Fan_In, Smurfing;
- fold 4: Stacked Bipartite, dopełniona przez Bipartite, Cash_Withdrawal i Layered_Fan_Out.

**Podłogi i progi.** Podłoga to ProtoNet z losowym, nietrenowanym MLP: 3 ziarna × 400 epizodów testowych. Reguła progu została ustalona przed policzeniem podłóg: podłoga + 0,10 albo podłoga + połowa jej błędów, co mniejsze. Druga część chroni przed progiem ≥ 1,0 przy wysokiej podłodze. Podłogi wyszły od 0,767 (fold 4) do 0,886 (fold 2), a progi od 0,867 do 0,943. Bramka sprawdza dolną granicę 95% CI.

Przed kampanią poprawiono błąd: `architecture.json` nie zapisywał foldu dla `samld-case`. Heavy eval i explain oceniałyby wtedy każdy model na foldzie 0.

### 2.5 Kampania: uczona metryka vs losowa projekcja

Pięć przebiegów, po jednym na fold, bez promocji (`enable_promotion=False`). Wyniki: trafność, dolna granica CI, lift wobec podłogi.
- fold 0: 0,8814, lowerCI 0,8774, lift +0,001 (podłoga 0,8804, próg 0,9402);
- fold 1: 0,8795, lowerCI 0,8752, lift +0,053 (podłoga 0,8262, próg 0,9131);
- fold 2: 0,8301, lowerCI 0,8249, lift −0,056 (podłoga 0,8856, próg 0,9428);
- fold 3: 0,9132, lowerCI 0,9095, lift +0,052 (podłoga 0,8611, próg 0,9305);
- fold 4: 0,6568, lowerCI 0,6502, lift −0,110 (podłoga 0,7670, próg 0,8670).

Wszystkie 5 odrzucone. Średni lift −0,012, t-CI [−0,100; +0,076].

Kampania w jednym przebiegu pipeline'u, z treningiem bez cache (16.10), odtworzyła te liczby: lift per fold +0,001 / +0,053 / −0,055 / +0,052 / −0,110, średnia i przedział bez zmian, 0/5. Jedyna różnica to 0,001 na foldzie 2 przy ponownym treningu.

**Jak to czytać.** Na 31 ręcznie zaprojektowanych, wystandaryzowanych cechach losowa projekcja zachowuje geometrię klas prawie w całości. Metryka uczona na 9 typologiach treningowych (4 kolejne to walidacja) przenosi się na nowe typologie nierówno: na foldach 1 i 3 pomaga o około 5 pp, na foldach 2 i 4 szkodzi. Fold 4 wypada najgorzej. Jest to jedyny fold, w którym w teście są razem Bipartite i Stacked Bipartite. Hipoteza, niesprawdzona: trening na innych typologiach osłabia kierunki, które rozdzielają tę parę (głębokość warstw, role pośredników).

**Czego to nie znaczy.** Nie znaczy, że meta-learning nie działa na danych tabelarycznych. Znaczy, że przy tej reprezentacji i tej liczbie typologii treningowych uczenie metryki nie dodaje mierzalnej wartości ponad cechy. Każdy fold miał jedno ziarno treningu, więc na poziomie pojedynczego foldu różnica ±5 pp mieści się w zmienności ziaren. Wniosek dotyczy średniej.

**Co z tego wynika dla systemu.** Pętla z analitykiem (2.6) liczy prototypy na surowych cechach. Embedding ProtoNetu jest raportowany obok jako porównanie.

### 2.6 Active learning (PASS)

**Pytanie.** Czy model, który wybiera następny przypadek do oznaczenia, dochodzi do tej samej trafności z mniejszą liczbą etykiet niż wybór losowy?

**Protokół** (per fold, tylko typologie testowe, niewidziane w treningu):
- przypadki dzielone po grupach kont na pulę (60%, model może prosić o etykiety) i zbiór oceny (40%, nigdy nie oznaczany);
- start od 1 oznaczonego przypadku na typologię;
- 20 rund po 1 etykiecie. Nowa etykieta dołącza do prototypu swojej typologii, bez trenowania;
- 30 powtórzeń na fold, 5 foldów;
- miarą jest trafność zbalansowana na zbiorze oceny.

**Strategie.** Kryterium dotyczyło tylko `margin`. Pozostałe strategie są raportowane:
- `random`: punkt odniesienia;
- `margin`: najmniejsza różnica odległości do dwóch najbliższych prototypów;
- `typical`: w stylu TypiClust (Hacohen i in., ICML 2022);
- `hybrid`: 4 zapytania `typical` na rozgrzewkę, potem `margin`, jak u Lorenza i in. (2020) w AML;
- `kcenter`: najdalszy od oznaczonych.

**Kryterium (ustalone przed uruchomieniem, reprezentacja surowa).** Δ = trafność(`margin`) − trafność(`random`), uśredniona po budżetach 8–16 etykiet i liczona per fold. PASS wymagał średniej ≥ +0,03 i dolnej granicy 95% CI (t, n = 5) > 0.

Literatura sugerowała, żeby kryterium przenieść na `hybrid`. Suchy bieg na danych syntetycznych tego nie poparł: `typical` przegrał tam z losowym, bo skupiał etykiety na typologiach z wieloma skupiskami. Kryterium zostało przy `margin`. Na prawdziwych danych `typical` i `hybrid` wypadły dobrze. To lekcja o suchych biegach: dane syntetyczne testują kod, nie strategie.

**Wynik, reprezentacja surowa:**
- `margin` vs `random`: +0,042, CI [+0,022; +0,061]; per fold +0,022, +0,054, +0,044, +0,059, +0,031 → PASS;
- `hybrid` +0,052, CI [+0,026; +0,079]; `typical` +0,041; `kcenter` −0,014;
- `margin` i `hybrid` osiągają trafność `random`@24 (0,834) przy 11 etykietach;
- przy 24 etykietach: `hybrid` 0,890, `margin` 0,882, `random` 0,834.

Na embeddingu ProtoNetu `margin` daje +0,034, a `hybrid` +0,035. Działa, ale słabiej niż na surowych cechach, co zgadza się z kampanią.

**Mechanizm.** Przy jednym przykładzie na typologię prototyp jest bardzo zaszumionym estymatorem środka klasy. Etykieta przypadku leżącego daleko od granic prawie nic nie zmienia. Etykieta przypadku na granicy dwóch prototypów przesuwa prototyp dokładnie tam, gdzie powstają pomyłki. `kcenter` wybiera przypadki odstające i dlatego przegrywa z losowym.

**Zastrzeżenia.** Wyrocznią są etykiety generatora: analityk jest wolniejszy, czasem się myli, a koszt oznaczenia rośnie z rozmiarem przypadku. `hybrid` jest najlepszy liczbowo, ale nie był kryterium, więc nie jest zwalidowany. Symulacja jest sekwencyjna (1 etykieta na rundę). Wersję wsadową, w której analityk oznacza 8 przypadków naraz, opisuje 2.9.

### 2.7 Active learning w pipelinie

Komponent `al_container` (`scripts/al_simulation_entry.py`) uruchamia ten sam protokół na modelu z bieżącego przebiegu i typologiach testowych jego foldu. Działa w obu gałęziach bramki: przy modelu przyjętym (z numerem wersji) i odrzuconym (`not-registered`). Prototypy na surowych cechach nie potrzebują, żeby uczona metryka przeszła bramkę. Włącza się parametrem `enable_al`. Nie wnosi nowych zależności (numpy i torch), kluczy ani zewnętrznych usług.

Raport HTML zawiera:
- krzywe uczenia dla obu reprezentacji;
- tabelę strategii z liczbą etykiet potrzebnych do poziomu `random`@24;
- kolejkę następnych przypadków dla analityka.

Każdy przypadek w kolejce ma dokładne wyjaśnienie. Dla prototypów A (bieżąca odpowiedź) i B (druga najbliższa) w surowych cechach

    d(x, B)² − d(x, A)² = Σ_j [(x_j − B_j)² − (x_j − A_j)²]

więc wkład każdej cechy w „czemu A, a nie B” jest jawnym składnikiem sumy. Nie trzeba metody atrybucji.

Przebieg demonstracyjny (fold 3, model odrzucony przez bramkę):
- `margin` vs `random`: +0,059. `margin` osiąga poziom `random`@24 przy 10 etykietach. Przy 24 etykietach 0,925 vs 0,831;
- na embeddingu ProtoNetu `margin` potrzebuje do tego samego poziomu 17 etykiet;
- pierwszy w kolejce jest Case #1156 (14 kont, 18 transakcji), rozdarty między Gather-Scatter a Layered_Fan_In (margines 0,091). Bieżąca odpowiedź jest błędna, prawdziwa typologia to Layered_Fan_In;
- przypadki #285, #822 i #431 dotyczą tej samej pary, a #1019 to Smurfing vs Layered_Fan_In.

Wszystkie przypadki w kolejce były w tym momencie źle klasyfikowane. Margines wskazuje więc dokładnie pomyłki modelu.

To jest jeden przebieg na jednym foldzie: ilustracja działania komponentu. Dowodem jest pięciofoldowy wynik z 2.6.

### 2.8 Explainability

Explainability składa się z dwóch warstw o różnym statusie.

**Raport modelu (`explain_container`, w obu gałęziach bramki).** ProtoNet decyduje po odległości, więc wyjaśniany jest margines decyzji, czyli dokładnie różnica logitów:

    m(x) = ‖f(x) − p_kontrast‖ − ‖f(x) − p_wybrana‖

Metoda to kontrastowe Integrated Gradients. Kontrastem jest druga najbliższa klasa przy trafnej decyzji, a prawdziwa klasa przy pomyłce („czemu B, a nie A”). Punktem odniesienia jest średni przykład supportu klasy kontrastowej. Raport zawiera:
- grupy cech;
- studia przypadków w jednostkach surowych (kwoty, dni, liczby kont) z identyfikatorem i rozmiarem przypadku;
- macierz pomyłek per typologia;
- cechy, po których każda typologia jest rozpoznawana.

Celowo nie ma kontrfaktyków w stronę klasy normalnej: w AML byłby to przepis na ominięcie detekcji, a nie wyjaśnienie.

Jakość wyjaśnień sprawdzają testy z progami ustalonymi z góry:
- kompletność: p95 względnej reszty ≤ 1%;
- wierność przez usuwanie: top-5 cech vs losowe 5, stosunek ≥ 2×;
- stabilność przy losowaniu supportu: Spearman ≥ 0,7, 20 przelosowań;
- wrażliwość na punkt odniesienia: tylko raportowana.

Na modelach v0 kompletność nie przeszła (p95 1,3–2,3% przy progu 1%). Na danych syntetycznych pokazano, że to błąd dyskretyzacji całki: zanika jak 1/liczba kroków, bo ReLU ma załamania. Parametr `--ig-steps` jest gotowy, a podniesienie domyślnych 50 kroków do 200 czeka na potwierdzenie na prawdziwym artefakcie. Stabilność też nie przeszła. Zostaje to jako wynik: pojedyncze rankingi cech zależą od wylosowanego supportu, więc czyta się je na poziomie grup cech, nie dokładnej kolejności. Progów nie przesuwano. Werdykty testów dla modeli `samld-case` są w węzłach explain poszczególnych przebiegów kampanii.

**Kolejka active learning.** Tu wyjaśnienie jest dokładne z konstrukcji (2.7), bez testów jakości. Skoro uczona metryka nie bije surowych cech, to dla analityka ważniejsza jest ta warstwa: tłumaczy tę samą reprezentację, na której liczone są prototypy w pętli.

### 2.9 Pętla z analitykiem

**Architektura.** Są dwa pipeline'y i łącząca je funkcja:
- pipeline treningowy (lub kampania) kończy się krokiem active learning, który zapisuje partię do `analyst/do_oznaczenia/<przebieg>/fold<k>/`: `kolejka.csv` z pustą kolumną `typologia_analityka`, `transakcje.csv`, raport i `stan.json` (oznaczone przypadki, pula, zbiór kontrolny, kolejka);
- analityk odkłada wypełniony `kolejka.csv` do `analyst/oznaczone/` pod tą samą ścieżką;
- funkcja Cloud Run (Eventarc na zdarzeniu „nowy obiekt w buckecie”) reaguje tylko na ten plik i uruchamia pipeline rundy. Identyfikator przebiegu powstaje z partii i wersji pliku, więc podwójne zdarzenie nie uruchamia dwóch rund, a poprawiony plik uruchamia nową;
- pipeline rundy (`scripts/al_round_entry.py`, jeden krok) dokłada odpowiedzi do wzorców, sprawdza bramkę, zapisuje wzorce w użyciu (`analyst/wzorce/.../produkcja.json`) i następną partię `...-r1`, `-r2` itd., a analityk dostaje maila.

Nigdzie nie ma kluczy: funkcja i pipeline działają na kontach serwisowych projektu, maile wysyła reguła alertu z logów.

**Partia zamiast jednej etykiety.** Analityk oznacza 8 przypadków naraz, więc kolejka nie może być po prostu ośmioma najmniejszymi marginesami. Takie przypadki są zwykle prawie identyczne, leżą na jednej granicy i ciągną jeden prototyp. Kolejka jest wybierana po jednym przypadku: po każdym wyborze podpowiedź modelu tymczasowo zastępuje etykietę, prototyp się przesuwa i liczy się następny margines. Na zbiorze syntetycznym (5 foldów × 30 podziałów) jedna partia 8 przypadków podnosiła trafność zrównoważoną średnio o 4,6 pp. Osiem najmniejszych marginesów dawało 3,1 pp, 8 losowych przypadków 3,2 pp, a etykiety dodawane pojedynczo 5,6 pp. To pomiar na danych deweloperskich, nie na SAML-D.

**Bramka rundy.** Nowe wzorce wchodzą do użycia tylko wtedy, gdy spełniają dwie reguły:
- trafność zrównoważona na zbiorze kontrolnym nie spada więcej niż o 2 pp;
- żadna typologia nie traci więcej niż 10 pp, licząc co najmniej 3 przypadki (jedna zmiana w małej klasie to szum).

Przy odrzuceniu w użyciu zostają poprzednie wzorce, a odpowiedzi z rundy czekają w `stan.json` na decyzję człowieka. Bramka ocenia wzorce, nie analityka: nie odróżnia złej odpowiedzi od poprawnej, która przesunęła średnią w złą stronę.

Druga reguła powstała po rundzie 2 na SAML-D (niżej). Na zbiorze syntetycznym bramka tylko ze średnią wpuszczała do użycia załamanie jednej typologii w około 7% rund. Po dodaniu drugiej reguły takich rund nie było. Kosztem jest więcej odrzuceń (około 3 na 10 rund zamiast 2 na 10) i około 1 pp niższa trafność zrównoważona po trzech rundach. W AML utrata rozpoznawania jednej typologii jest gorsza niż wolniejsza nauka. Sprawdzono też zamianę średniej na trzech najbliższych oznaczonych sąsiadów: załamań nie było mniej, więc klasyfikator został bez zmian.

**Wynik na SAML-D, fold 3** (typologie Behavioural_Change_2, Gather-Scatter, Layered_Fan_In i Smurfing; zbiór kontrolny 124 przypadki; odpowiedzi analityka symulowane prawdziwymi typologiami):
- start: 8 etykiet, 80,1%;
- runda 1: przyjęta, 86,8%. Prawie cały zysk to Layered_Fan_In (z 22 do 33 trafnych na 41). 7 z 8 przypadków w kolejce to Layered_Fan_In, a analityk poprawił podpowiedź modelu w 2 z nich;
- runda 2: odrzucona. Kolejka siedziała na granicy Layered_Fan_In / Gather-Scatter. Gather-Scatter wzrósł z 67% do 92%, a Layered_Fan_In spadł z 80,5% do 48,8% (13 przypadków z 41). Trafność zrównoważona spadła tylko o 1,7 pp, a zwykła o 8 pp. Wszystkie 8 odpowiedzi było poprawnych;
- runda 3: przyjęta, 91,0% przy 24 etykietach w użyciu. Layered_Fan_In doszedł do 92,7%, a Behavioural_Change_2 stracił 1 przypadek z 26;
- runda 4, pierwsza uruchomiona przez trigger: przyjęta, 92,8% przy 32 etykietach w użyciu. Analityk poprawił podpowiedź modelu już tylko w 1 z 8 przypadków. Następna partia nadal siedzi na granicach Layered_Fan_In (z Smurfing, Gather-Scatter i Behavioural_Change_2).

Zysk z przyjętych rund maleje: +6,7, +4,2, +1,8 pp, a model coraz częściej sam ma rację na przypadkach, które podsuwa. Na tym foldzie pętla zbliża się do nasycenia. Kryterium zatrzymania (np. dwie rundy z rzędu poniżej 1 pp) nie było ustalone z góry, więc to obserwacja, nie wynik.

Dla porównania: symulacja z 2.7 na tym samym foldzie, z etykietami dodawanymi pojedynczo, dawała 0,925 przy 24 etykietach, a wybór losowy 0,831. Tamto to średnia z 30 podziałów, a tu jeden podział, więc porównanie jest orientacyjne.

**Błąd znaleziony po drodze.** Folder zapisu modelu nazywał się `protonet-seed0-<sekunda Unix>`. Dwa foldy kampanii kończące trening w tej samej sekundzie zapisywały do tego samego folderu, a późniejszy nadpisywał wcześniejszy. Ponieważ trening brał się z cache, fold 0 wskazywał na model foldu 3. Wyszło to w mailu kampanii, w którym fold 0 miał kolejkę z typologii foldu 3. Wynik bramki i lift nie ucierpiały, bo są liczone w treningu przed zapisem. Ucierpiałoby wszystko, co wczytuje zapisany model, łącznie z rejestrem przy przyjętym modelu. Teraz nazwa folderu zawiera fold, czas i losowy sufiks. Kampania bez cache z 16.10 to potwierdziła: kolejka foldu 0 zawiera jego własne typologie (Layered_Fan_Out vs Bipartite), a fold 0 potrzebuje 12 etykiet do poziomu losowego wyboru przy 24, a nie 10 jak w błędnym przebiegu, w którym liczba pochodziła z foldu 3. Lift i werdykty bramki są takie same jak przed poprawką.

## 3. Czego te wyniki nie mówią

- **Dane są syntetyczne.** Typologie SAML-D pochodzą z reguł generatora. Kontrola rozmiaru wyklucza najprostszy podpis generatora, ale nie wszystkie.
- **To nie jest detekcja.** Granice przypadków wyznacza `Is_laundering`. W produkcji wyznaczałby je alert albo dochodzenie, z innym szumem.
- **Zysk z przejścia na przypadki jest częściowo efektem rozcieńczenia** (kontrola BC1/BC2), nie tylko topologii.
- **Uczona metryka nie dodaje wartości** w tej konfiguracji (jedno ziarno na fold, 31 cech, 9 typologii treningowych).
- **Active learning był symulowany z idealną wyrocznią**, także w rundach z analitykiem na SAML-D. `hybrid` jest najlepszy liczbowo, ale niezwalidowany.
- **Rundy z analitykiem to jeden fold i jeden podział.** Zbiór kontrolny jest zarazem kryterium bramki i miarą w raporcie, więc po odrzuconych rundach raportowana trafność jest lekko optymistyczna. W produkcji zbiorem kontrolnym byłyby przypadki wcześniej potwierdzone przez analityków, a nie etykiety generatora.

## 4. Co dalej

Otwarte są cztery rzeczy. Inne kierunki (inne enkodery, modele fundamentowe tabelaryczne, GNN) są świadomie odłożone.

1. **Ocena nowych przypadków.** Codzienny przebieg, który ocenia nowe wykryte przypadki wzorcami z `produkcja.json`. W produkcji to z niego, a nie ze zbioru testowego foldu, powinna pochodzić kolejka dla analityka.
2. **Decyzja o odłożonych odpowiedziach.** Odpowiedzi z rund odrzuconych przez bramkę czekają w `stan.json`; potrzebny jest sposób, żeby je potwierdzić albo odrzucić.
3. **`--ig-steps 200`.** Do włączenia po potwierdzeniu na jednym prawdziwym artefakcie, że kompletność spada poniżej 1%.
4. **Enkoder, który nie może być gorszy od surowych cech**, np. embedding [x, s·f(x)]. Ma sens tylko z kryterium śmierci ustalonym przed uruchomieniem: lift na walidacji ≥ +0,03 na co najmniej 4 z 5 foldów, inaczej kierunek zamknięty.

## 5. Reprodukcja

Kolejność:
1. `samld_v1_proto.ipynb`: falsyfikacja v1.
2. `samld_case_proto.ipynb`: test przypadków (pary, kontrole BC i rozmiaru).
3. `samld_case_freeze.ipynb`: freeze, weryfikacja ze STOP, podłogi i progi (`~/data_cache/samld_case_floors.json`).
4. Kampania z `rung9_hpo.ipynb` z załadowanym `configs/pipeline_config_samld.json` (komórka kampanii czyta progi z pliku podłóg).
5. `samld_case_active_learning.ipynb`: kryterium active learning na 5 foldach.
6. Przebieg pipeline'u z `enable_al=True` (albo kampania) dla raportu AL i pierwszej partii dla analityka.
7. `al_runda.ipynb`: komórka 4 kompiluje pipeline rundy i zapisuje go w GCS, `deploy_trigger.sh` w Cloud Shell wdraża trigger. Każda kolejna runda startuje od wrzucenia `kolejka.csv` do `oznaczone/`; na demo wypełnia go komórka 3.

Wszystkie losowania są seedowane. Każdy krok pipeline'u, który wczytuje dane, pobiera archiwum freeze'u, liczy jego SHA-256 i zatrzymuje się przy niezgodności z manifestem.
