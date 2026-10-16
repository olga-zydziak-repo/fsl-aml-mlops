# Scenariusz demo: typologie AML, MLOps i analityk w pętli

Około 15 minut. Wszystko pokazujemy z przygotowanych wcześniej przebiegów, a nie na żywo: przebieg treningowy trwa kilkanaście minut, a runda około 5. Jeden element można puścić na żywo (krok 7), bo pierwsze efekty widać po minucie.

## Przygotowanie (dzień wcześniej)

- Kampania i 4 rundy na foldzie 3 zrobione (start 80,1%, runda 1 86,8%, runda 2 odrzucona, runda 3 91,0%, runda 4 92,8%, ta już z triggera).
- Otwarte karty:
  - Vertex AI → Pipelines → Runs: przebieg kampanii, jeden przebieg `fsl-al-round`;
  - raport AL foldu 3 (`do_oznaczenia/<kampania>/fold3/raport.html`) i raporty rund (`fold3-r1/` do `fold3-r4/`, w każdym `raport.html`);
  - folder `analyst/do_oznaczenia/<kampania>/fold3-r4/` w GCS;
  - skrzynka z mailami: zbiorczy z kampanii i cztery maile z rund.
- Linki do GCS otwieraj z `?authuser=<konto GCP>`, inaczej przeglądarka z kilkoma kontami pokaże 403.
- Na żywo (krok 7): partia `fold3-r4` gotowa, `al_runda.ipynb` otwarty po komórkach 1–2 (`BATCH` kończy się na `fold3-r4`).

## Przebieg

**1. Problem (bez ekranu, 30 s).** „Analityk AML ma wykryty przypadek, kilka kont i kilkanaście transakcji, i musi przypisać mu typologię. Typologii jest kilkanaście, a potwierdzonych przykładów niektórych garstka. To jest zadanie few-shot, i do tego zbudowałam cykl MLOps na Vertex AI.”

**2. Pipeline treningowy (graf przebiegu w Vertex).** „Zamrożone dane z sumą kontrolną, trening, bramka, a dalej rejestr albo raport odrzucenia, wyjaśnienia, active learning i mail. Każdy krok to kontener z tego samego obrazu, bez kluczy i bez zewnętrznych usług.”

**3. Bramka mówi prawdę (mail zbiorczy kampanii).** „Pięć foldów, pięć modeli odrzuconych. Bramka nie pyta o trafność, tylko o to, ile uczenie dodaje ponad losowy, nietrenowany enkoder. Wyszło −1,2 pp z przedziałem przechodzącym przez zero. Wysoka trafność bierze się z cech przypadku, nie z meta-learningu, i pipeline to pokazał, zamiast zarejestrować ładnie wyglądający model.”

**4. Active learning (raport AL foldu 3, wykres u góry).** „Zamiast losowych przypadków model podsuwa analitykowi te, przy których się waha. Do trafności, którą losowy wybór osiąga przy 24 etykietach, wystarcza 10. Kryterium ustaliliśmy przed uruchomieniem i przeszło na wszystkich pięciu foldach.”

**5. Wyjaśnienie (karta przypadku w tym samym raporcie).** „Przy każdym przypadku model mówi, między czym się waha i które cechy ciągną w każdą stronę. To nie jest przybliżenie: odległość do wzorca rozkłada się dokładnie na sumę wkładów cech.”

**6. Co dostaje analityk (folder w GCS, `kolejka.csv` w Excelu).** „Osiem wierszy, jeden przypadek na wiersz, podpowiedź modelu, druga typologia, uzasadnienie i pusta kolumna na odpowiedź. Obok surowe transakcje tych przypadków. Przykład: model podpowiadał Gather-Scatter, a z transakcji widać Layered_Fan_In, więc analityk to poprawia.”

**7. Pętla zamyka się sama (na żywo, opcjonalnie).** Komórka 3 w `al_runda.ipynb` odkłada wypełniony plik do `oznaczone/`. Po chwili w Vertex pojawia się przebieg `fsl-al-round`. „Nikt nie klika pipeline'u. Wrzucenie pliku uruchamia rundę przez funkcję Cloud Run, a za kilka minut przyjdzie mail z wynikiem i następną partią.” Wynik pokaż później albo od razu przejdź do gotowych raportów.

**8. Rundy (raporty `fold3-r1` → `r4`).**
- Runda 1: „8 odpowiedzi, trafność z 80,1% do 86,8%, bez trenowania. Odpowiedź analityka od razu zmienia wzorzec typologii.”
- Runda 2, tabela per typologia: „Średnio prawie bez zmian, ale Layered_Fan_In spadł z 80% do 49%. Bramka sprawdza każdą typologię osobno i takiej zamiany nie przepuszcza. Odpowiedzi były poprawne: bramka nie ocenia analityka, tylko to, czy wzorce po zmianie są lepsze. Odrzucone odpowiedzi czekają na decyzję człowieka.”
- Runda 3: „91,0% przy 24 etykietach.”
- Runda 4: „Ta poszła sama, z triggera. 92,8% przy 32 etykietach, a analityk poprawił model już tylko w 1 z 8 przypadków. Zysk z rundy maleje, bo model coraz częściej sam ma rację tam, gdzie się waha. Wykres pokazuje całą historię razem z odrzuconą rundą.”

**9. Zamknięcie.** „Typologię podpowiadają wzorce utrzymywane przez analityka, z bramką i historią każdej rundy. Następny krok to codzienna ocena nowych wykrytych przypadków tymi wzorcami. Wtedy kolejka dla analityka powstaje z prawdziwego strumienia.”

## Pytania, które pewnie padną

- **„Dane są syntetyczne?”** Tak, SAML-D to generator. Kontrola rozmiaru wyklucza najprostszy podpis generatora, ale wyniki mierzą rozdzielność typologii tego generatora, a nie banku.
- **„To wykrywa pranie?”** Nie. Przypisuje typologię już wykrytemu przypadkowi, czyli robi to, co analityk po alercie.
- **„A jeśli analityk się pomyli?”** Bramka łapie odpowiedzi, które psują wzorce, ale nie jest wykrywaczem błędów. Dlatego odrzucone odpowiedzi idą do drugiej weryfikacji, a nie do kosza.
- **„Czemu nie trenujecie modelu na nowo po każdej rundzie?”** Bo nie trzeba. Wzorzec typologii to średnia oznaczonych przypadków, więc nowa etykieta działa od razu. Trening jest osobnym, rzadkim pipeline'em.
- **„Skąd zbiór kontrolny w produkcji?”** Z przypadków wcześniej potwierdzonych przez analityków. Tu pochodzi z danych, bo dane są syntetyczne.
- **„Dlaczego model z rejestru nie jest w użyciu?”** Bo bramka pokazała, że uczona metryka nie bije losowej projekcji na tych cechach. W użyciu są wzorce na surowych cechach, a uczony enkoder jest raportowany obok.
