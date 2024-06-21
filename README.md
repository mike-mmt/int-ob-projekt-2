# Projekt 2 - kontrola agentami w minigrze typu MOBA (Derk's Gym)

### W tym projekcie zająłem się kontrolą agentami / kreaturami w minigrze typu MOBA. W grze tej mamy dwie drużyny, które walczą ze sobą. Każda drużyna składa się z trzech kreatur, z których każda ma swój zestaw przedmiotów (umiejętności) w ekwipunku. Celem gry jest uzyskanie jak największej liczby punktów poprzez pokonanie przeciwników lub zniszczenie statuy w ich bazie.

W moim przypadku, wszyscy agenci mają taki sam, prosty ekwipunek - "Talons" służace do zadawania obrażeń i "HealingGland" służący do leczenia innej jednostki.

Opiszę głównie sterowanie za pomocą sieci neuronowych, ponieważ ta metoda jest zdecydowanie najlepsza w porównaniu do innych, i najbardziej ciekawa.

Sterowanie agentami odbywa się poprzez prostą sieć neuronową, która przyjmuje jako wejście obserwacje z gry i zwraca jako wyjście akcje, które agent powinien wykonać. Każdy agent ma własną,oddzielną sieć neuronową.

Sieć neuronowa uczy się na każdym kroku (timestep) na podstawie nagród, które agent otrzymuje za swój stan lub wykonaną akcję. Nagrody są zdefiniowane w postaci funkcji nagród, która określa jakie punkty otrzymuje agent za poszczególne akcje lub stan. W projekcie zaimplementowałem kilka różnych funkcji nagród, które różnią się między sobą wagami przyznawanych punktów za różne akcje.

Ponieważ środowisko jest przystosowane do uruchamiania wielu instancji gry jednocześnie, uczenie sieci neuronowej odbywa się równolegle na wielu agentach (ilość gier \* 6).

Na koniec każdego epizodu (gry lub końca wielu gier na raz), wybierana jest najlepsza sieć neuronowa poprzez największą sumę punktów zdobytych w epizodzie przez agenta, następnie jest ona kopiowana na wszystkie agenty oraz lekko mutowana, aby zachować różnorodność (Jest ona również zapisywana na dysku).

W projekcie zaimplementowałem również kilka różnych strategii sterowania agentami, które różnią się między sobą sposobem uczenia sieci neuronowej, funkcją nagród, strategią gry itp.

### Porównanie algorytmów sterowania

|                             | NN        | Q Learning    |     |     |
| --------------------------- | --------- | ------------- | --- | --- |
| Ilość epizodów x ilość aren | 350 x 256 | 100 x 4       |     |     |
| Prędkość trenowania         | b. szybka | b. wolna      |     |     |
| Działanie algorytmu         | b. dobre  | słabe/średnie |     |     |

Drugim algorytmem, jaki wypróbowałem był Q Learning. Musiałem w pewien sposób ograniczać i dyskretyzować przestrzeń stanów, aby ograniczyć ilość możliwych stanów. Mimo tego, algorytm ten działał bardzo wolno, a wyniki były słabe w porównaniu do sterowania za pomocą sieci neuronowych. Przez większość czasu agenci biegali w kółko, nie wykonując żadnych akcji, co skutkowało zerowymi lub bliskimi zeru wynikami.

### Typy przyjętych algorytmów

Próbowałem uczyć sieci na różnych funkcjach nagród i strategii, było ich bardzo wiele ale jestem w stanie podzielić na najważniejsze typy:

1. Prosta funkcja nagrody za zadawanie obrażeń.
2. Prosta funkcja nagrody za zadawanie obrażeń i leczenie sojuszników
   3 Bardziej rozbudowana funkcja nagrody, przyjmująca wiele parametrów m.in spędzanie czasu na danych terytoriach, zadawanie obrażeń, leczenie sojuszników, niszczenie statuy przeciwnika itp.
3. Strategia gry typu atakujący/obrońcy, gdzie jedna drużyna próbuje zniszczyć wrogą statuę, a druga broni swojej.

Próbowałem również w inny sposób wybierać i zapisywać sieci, np.
zapisywać 6 sieci z dwóch najlepszych drużyn, albo wybierać. Nie przyniosło to zbytnio lepszych rezultatów.

### Najważniejsze pliki:

- moba/nn.py - oryginalna implementacja
- moba/nn2.py - druga, najlepsza implementacja
- moba/nn-2-divide-networks.py - implementacja z wybieraniem wielu sieci
- moba/nn_a_d.py - implementacja ze strategią atakujący/obrońcy
- moba/qtable.py - implementacja z Q Learningiem

## Wnioski

Zaskakująco, najlepsze wyniki osiągałem na najprostszych funkcjach.
Dodatkowo, długość treningu niekoniecznie dobrze wpływała na jakość wyników. Możliwe też, że właśnie potrzeba było o wiele więcej epizodów, aby osiągnąć lepsze wyniki.

## Dodatek

W pliku fuzzy-controller/one.js znajduje się implementacja sterowania agentami w grze "czołgi" p. Madejskiego, która świetnie sobie radzi z przeciwnikami.
