# Ekonomatiki
Projekt FERI RIT-UN Umetna inteligenca pri avtonomni vožnji

# O projektu
Smo študenti Fakultete za elektrotehniko, računalništvo in informatiko, ki v sklopu izbirnega projekta Umetna inteligenca pri avtonomni vožnji sodelujemo na projektu Ekonomatiki.

Cilj projekta je izdelava programa, ki bo uporabniku s pomočjo GPS podatkov podal oceno o stopnji ekonomičnosti njegove vožnje

Delo na projektu poteka v sklopu večih predmetov 2. in 3. letnika RIT-UN programa, predmeti so SIS, SPO, UI in URG

# O skupini
Skupino Ekonomatiki sestavljamo 4 študenti RIT-UN program na FERI-ju:

### Člani skupine

Luka Pečar 2. letnik RIT-UN, šolski mail luka.pecar@student.um.si

Nejc Keglevič 2. letnik RIT-UN, šolski mail nejc.keglevic@student.um.si

Stella Pogačič 2. letnik RIT-UN, šolski mail stella.pogacic@student.um.si

Milica Popovič 2. letnik RIT-UN, šolski mail milica.popovic1@student.um.si

### Mentor skupine

Profesor Niko Lukač 

# Podrobneje o predmetih

### SIS - Signali in slike

Pri predmetu signali in slike je glavna naloga zajemanje in označevanje podatkov, katere bomo pri drugih predmetih uporabili za učenje nevronskih mrež, oblikovanje povratnih informacij, ter analizo delovanja programa

Pri našem projektu bo to bolj natančno zajemanje GPS podatkov, izvoz podatkov v .gpx obliko, označevanje stopnje ekonomičnosti na določenem časovnem intervalu, augmentacija zajetih podatkov za večji nabor testnih podatkov, ter avtomatizacija procesa označevanja.

### SPO - Sistemska programska oprema

Pri predmetu sistemska programska oprema je glavna naloga vzpostavitev infrastrukture za deljenje podatkov in programa znotraj skupine, ter med napravo za zajemanje podatkov in napravo za analizo teh podatkov

Pri našem projektu to pomeni vzpostavitev MQTT sistema za komunikacijo in prenos podatkov med člani skupine, real time prenos GPS podatkov iz naprave za zajemanje, na računalnik za analizo podatkov, ter nazaj na napravo za povratne informacije

### UI - Umetna inteligenca

Pri predmetu umetna inteligenca je glavna naloga izdelava nevronskih mrež, ki se bodo na podlagi zajetih označenih podatkov naučile analizirati ne označene podatke katere bo zajemal uporabnik

Pri našem projektu se bodo nevronske mreže na podlagi testnih označenih podatkov naučile kako določati stopnjo ekonomičnosti vožnje, ter omogočile analizo ne označenih podatkov, končni cilj pa je te nevronske mreže tako izdelati, da bodo omogočale čim bolj real time analizo vožnje

### URG - Uvod v računalniško geometrijo

Pri predmetu uvod v računalniško geometrijo je glavna naloga izdelava 3D modelov za razne naprave, kot so držala za naprave za zajemanje, držalo za napravo za povratne informacije in podobno

Pri našem projektu je to 3D model držala za telefon, ki je naša naprava za zajemanje GPS podatkov

#### Za vsak predmet je v github repositoriju poseben feature branch, v katerem se nahaja zbirka z programi, datotekami in dokumentacijo za vsak predmet