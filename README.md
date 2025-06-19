# Ekonomatiki
Projekt FERI RIT-UN: Umetna inteligenca pri avtonomni vožnji

## Kazalo

- [O projektu](#o-projektu)
- [O skupini](#o-skupini)
- [Sistemske zahteve](#sistemske-zahteve)
- [Namestitev](#namestitev)
- [Struktura projekta](#struktura-projekta)
- [SIS - Signali in slike](#sis---signali-in-slike)
  - [Obdelava.py - GPS Tool](#obdelavaby---gps-tool)
  - [Struktura map](#struktura-map-sis)
- [UI - Umetna inteligenca](#ui---umetna-inteligenca)
  - [zdruzevanje.py - AI Analiza ekonomičnosti](#zdruzevanjenepy---ai-analiza-ekonomičnosti)
  - [Modeli in funkcionalnosti](#modeli-in-funkcionalnosti)
- [SPO - Sistemska programska oprema](#spo---sistemska-programska-oprema)
- [URG - Uvod v računalniško geometrijo](#urg---uvod-v-računalniško-geometrijo)
- [Uporaba](#uporaba)
- [Tehnične podrobnosti](#tehnične-podrobnosti)

## O projektu

Smo študenti Fakultete za elektrotehniko, računalništvo in informatiko, ki v sklopu izbirnega projekta Umetna inteligenca pri avtonomni vožnji sodelujemo na projektu Ekonomatiki.

**Cilj projekta**: Izdelava programa, ki bo uporabniku s pomočjo GPS podatkov podal oceno o stopnji ekonomičnosti njegove vožnje.

Delo na projektu poteka v sklopu večih predmetov 2. in 3. letnika RIT-UN programa:
- SIS - Signali in slike
- SPO - Sistemska programska oprema
- UI - Umetna inteligenca
- URG - Uvod v računalniško geometrijo

## O skupini

Skupino Ekonomatiki sestavljamo 4 študenti RIT-UN programa na FERI-ju:

### Člani skupine

- **Luka Pečar** - 2. letnik RIT-UN, šolski mail: luka.pecar@student.um.si
- **Nejc Keglevič** - 2. letnik RIT-UN, šolski mail: nejc.keglevic@student.um.si
- **Stella Pogačič** - 2. letnik RIT-UN, šolski mail: stella.pogacic@student.um.si
- **Milica Popovič** - 2. letnik RIT-UN, šolski mail: milica.popovic1@student.um.si

### Mentor skupine
- **Profesor Niko Lukač**

## Sistemske zahteve

### Python verzija
- Python 3.8 ali novejši

### Potrebne knjižnice

```bash
# Osnovne knjižnice
pip install numpy matplotlib pillow

# GUI in vizualizacija
pip install customtkinter
pip install tkinter  # Običajno že vključen v Python

# GPS in zemljevidi
pip install gpxpy
pip install geopy
pip install folium

# Machine Learning
pip install torch torchvision
pip install scikit-learn

# Obdelava podatkov
pip install pandas
pip install csv  # Vključen v Python

# Dodatne knjižnice
pip install webbrowser  # Vključen v Python
pip install os  # Vključen v Python
pip install random  # Vključen v Python
pip install math  # Vključen v Python
pip install json  # Vključen v Python
pip install subprocess  # Vključen v Python
pip install sys  # Vključen v Python
pip install gc  # Vključen v Python
```

### Enostavna namestitev vseh knjižnic
```bash
pip install customtkinter gpxpy geopy folium torch torchvision scikit-learn pandas numpy matplotlib pillow
```

## Namestitev

### Hitra namestitev

1. **Prenesite projekt**
   ```bash
   git clone [URL_REPOZITORIJA]
   cd Ekonomatiki-Projekt
   ```

2. **Namestite Python** (če ga še nimate)
   - Potrebujete Python 3.8 ali novejši
   - Prenesite z [python.org](https://www.python.org/downloads/)

3. **Namestite potrebne knjižnice**
   
   Enostavna namestitev vseh knjižnic naenkrat:
   ```bash
   pip install customtkinter gpxpy geopy folium torch torchvision scikit-learn pandas numpy matplotlib pillow
   ```

   Ali če želite virtualno okolje (priporočeno):
   ```bash
   # Ustvarite virtualno okolje
   python -m venv venv
   
   # Aktivirajte ga
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   
   # Namestite knjižnice
   pip install customtkinter gpxpy geopy folium torch torchvision scikit-learn pandas numpy matplotlib pillow
   ```

4. **Preverite namestitev**
   ```bash
   # Preverite če dela Obdelava.py
   cd SIS/Obdelava
   python Obdelava.py
   
   # Preverite če dela zdruzevanje.py
   cd ../../UI
   python zdruzevanje.py
   ```

### Struktura projekta po namestitvi

Ko uspešno namestite projekt, boste imeli naslednjo strukturo:

```
Ekonomatiki-Projekt/
│
├── SIS/                      # Predmet Signali in Slike
│   ├── Obdelava/            
│   │   ├── Obdelava.py      # Program za GPS analizo
│   │   ├── Assets/          # Ikone za program
│   │   ├── Maps/            # Tu se shranijo zemljevidi
│   │   └── Analysis/        # Tu se shranijo analize
│   │
│   └── Zgled/               # Primeri in dokumentacija
│
├── UI/                       # Predmet Umetna Inteligenca
│   ├── zdruzevanje.py       # Program za AI analizo
│   └── modeli/              # AI modeli (.pt datoteke)
│
├── SPO/                      # Predmet Sistemska Programska Oprema
├── URG/                      # Predmet Uvod v Računalniško Geometrijo
└── README.md                 # Ta datoteka
```

### Težave pri namestitvi?

Če naletite na težave:

1. **"pip ni prepoznan"**: Ponovno namestite Python in obkljukajte "Add Python to PATH"
2. **"No module named..."**: Ponovno zaženite `pip install [ime_modula]`
3. **Težave s torch**: Za PyTorch obiščite [pytorch.org](https://pytorch.org/) in uporabite njihov selector
4. **Program se ne zažene**: Preverite, da ste v pravi mapi in da uporabljate Python 3.8+

## Struktura projekta

```
Ekonomatiki-Projekt/
│
├── SIS/
│   ├── Obdelava/
│   │   ├── Assets/
│   │   │   ├── light_mode_icon.png
│   │   │   └── dark_mode_icon.png
│   │   ├── Maps/              # Generirani zemljevidi
│   │   ├── Analysis/          # CSV analize
│   │   └── Obdelava.py        # Glavni program za obdelavo GPS
│   │
│   └── Zgled/
│       ├── map.png
│       ├── primer.png
│       └── program.png
│
├── UI/
│   ├── modeli/                # Shranjeni PyTorch modeli
│   │   ├── BiLSTMModelHitrosti.py
│   │   ├── BiLSTMPospesevanjeModel.py
│   │   ├── BiLstmModelHitrosti.py
│   │   ├── modelHitrosti3.py
│   │   └── ...
│   │
│   ├── integriranoPrediprocirane.py
│   ├── izrisZemljevidaPOStopnjah.py
│   ├── pospesevanjeModel.py
│   ├── predprocesiranje.py
│   ├── zdruzevanje.py         # Glavni program za AI analizo
│   └── UI.py
│
├── SPO/                       # MQTT komunikacija
├── URG/                       # 3D modeli
└── README.md

```

## SIS - Signali in slike

Pri predmetu signali in slike je glavna naloga zajemanje in označevanje podatkov, katere bomo pri drugih predmetih uporabili za učenje nevronskih mrež, oblikovanje povratnih informacij, ter analizo delovanja programa.

### Obdelava.py - GPS Tool

Program za celovito analizo GPS podatkov vožnje s preprostim grafičnim vmesnikom. Namenjen je voznikom, ki želijo analizirati svoje vožnje in pripraviti podatke za oceno ekonomičnosti.

#### Kaj program omogoča

Program vam omogoča, da iz GPS sledi vaše vožnje izveste pomembne informacije o vašem načinu vožnje. Ko naložite GPS datoteko (GPX format), program samodejno:

- **Analizira vašo pot** - Prikaže celotno prevoženo razdaljo, čas vožnje in spremembe nadmorske višine
- **Oceni hitrost vožnje** - Izračuna povprečno hitrost, najde mesta največjih hitrosti in prikaže graf hitrosti skozi čas
- **Določi način vožnje** - Vožnjo razdeli na segmente glede na to, kako ekonomično ste vozili (mirovanje, enakomerna vožnja, pospeševanje, zaviranje)
- **Ustvari interaktiven zemljevid** - Na zemljevidu prikaže vašo pot z barvami, ki označujejo različne načine vožnje
- **Pripravi podatke za AI analizo** - Samodejno pripravi datoteke, ki jih kasneje uporabite za podrobno AI analizo ekonomičnosti

#### Kako uporabljati program

**1. Zagon programa**
```bash
python Obdelava.py
```
Odpre se okno programa v polni velikosti zaslona.

**2. Osnovno delo s programom**

Ko se program odpre, boste videli:
- **Naslov "GPS Tool"** na vrhu
- **Gumb za preklapljanje teme** (svetla/temna) v zgornjem desnem kotu
- **Tri glavna področja za vnos**:
  - Izbira posamezne GPS datoteke
  - Ime za izhodni zemljevid
  - Izbira mape z več GPS datotekami (za hkratno obdelavo)

**3. Obdelava ene vožnje**

Za analizo ene vožnje:
1. Kliknite "Choose File" pri "Input GPS File"
2. Poiščite in izberite vašo GPX datoteko (to je datoteka, ki jo izvozite iz GPS naprave ali aplikacije)
3. V polje "Output Map Name" vpišite željeno ime zemljevida (npr. "voznja_maribor_ljubljana")
4. Kliknite "Generate Map"

**4. Obdelava več voženj hkrati**

Če imate mapo z več GPX datotekami:
1. Kliknite "Choose Folder" pri "Input GPS Folder"
2. Izberite mapo z vašimi GPX datotekami
3. Program bo samodejno obdelal vse datoteke in ustvaril zemljevide

**5. Razumevanje rezultatov**

Po obdelavi boste videli:
- **Statistike vožnje**: razdalja v km, povprečna hitrost, najvišja hitrost, višinska razlika, čas vožnje
- **Graf hitrosti**: kako se je vaša hitrost spreminjala tekom vožnje
- **Graf con vožnje**: kdaj ste vozili ekonomično (zeleno) in kdaj ne (rdeče)
- **Pot do zemljevida**: kje je shranjen vaš interaktivni zemljevid

**6. Kaj narediti z rezultati**

Program ustvari več datotek:
- **HTML zemljevid** (v mapi Maps/) - odprite ga v brskalniku za ogled poti
- **CSV datoteke** (v mapi Analysis/) - te uporabite v programu za AI analizo
- **Tekstovna datoteka s conami** - pregled kdaj ste vozili kako ekonomično

### Struktura map SIS

Ko uporabljate program Obdelava.py, se samodejno ustvarijo naslednje mape:

- **Maps/** - Tukaj najdete vse ustvarjene zemljevide
  - Datoteke: `ime_voznje.html` - odprite v brskalniku
  
- **Analysis/** - Tukaj so shranjene vse analize
  - `ime.csv` - osnovni podatki o vožnji
  - `ime_hybrid_predprocesirano.csv` - podatki za učenje modelov
  - `ime_hybrid_predict.csv` - **to datoteko uporabite v AI programu**
  
- **Assets/** - Sistemske datoteke
  - Ikone za svetlo/temno temo programa

### HybridPreprocessor - Priprava podatkov za AI

Ta komponenta samodejno pripravi GPS podatke za analizo z nevronskimi mrežami. Deluje v ozadju programa Obdelava.py.

#### Kaj naredi

- Razdeli vožnjo na 20-sekundne odseke
- Izračuna pospeške iz hitrosti
- Doda oznake ekonomičnosti za učenje
- Pripravi podatke v formatu za AI modele

#### Razumevanje con ekonomičnosti

Program vožnjo razdeli na 6 con:

- **Cona 0**: Mirovanje (avto stoji)
- **Cona 1**: Zelo ekonomično (skoraj konstantna hitrost)
- **Cona 2**: Ekonomično (majhne spremembe hitrosti)
- **Cona 3**: Zmerno ekonomično (zmerne spremembe)
- **Cona 4**: Neekonomično (velike spremembe)
- **Cona 5**: Zelo neekonomično (ekstremne spremembe)

Cona se določi glede na to, kako močno pospešujete ali zavirate.

## UI - Umetna inteligenca

Pri predmetu umetna inteligenca je glavna naloga izdelava nevronskih mrež, ki se bodo na podlagi zajetih označenih podatkov naučile analizirati neoznačene podatke.

### zdruzevanje.py - Program za oceno ekonomičnosti vožnje

Program, ki s pomočjo umetne inteligence analizira vašo vožnjo in oceni, kako ekonomično vozite. Uporablja napredne nevronske mreže, ki so se učile na tisočih primerov vožnje.

#### Kaj program omogoča

Program analizira podatke o vaši vožnji in vam poda:
- **Oceno ekonomičnosti** na lestvici od 0 (mirovanje) do 5 (zelo neekonomično)
- **Zaupanje v napoved** - kako prepričan je model v svojo oceno
- **Vizualizacijo na zemljevidu** - vašo pot pobarva glede na ekonomičnost vožnje
- **Analizo velikih količin podatkov** - lahko obdela celotne vožnje ali več voženj hkrati

#### Kako uporabljati program

**1. Zagon programa**
```bash
python zdruzevanje.py
```
Odpre se okno s tremi zavihki.

**2. Prvi korak - Nalaganje modelov (zavihek "Upravljanje Modelov")**

Preden lahko analizirate vožnjo, morate naložiti AI modele:

1. **Model za hitrost**: 
   - Kliknite "Naloži Model za Hitrost"
   - Izberite datoteko modela (končnica .pt)
   - Videli boste zeleno kljukico, ko je model naložen

2. **Model za pospešek** (opcijsko, vendar priporočeno):
   - Kliknite "Naloži Model za Pospešek"
   - Izberite datoteko modela
   - Za boljše rezultate uporabite oba modela

3. **Model za smer** (opcijsko):
   - Kliknite "Naloži Model za Smer"
   - Ta model analizira, kako pogosto menjavate smer

4. **Nastavitev pomembnosti modelov**:
   - Če imate naložena oba modela (hitrost in pospešek)
   - Lahko nastavite, kateri je pomembnejši
   - Privzeto: 60% hitrost, 40% pospešek

**3. Analiza posameznih odsekov (zavihek "Napovedi")**

Za hitro analizo kratkega odseka vožnje:

1. V prvo polje vnesite 20 vrednosti hitrosti (ločene s presledki)
   - Primer: `50 52 54 55 55 56 58 60 58 55 52 50 48 45 43 40 38 35 30 25`
   
2. V drugo polje lahko vnesite 20 vrednosti pospeškov (opcijsko)
   
3. Kliknite "Napovej"

4. Program vam pokaže:
   - Stopnjo ekonomičnosti (0-5)
   - Opis (npr. "Ekonomično", "Zelo neekonomično")
   - Odstotek zaupanja v napoved
   - Če imate več modelov, tudi skupno oceno

**4. Analiza celotne vožnje (zavihek "CSV Obdelava")**

Za analizo celotne vožnje:

1. **Priprava podatkov**:
   - Najprej morate obdelati GPX datoteko s programom Obdelava.py
   - Ta ustvari CSV datoteko s pripravljenimi podatki

2. **Obdelava v AI programu**:
   - Kliknite "Obdelaj CSV Datoteko"
   - Izberite datoteko `..._hybrid_predict.csv` iz mape Analysis
   - Program predlaga ime izhodne datoteke (lahko spremenite)
   - Kliknite Shrani

3. **Kaj dobite**:
   - CSV datoteko z ocenami za vsak 20-sekundni odsek
   - Statistiko obdelave (koliko odsekov, morebitne napake)
   - Analizo, kje se modeli ne strinjajo (če imate več modelov)

**5. Ustvarjanje zemljevida z ocenami**

Za vizualizacijo rezultatov na zemljevidu:

1. Kliknite "Izris Zemljevida"
2. Izberite:
   - Originalno GPX datoteko vaše vožnje
   - CSV datoteko z ocenami (tisto, ki ste jo pravkar ustvarili)
3. Program ustvari interaktiven zemljevid z:
   - Vašo potjo pobarvano glede na ekonomičnost
   - Animacijo vožnje (gumb za predvajanje)
   - Legendo barv
   - Označbo začetka in konca

**6. Razumevanje rezultatov**

Barve na zemljevidu pomenijo:
- **Modra**: Mirovanje (avto stoji)
- **Svetlo zelena**: Zelo ekonomična vožnja
- **Zelena**: Ekonomična vožnja
- **Oranžna**: Zmerno ekonomična vožnja
- **Rdeča**: Neekonomična vožnja
- **Temno rdeča**: Zelo neekonomična vožnja

**Nasveti za boljše rezultate**:
- Uporabljajte oba modela (hitrost in pospešek) za natančnejše ocene
- Pri daljših vožnjah pričakujte različne ocene za različne odseke
- Modri odseki (mirovanje) so normalni pri semaforjih in zastojih
- Cilj je čim več zelenih odsekov

### Tehnične podrobnosti modelov

Program uporablja napredne nevronske mreže tipa BiLSTM (Bidirectional Long Short-Term Memory), ki so posebej primerne za analizo časovnih podatkov kot so GPS sledi.

#### Kako delujejo modeli

1. **Vhodni podatki**: 20 zaporednih meritev (20 sekund vožnje)
2. **Analiza v obe smeri**: Model gleda naprej in nazaj za boljše razumevanje
3. **Attention mehanizem**: Model se osredotoči na najpomembnejše dele
4. **Združevanje informacij**: Končna ocena temelji na celotnem vzorcu

#### Natančnost modelov

- Modeli so učeni na tisočih realnih voženj
- Povprečna natančnost: 85-90%
- Najboljši rezultati pri kombinaciji več modelov
- Stalno izboljševanje z novimi podatki

## SPO - Sistemska programska oprema

Pri predmetu sistemska programska oprema je glavna naloga vzpostavitev infrastrukture za deljenje podatkov in programa znotraj skupine ter med napravo za zajemanje podatkov in napravo za analizo.

### Načrtovane funkcionalnosti
- MQTT sistem za real-time komunikacijo
- Prenos GPS podatkov med napravami
- Sinhronizacija rezultatov analize
- Povratne informacije vozniku v realnem času

## URG - Uvod v računalniško geometrijo

Pri predmetu uvod v računalniško geometrijo je glavna naloga izdelava 3D modelov za razne naprave.

### Načrtovani 3D modeli
- Držalo za telefon (naprava za zajemanje GPS)
- Držalo za prikazovalnik povratnih informacij
- Ohišje za računalniško enoto

## Uporaba - Celoten potek analize vožnje

### Praktični primer: Od GPS sledi do ocene ekonomičnosti

Predstavljajte si, da ste pravkar končali vožnjo iz Maribora v Ljubljano in želite analizirati, kako ekonomično ste vozili.

#### 1. Korak: Pridobitev GPS podatkov

Najprej potrebujete GPS sled vaše vožnje:
- **Če uporabljate telefon**: Aplikacije kot so Strava, GPX Tracker, ali GPS Logger
- **Če imate GPS napravo**: Izvozite podatke v GPX format
- **Pomembno**: Datoteka mora biti v GPX formatu (primer: `voznja_mb_lj.gpx`)

#### 2. Korak: Prva analiza s programom Obdelava.py

1. Zaženite program za obdelavo:
   ```bash
   cd SIS/Obdelava
   python Obdelava.py
   ```

2. V programu:
   - Kliknite "Choose File" in izberite vašo GPX datoteko
   - Vpišite ime za zemljevid, npr. "voznja_mb_lj_oktober"
   - Kliknite "Generate Map"

3. Počakajte nekaj sekund in preglejte:
   - **Statistike**: "Skupaj ste prevozili 120 km s povprečno hitrostjo 85 km/h"
   - **Graf hitrosti**: Vidite, kje ste vozili hitreje (avtocesta) in počasneje (mesta)
   - **Graf con**: Preliminary ocena vašega načina vožnje

4. Program je ustvaril:
   - Zemljevid v `Maps/voznja_mb_lj_oktober.html`
   - Podatke za AI v `Analysis/voznja_mb_lj.gpx_hybrid_predict.csv`

#### 3. Korak: Podrobna AI analiza z zdruzevanje.py

1. Zaženite AI program:
   ```bash
   cd UI
   python zdruzevanje.py
   ```

2. Najprej naložite modele:
   - V zavihku "Upravljanje Modelov"
   - Naložite vsaj model za hitrost (bolje je, če naložite tudi model za pospešek)
   - Potrdite, da piše "Model uspešno naložen"

3. Analizirajte celotno vožnjo:
   - Pojdite na zavihek "CSV Obdelava"
   - Kliknite "Obdelaj CSV Datoteko"
   - Izberite datoteko `voznja_mb_lj.gpx_hybrid_predict.csv`
   - Shranite kot `voznja_mb_lj_output.csv`

4. Ustvarite končni zemljevid:
   - Kliknite "Izris Zemljevida"
   - Izberite originalno GPX datoteko
   - Izberite pravkar ustvarjeno output CSV datoteko
   - Počakajte, da se zemljevid odpre v brskalniku

#### 4. Korak: Interpretacija rezultatov

Na končnem zemljevidu boste videli:

- **Zeleni odseki**: Tu ste vozili ekonomično (enakomerna hitrost, malo pospeševanja)
- **Oranžni odseki**: Zmerno ekonomično (nekaj pospeševanja in zaviranja)
- **Rdeči odseki**: Neekonomično (močno pospeševanje, naglo zaviranje)

Kliknite "Začni animacijo" za ogled celotne vožnje.

### Scenariji uporabe

#### Za vsakodnevno vožnjo na delo

1. Snemajte isto pot več dni zapored
2. Primerjajte zemljevide različnih dni
3. Ugotovite, kdaj vozite najbolj ekonomično
4. Prilagodite čas odhoda za bolj ekonomično vožnjo

#### Za daljše potovanje

1. Analizirajte različne odseke poti
2. Identificirajte "problematične" odseke
3. Načrtujte postanke na mestih, kjer vožnja postane neekonomična
4. Primerjajte različne poti do istega cilja

#### Za učenje ekonomične vožnje

1. Naredite testno vožnjo
2. Analizirajte rezultate
3. Naslednjič se osredotočite na:
   - Enakomernejšo hitrost
   - Predvidevanje prometa
   - Mehkejše pospeševanje in zaviranje
4. Primerjajte napredek

### Nasveti za najboljše rezultate

1. **Kvaliteta GPS signala**: Uporabljajte napravo z dobrim GPS sprejemnikom
2. **Rednost snemanja**: GPS naj snema lokacijo vsaj enkrat na sekundo
3. **Dolžina vožnje**: Analize so najboljše pri vožnjah daljših od 10 minut
4. **Vreme**: V slabem vremenu (dež, megla) so rezultati lahko drugačni

### Pogosta vprašanja

**V: Zakaj nekateri odseki nimajo ocene?**
O: Program potrebuje vsaj 20 sekund podatkov za oceno. Kratki odseki ali mesta s slabim GPS signalom morda ne bodo ocenjeni.

**V: Kaj pomeni "mirovanje" (modra barva)?**
O: To je normalno pri semaforjih, zastojih ali parkiranju. Ne šteje kot neekonomično.

**V: Ali lahko primerjam različne voznike?**
O: Da, če vsi uporabljajo isti sistem. Lahko primerjate zemljevide in statistike.

**V: Kako natančne so ocene?**
O: Modeli so učeni na tisočih primerov vožnje. Natančnost je običajno nad 85%, odvisno od kvalitete GPS podatkov.

## Tehnične podrobnosti

### Obdelava GPS signalov

Program Obdelava.py izvaja naslednje korake:

1. **Branje GPS podatkov**
   - Odpre GPX datoteko in prebere vse točke
   - Za vsako točko shrani koordinate, čas in višino
   
2. **Izračun hitrosti in razdalje**
   - Med zaporednimi točkami izračuna razdaljo
   - Iz razdalje in časa izračuna hitrost
   - Sešteje vse razdalje za skupno pot

3. **Določanje con vožnje**
   - Analizira spremembe hitrosti
   - Določi ekonomičnost glede na pospeševanje/zaviranje
   - Združi kratke segmente v daljše cone

4. **Priprava za AI**
   - Razdeli podatke na 20-sekundne odseke
   - Doda izračunane pospeške
   - Formatira za nevronske mreže

### AI analiza

Program zdruzevanje.py uporablja:

1. **Normalizacija podatkov**
   - Prilagodi vrednosti za boljše delovanje modelov
   - Uporablja IQR metodo, odporno na ekstremne vrednosti
   
2. **Napoved z več modeli**
   - Vsak model da svojo oceno
   - Program jih združi glede na nastavljene uteži
   - Končna ocena je najbolj verjetna

3. **Vizualizacija rezultatov**
   - Povezuje GPS koordinate z ocenami
   - Ustvari interaktiven zemljevid
   - Doda animacijo za lažje razumevanje

### Optimizacije za hitrost

- Obdelava velikih datotek po delih
- Učinkovito upravljanje pomnilnika
- Paralelno procesiranje kjer je mogoče
- Predpomnjenje izračunov

## Razširitve in prihodnji razvoj

### Načrtovane izboljšave

1. **Mobilna aplikacija**
   - Zajemanje GPS podatkov direktno na telefonu
   - Analiza v realnem času med vožnjo
   - Glasovni nasveti za ekonomičnejšo vožnjo

2. **Povezava z vozilom**
   - Branje podatkov iz OBD-II vmesnika
   - Natančnejši podatki o porabi goriva
   - Analiza vzdrževanja vozila

3. **Napredna analitika**
   - Primerjava z drugimi vozniki
   - Tedenski/mesečni trendi
   - Predlogi alternativnih poti

4. **Integracije**
   - Izvoz poročil v PDF
   - Deljenje rezultatov
   - Povezava s sistemi za upravljanje voznih parkov

### Kontakt in podpora

Za vprašanja o projektu se obrnite na člane skupine preko šolskih mailov ali mentorja prof. Nika Lukača.

---

Za vsak predmet je v GitHub repozitoriju poseben feature branch, v katerem se nahaja zbirka s programi, datotekami in dokumentacijo za vsak predmet.