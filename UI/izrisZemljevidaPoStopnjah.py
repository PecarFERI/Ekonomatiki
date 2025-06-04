import gpxpy
import folium
from folium import PolyLine
from tkinter import Tk, filedialog
import os

#barve za stopnje
colors = {
    0: 'blue',
    1: 'lightgreen',
    2: 'green',
    3: 'darkorange',
    4: 'red',
    5: 'darkred'
}

def preberi_koordinate(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)

    koordinate = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                koordinate.append((point.latitude, point.longitude))
    return koordinate

def preberi_stopnje_iz_matrike(stopnje_file):
    stopnje = []
    with open(stopnje_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                stopnja = int(parts[-1])
                hitrosti = parts[:-1]
                stevilo_veljavnih = sum(1 for h in hitrosti if h.strip() != "0.0")
                stopnje.extend([stopnja] * stevilo_veljavnih)
            except ValueError:
                print(f"Opozorilo: preskakujem vrstico z napako: {line}")
    return stopnje

def dodaj_legendo(m):
    legenda_html = """
     <div style='position: fixed; 
                 bottom: 50px; left: 50px; width: 180px; height: 220px; 
                 background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                 padding: 10px;'>
     <b>Legenda stopenj:</b><br>
     <i style='background: blue; width: 10px; height: 10px; display: inline-block;'></i> 0 – Mirovanje<br>
     <i style='background: lightgreen; width: 10px; height: 10px; display: inline-block;'></i> 1<br>
     <i style='background: green; width: 10px; height: 10px; display: inline-block;'></i> 2<br>
     <i style='background: darkorange; width: 10px; height: 10px; display: inline-block;'></i> 3<br>
     <i style='background: red; width: 10px; height: 10px; display: inline-block;'></i> 4<br>
     <i style='background: darkred; width: 10px; height: 10px; display: inline-block;'></i> 5 
     </div>
    """
    m.get_root().html.add_child(folium.Element(legenda_html))

def izrisi_pot_na_zemljevidu(gpx_path, levels_path):
    koordinate = preberi_koordinate(gpx_path)
    stopnje = preberi_stopnje_iz_matrike(levels_path)

    if len(koordinate) < 2:
        print("Premalo koordinat za izris poti.")
        return

    if len(koordinate) - 1 > len(stopnje):
        print(f"Opozorilo: več koordinat kot stopenj. Odrezujemo {len(koordinate) - 1 - len(stopnje)} presežnih točk.")
        koordinate = koordinate[:len(stopnje) + 1]
    elif len(koordinate) - 1 < len(stopnje):
        print(f"Opozorilo: več stopenj kot koordinat. Odrezujemo {len(stopnje) - (len(koordinate) - 1)} presežnih stopenj.")
        stopnje = stopnje[:len(koordinate) - 1]

    map_center = koordinate[0]
    tileset = "CartoDB positron"
    m = folium.Map(location=map_center, zoom_start=14, tiles=tileset)

    for i in range(len(stopnje)):
        segment = [koordinate[i], koordinate[i + 1]]
        level = stopnje[i]
        color = colors.get(level, 'gray')

        PolyLine(segment, color=color, weight=5, opacity=0.8).add_to(m)

    dodaj_legendo(m)
    
    base_name = os.path.splitext(os.path.basename(gpx_path))[0]
    output_file = f"{base_name}_map.html"
    m.save(output_file)

    print(f"Zemljevid shranjen kot '{output_file}'.")

def izberi_datoteko(naslov="Izberi datoteko"):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=naslov)
    return file_path


#glavni programcek
gpx_file = izberi_datoteko("Izberi GPX datoteko")
if gpx_file:
    levels_file = izberi_datoteko("Izberi datoteko s stopnjami (ena vrstica = ena sekunda, vrednosti 0–5)")
    if levels_file:
        izrisi_pot_na_zemljevidu(gpx_file, levels_file)
    else:
        print("Nobena datoteka s stopnjami ni bila izbrana.")
else:
    print("Nobena GPX datoteka ni bila izbrana.")
