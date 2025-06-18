# Vključitev potrebnih knjižnic
import paho.mqtt.client as mqtt
import time
from datetime import datetime
from prometheus_client import start_http_server, Counter, Histogram
import gpxpy.gpx
import json
from geopy.distance import geodesic

# Prometheus metrics
counter_sending = Counter('sending_counter', 'Number of messages sent')
counter_proccesed = Counter('procesed_counter', 'Number of messages processed')
counter_0 = Counter('stopnja_0', 'Number of messages received')
counter_1 = Counter('stopnja_1', 'Number of messages received')
counter_2 = Counter('stopnja_2', 'Number of messages received')
counter_3 = Counter('stopnja_3', 'Number of messages received')
counter_4 = Counter('stopnja_4', 'Number of messages received')
hitrosti = Histogram('Povprecne_hitrosti', 'Hitrosti')

# MQTT Broker
broker = "10.147.17.36"
port = 1883
topic = "/data"

# Callback ob povezavi
def on_connect(client, userdata, flags, reasonCode, properties=None):
    print("Povezava z MQTT: " + str(reasonCode))

# Nastavitev MQTT klienta
producer = mqtt.Client(client_id="producer_1", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
producer.on_connect = on_connect
producer.connect(broker, port, 60)

# Prometheus server
start_http_server(8000)

# Branje GPX datoteke
gpx_path = "14-Mar-2025-2005-keglevic.gpx"
with open(gpx_path, 'r') as gpx_file:
    gpx = gpxpy.parse(gpx_file)

# Pridobivanje točk
points = []
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            points.append(point)

# Pomnilniki
speed_list = []  # shranjene hitrosti za vsakih 10 točk

# Funkcija za določitev stopnje ekonomičnosti
def oceni_ekonomicnost(sprememba):
    if sprememba < 10:
        return 0, "zelo ekonomično"
    elif sprememba < 20:
        return 1, "ekonomično"
    elif sprememba < 30:
        return 2, "zmerno"
    elif sprememba < 40:
        return 3, "neekonomično"
    else:
        return 4, "zelo neekonomično"

# Glavna zanka
buffer_10 = []  # začasni seznam za 10 točk

for idx, point in enumerate(points):
    message = {
        "latitude": point.latitude,
        "longitude": point.longitude,
        "elevation": point.elevation,
        "time": point.time.isoformat() if point.time else datetime.utcnow().isoformat()
    }

    payload = json.dumps(message)
    ret = producer.publish(topic, payload, qos=1, retain=False)
    print(f"Pošiljanje: {payload}  Status: {ret.rc}")

    counter_sending.inc()
    buffer_10.append(point)

    # Ko zberemo 10 točk, izračunamo hitrost
    if len(buffer_10) == 10:
        start = buffer_10[0]
        end = buffer_10[-1]

        if start.time and end.time:
            dist_m = geodesic(
                (start.latitude, start.longitude),
                (end.latitude, end.longitude)
            ).meters
            time_diff_s = (end.time - start.time).total_seconds()

            if time_diff_s > 0:
                speed_km_h = (dist_m / time_diff_s) * 3.6
                hitrosti.observe(speed_km_h)
                print(f"🚗 Povprečna hitrost (točke {idx-9}–{idx}): {speed_km_h:.2f} km/h")

                # primerjava s prejšnjo 10-točkovno hitrostjo
                if speed_list:
                    sprememba = abs(speed_km_h - speed_list[-1])
                    stopnja, opis = oceni_ekonomicnost(sprememba)
                    if stopnja == 0:
                        counter_0.inc()
                    elif stopnja == 1:
                        counter_1.inc()
                    elif stopnja == 2:
                        counter_2.inc()
                    elif stopnja == 3:
                        counter_3.inc()
                    elif stopnja == 4:
                        counter_4.inc()
                    print(f"📊 Sprememba hitrosti: {sprememba:.2f} km/h – stopnja {stopnja} ({opis})")
                    counter_proccesed.inc()


                speed_list.append(speed_km_h)
            else:
                print("⚠️ Ni mogoče izračunati hitrosti – časovna razlika = 0")

        buffer_10 = []  # počistimo za naslednjih 10 točk

    time.sleep(1)

print("✅ Vse točke poslane.")
