# Vključitev potrebnih knjižnic
import paho.mqtt.client as mqtt
import json
from datetime import datetime
from geopy.distance import geodesic

# MQTT nastavitve
broker = "127.0.0.1"
port = 1883
topic = "/data"

# Pomnilniki
buffer_10 = []
speed_list = []

# Funkcija za oceno ekonomičnosti
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

# Callback funkcije
def on_connect(client, userdata, flags, reasonCode, properties=None):
    print("Povezava z MQTT: " + str(reasonCode))

def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Prijava na topic")

def on_message(client, userdata, msg):
    global buffer_10, speed_list

    podatki = json.loads(msg.payload.decode())
    print("📥 Prejeto:", podatki)

    if "latitude" in podatki and "longitude" in podatki and "time" in podatki:
        try:
            podatki["time"] = datetime.fromisoformat(podatki["time"])
        except:
            print("⚠️ Neveljaven časovni zapis!")
            return

        buffer_10.append(podatki)

        # Ko imamo 10 točk, izračunamo hitrost
        if len(buffer_10) == 10:
            start = buffer_10[0]
            end = buffer_10[-1]

            coords_start = (start["latitude"], start["longitude"])
            coords_end = (end["latitude"], end["longitude"])

            dist_m = geodesic(coords_start, coords_end).meters
            time_diff_s = (end["time"] - start["time"]).total_seconds()

            if time_diff_s > 0:
                speed_km_h = (dist_m / time_diff_s) * 3.6
                print(f"🚗 Povprečna hitrost (10 točk): {speed_km_h:.2f} km/h")

                if speed_list:
                    sprememba = abs(speed_km_h - speed_list[-1])
                else:
                    sprememba = speed_km_h
                stopnja, opis = oceni_ekonomicnost(sprememba)
                print(f"📊 Sprememba hitrosti: {sprememba:.2f} km/h – stopnja {stopnja} ({opis})")

                speed_list.append(speed_km_h)
            else:
                print("⚠️ Neveljavna časovna razlika.")

            buffer_10 = []

# Nastavitev MQTT klienta
client = mqtt.Client(client_id="client_1", clean_session=False, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_subscribe = on_subscribe
client.on_message = on_message

client.connect(broker, port, 60)
client.subscribe(topic, qos=1)

# Zanka poslušanja
client.loop_forever()
