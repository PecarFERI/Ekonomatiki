import paho.mqtt.client as mqtt
import json
from datetime import datetime
from geopy.distance import geodesic
import time
from collections import deque  # Import deque for efficient rolling buffer

broker = "localhost"
port = 1883
topic = "test/topic"

gps_buffer = deque(maxlen=20)  # Buffer to hold the last 20 points


def on_connect(client, userdata, flags, rc):  # Adjusted for CallbackAPIVersion.VERSION2
    print("Povezano z MQTT brokerjem (rc =", rc, ")")
    client.subscribe(topic, qos=1)


def on_message(client, userdata, msg):
    global gps_buffer  # Still need global to modify the deque
    try:
        data = json.loads(msg.payload.decode())
        print(f"Prejeto: {data}")

        if not all(k in data for k in ("lat", "lon", "time")):
            print("[OPOZORILO] Manjkajoča polja v podatku, preskakujem.")
            return

        point = {
            "lat": float(data["lat"]),
            "lon": float(data["lon"]),
            "time": datetime.fromisoformat(data["time"])
        }

        gps_buffer.append(point)


        if len(gps_buffer) == 20:  # Changed from >= and % 20 to simply == 20
            # Since deque already holds the last 20, we don't need gps_buffer[-20:]
            # We can directly iterate over gps_buffer

            total_distance = 0.0
            total_time = 0.0

            for i in range(1, 20):
                p1 = gps_buffer[i - 1]
                p2 = gps_buffer[i]

                dist = geodesic((p1["lat"], p1["lon"]), (p2["lat"], p2["lon"])).meters
                dt = (p2["time"] - p1["time"]).total_seconds()

                if dt > 0:
                    total_distance += dist
                    total_time += dt

            if total_time > 0:
                avg_speed_kmh = (total_distance / total_time) * 3.6
                print(f"[ANALIZA] Povprečna hitrost za zadnjih 20 točk: {avg_speed_kmh:.2f} km/h")
            else:
                print("[ANALIZA] Ni mogoče izračunati hitrosti (čas = 0).")
        else:
            print(f"[INFO] Trenutno število točk v medpomnilniku: {len(gps_buffer)}")

    except Exception as e:
        print(f"[NAPAKA v on_message] {e}")


# --- Glavni del ---
# Use CallbackAPIVersion.VERSION2
client = mqtt.Client(client_id="consumer1", clean_session=False)
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker, port)
client.loop_start()  # NE blokira

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Ustavljam program...")
    client.loop_stop()
    client.disconnect()