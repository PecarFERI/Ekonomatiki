import time
import json
import gpxpy
import gpxpy.gpx
import paho.mqtt.client as mqtt

# MQTT nastavitve
broker = "localhost"
port = 1883
topic = "test/topic"
client = mqtt.Client(client_id="producer1")
client.connect(broker, port)

# Preberi GPX datoteko
with open("27-Apr-2025-1742.gpx", "r") as gpx_file:  # zamenjaj z imenom tvoje .gpx datoteke
    gpx = gpxpy.parse(gpx_file)

# Pošlji vsako točko kot JSON prek MQTT
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            message = {
                "lat": point.latitude,
                "lon": point.longitude,
                "ele": point.elevation,
                "time": point.time.isoformat() if point.time else None
            }

            message_json = json.dumps(message)
            client.publish(topic, message_json, qos=1)
            print(f"Sent: {message_json}")
            time.sleep(1)  # simulacija realnega časa (1s med točkami)

print("Vse točke so bile poslane.")
