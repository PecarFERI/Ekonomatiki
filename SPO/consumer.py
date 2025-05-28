import paho.mqtt.client as mqtt

# Zamenjaj z ZeroTier IP naslovom naprave, kjer teče MQTT broker
broker = "10.147.17.36"
port = 1883
topic = "test/topic"

# Callback za ob vzpostavitvi povezave
def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected with result code " + str(rc))
    client.subscribe(topic, qos=1)

# Callback za prejeta sporočila
def on_message(client, userdata, msg):
    print(f"Received: {msg.payload.decode()} on topic {msg.topic}")

# Ustvari MQTT client z enoličnim ID-jem in brez clean session (za vzdrževanje seje)
client = mqtt.Client(client_id="consumer1", clean_session=False)
client.on_connect = on_connect
client.on_message = on_message

# Poveži se na MQTT broker prek ZeroTier IP naslova
client.connect(broker, port)

# Neskončna zanka za poslušanje sporočil
client.loop_forever()
