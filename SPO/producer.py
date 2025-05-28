import time
import paho.mqtt.client as mqtt

# Zamenjaj z ZeroTier IP naslovom naprave, kjer teče MQTT broker
broker = "10.147.17.36"
port = 1883
topic = "test/topic"

# Ustvari MQTT client z enoličnim ID-jem
client = mqtt.Client(client_id="producer1")

# Poveži se na MQTT broker prek ZeroTier IP naslova
client.connect(broker, port)

# Pošiljaj sporočila vsakih 2 sekundi
while True:
    message = f"Hello MQTT at {time.ctime()}"
    client.publish(topic, message, qos=1)
    print(f"Sent: {message}")
    time.sleep(2)
