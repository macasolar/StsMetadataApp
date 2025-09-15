import base64
import json
import cv2
import numpy as np
import paho.mqtt.client as mqtt

# Path to OpenCV's pretrained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# MQTT configuration
MQTT_BROKER = "localhost"
MQTT_TOPIC = "metadata/cameraTest/consolidated_track"

def process_image(img_data, obj_type, bbox=None):
    # Decode base64 to numpy array
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if obj_type.lower() == "human" and bbox:
        # Extract bounding box region (assuming normalized coordinates)
        h, w = img.shape[:2]
        top = int(bbox["top"] * h)
        bottom = int(bbox["bottom"] * h)
        left = int(bbox["left"] * w)
        right = int(bbox["right"] * w)
        img = img[top:bottom, left:right]

        # Detect face in the cropped human
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            x, y, w_f, h_f = faces[0]
            img = img[y:y+h_f, x:x+w_f]

    # Save the resulting image
    filename = f"{obj_type}_{np.random.randint(10000)}.jpg"
    cv2.imwrite(filename, img)
    print(f"Saved image: {filename}")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
    except json.JSONDecodeError:
        return  # skip non-JSON messages

    if "classes" not in data or "image" not in data:
        return  # skip other types

    obj_type = data["classes"][0]["type"]
    img_data = data["image"]["data"]
    bbox = data["image"].get("bounding_box")

    if obj_type.lower() in ["human", "face"]:
        process_image(img_data, obj_type, bbox)

client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER)
client.subscribe(MQTT_TOPIC)
client.loop_forever()
