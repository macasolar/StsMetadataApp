import paho.mqtt.client as mqtt
import json
import base64
import os
from dotenv import load_dotenv
from face_processor import FaceProcessor

# Load environment variables
load_dotenv()


def is_face_event(data):
    # Filter logic for face events: only pass if any class has type 'face' (case-insensitive)
    classes = data.get('classes', [])
    return any(
        isinstance(cls, dict) and str(cls.get('type', '')).lower() == 'face'
        for cls in classes
    )


class MQTTFaceHandler:
    def __init__(self):
        self.client = mqtt.Client()
        self.face_processor = FaceProcessor()
        self.mqtt_host = os.getenv("MQTT_BROKER_HOST", "localhost")
        self.mqtt_port = int(os.getenv("MQTT_BROKER_PORT", 1883))
        self.topic_cam1 = os.getenv("MQTT_TOPIC_CAM1")
        self.topic_cam2 = os.getenv("MQTT_TOPIC_CAM2")
        self.setup_handlers()

    def setup_handlers(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        if self.topic_cam1:
            client.subscribe(self.topic_cam1)
        if self.topic_cam2:
            client.subscribe(self.topic_cam2)

    def on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            if is_face_event(data):
                self.process_face_event(data, msg.topic)
        except Exception as e:
            print(f"Error processing message: {e}")

    def process_face_event(self, data, topic):
        # Extract base64 image and process
        image_info = data.get('image')
        if image_info and 'data' in image_info:
            image_data = base64.b64decode(image_info['data'])
            person_name = self.face_processor.recognize_face(image_data)
            # Extract camera name from topic (e.g., cam1 or cam2)
            cam = topic.split('/')[-1] if '/' in topic else topic
            print(f"[EVENT] {cam}: {person_name}")
            # Optionally log to DB or further processing
            # self.face_processor.log_person_entry(person_name, topic)
