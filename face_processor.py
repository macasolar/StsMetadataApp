# face_processor.py
import base64
from io import BytesIO
from PIL import Image
import os
import psycopg2
import numpy as np
import insightface
import cv2
from dotenv import load_dotenv

load_dotenv()


class FaceProcessor:
    def __init__(self):
        # Load face analysis model
        self.app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        # Load DB config from env
        self.db_config = {
            "dbname": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
            "host": os.getenv("POSTGRES_HOST"),
            "port": int(os.getenv("POSTGRES_PORT", 5432))
        }
        self.threshold = 0.7  # Recommended cosine distance threshold

    def decode_base64_image(self, base64_string):
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data))

    def recognize_face(self, image_data):
        # Decode image and get embedding
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces = self.app.get(img)
        if not faces:
            return "unknown"
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        embedding = face.normed_embedding.tolist()

        # Query DB for closest match
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("""
                SELECT name, embedding <-> %s::vector AS distance
                FROM people
                ORDER BY distance ASC
                LIMIT 1;
            """, (embedding,))
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result and result[1] is not None and result[1] < self.threshold:
                return result[0]
            else:
                return "unknown"
        except Exception as e:
            print(f"[ERROR] DB query failed: {e}")
            return "unknown"

    def log_person_entry(self, person_id, camera_topic):
        self.db_manager.insert_person_entry(person_id, camera_topic)
