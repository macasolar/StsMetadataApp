import os
import psycopg2
import numpy as np
import cv2
import insightface

# ---------- DB CONFIG ----------
DB_CONFIG = {
    "dbname": "yourdbname",
    "user": "youruser",
    "password": "yourpassword",
    "host": "localhost",
    "port": 5432
}

# ---------- INIT MODELS ----------
# Load RetinaFace for detection + ArcFace for embeddings
app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# ---------- CONNECT TO POSTGRES ----------
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# ---------- HELPER FUNCTION ----------
def insert_employee(name, embedding):
    embedding_list = embedding.tolist()
    cur.execute(
        "INSERT INTO employees (name, embedding) VALUES (%s, %s) "
        "ON CONFLICT (name) DO UPDATE SET embedding = EXCLUDED.embedding;",
        (name, embedding_list)
    )
    conn.commit()

# ---------- PROCESS FOLDER ----------
folder = "employees"

for file in os.listdir(folder):
    if not (file.endswith(".jpg") or file.endswith(".png")):
        continue

    name = os.path.splitext(file)[0]
    path = os.path.join(folder, file)

    img = cv2.imread(path)
    faces = app.get(img)

    if len(faces) == 0:
        print(f"[WARN] No face found in {file}")
        continue

    # Take the largest detected face (highest bbox area)
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    embedding = face.normed_embedding  # 512-dim vector

    print(f"[INFO] Inserting {name} into DB")
    insert_employee(name, np.array(embedding))

print("[DONE] All employee embeddings inserted.")

# ---------- CLOSE DB ----------
cur.close()
conn.close()
