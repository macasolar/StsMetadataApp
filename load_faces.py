import os
import psycopg2
import numpy as np
import cv2
import insightface

# ---------- DB CONFIG ----------
DB_CONFIG = {
    "dbname": "faces",
    "user": "face_app",
    "password": "Sts2025$",
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
def insert_person(name, image_path, embedding):
    embedding_list = embedding.tolist()  # convert NumPy array to Python list
    cur.execute(
        """
        INSERT INTO people (name, image_path, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (name) DO UPDATE
        SET image_path = EXCLUDED.image_path,
            embedding  = EXCLUDED.embedding;
        """,
        (name, image_path, embedding_list)
    )
    conn.commit()

# ---------- PROCESS FOLDER ----------
folder = "faces"
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
    insert_person(name, path, np.array(embedding))

print("[DONE] All embeddings inserted.")

# ---------- CLOSE DB ----------
cur.close()
conn.close()
