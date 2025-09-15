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
app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# ---------- CONNECT TO POSTGRES ----------
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# ---------- HELPER FUNCTION: compute embedding ----------
def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    return face.normed_embedding

# ---------- LOAD QUERY IMAGE ----------
query_image = "faces/foto.jpg"
query_embedding = get_embedding(query_image)

if query_embedding is None:
    print("[ERROR] No face detected in query image.")
    exit()

# Convert to list for psycopg2
query_embedding_list = query_embedding.tolist()

# ---------- PGVector QUERY ----------
# We use <-> operator for cosine distance (or euclidean if index is set differently)
cur.execute("""
    SELECT name, image_path, embedding <-> %s AS distance
    FROM people
    ORDER BY distance ASC
    LIMIT 1;
""", (query_embedding_list,))

result = cur.fetchone()
if result:
    name, path, distance = result
    print(f"[RESULT] Closest match: {name} ({path}) with distance {distance:.4f}")
else:
    print("[RESULT] No matches found.")

# ---------- CLOSE DB ----------
cur.close()
conn.close()
