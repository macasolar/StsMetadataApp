CREATE TABLE people (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    image_path TEXT,
    embedding vector(512)  -- ArcFace gives 512-dim vectors
);