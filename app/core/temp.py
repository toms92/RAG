import sqlite3

# Il path che vede PyCharm sull'host
conn = sqlite3.connect("/app/core/data/vector_db/chroma.sqlite3")
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM embeddings")
print("Embeddings:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM collections")
print("Collections:", cursor.fetchone()[0])

cursor.execute("SELECT * FROM collections LIMIT 3")
print("Collections rows:", cursor.fetchall())

conn.close()