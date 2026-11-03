
from pathlib import Path
import os
import pandas as pd
import mysql.connector

BASE_DIR = Path(__file__).parent
movies = pd.read_csv(BASE_DIR / "datasets" / "movies.csv")
ratings = pd.read_csv(BASE_DIR / "datasets" / "ratings.csv")

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password=os.getenv("DB_PASSWORD"),
    database="movie_rating_db"
)

cursor = connection.cursor()

try:
    cursor.execute("DELETE FROM ratings")
    cursor.execute("DELETE FROM movies")

    movie_data = list(
        movies[["movieId", "title", "genres"]]
        .itertuples(index=False, name=None)
    )

    cursor.executemany(
        "INSERT INTO movies (movieId, title, genres) VALUES (%s, %s, %s)",
        movie_data
    )

    rating_data = list(
        ratings[["userId", "movieId", "rating", "timestamp"]]
        .itertuples(index=False, name=None)
    )

    cursor.executemany(
        "INSERT INTO ratings (userId, movieId, rating, timestamp) VALUES (%s, %s, %s, %s)",
        rating_data
    )

    connection.commit()
    print("Movies inserted:", len(movie_data))
    print("Ratings inserted:", len(rating_data))
    print("Data imported successfully!")

except Exception as error:
    connection.rollback()
    print("Import failed:", error)

finally:
    cursor.close()
    connection.close()
