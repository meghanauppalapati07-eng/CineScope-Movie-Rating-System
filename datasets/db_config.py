
import os
import mysql.connector
import streamlit as st


def get_setting(name, default=None):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def get_connection():
    return mysql.connector.connect(
        host=get_setting("DB_HOST", "localhost"),
        port=int(get_setting("DB_PORT", 3306)),
        user=get_setting("DB_USER", "root"),
        password=get_setting("DB_PASSWORD", ""),
        database=get_setting("DB_NAME", "movie_rating_db"),
        ssl_disabled=False,
        connection_timeout=20
    )
