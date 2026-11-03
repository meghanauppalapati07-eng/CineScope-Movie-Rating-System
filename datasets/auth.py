
import hashlib
import hmac
import secrets

import streamlit as st
from db_config import get_connection


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    ).hex()

    return f"{salt}${password_hash}"


def verify_password(password, stored_hash):
    try:
        salt, saved_hash = stored_hash.split("$", 1)
        new_hash = hash_password(password, salt).split("$", 1)[1]
        return hmac.compare_digest(new_hash, saved_hash)
    except (ValueError, AttributeError):
        return False


def register_user(display_name, email, password):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO app_users (display_name, email, password_hash)
            VALUES (%s, %s, %s)
            """,
            (display_name.strip(), email.strip().lower(), hash_password(password)),
        )
        connection.commit()
        return True, "Account created successfully!"
    except Exception as error:
        connection.rollback()
        if getattr(error, "errno", None) == 1062:
            return False, "This email is already registered."
        return False, "Could not create account. Please check your database."
    finally:
        cursor.close()
        connection.close()


def login_user(email, password):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            SELECT user_id, display_name, password_hash
            FROM app_users
            WHERE email = %s
            """,
            (email.strip().lower(),),
        )
        user = cursor.fetchone()

        if user and verify_password(password, user[2]):
            st.session_state["user_id"] = user[0]
            st.session_state["display_name"] = user[1]
            return True

        return False
    finally:
        cursor.close()
        connection.close()


def logout_user():
    st.session_state.pop("user_id", None)
    st.session_state.pop("display_name", None)

def save_movie_rating(user_id, movie_id, rating):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO user_ratings (user_id, movie_id, rating)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                rating = VALUES(rating),
                rated_at = CURRENT_TIMESTAMP
            """,
            (user_id, movie_id, rating),
        )
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()


def save_movie_review(user_id, movie_id, review_text):
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO user_reviews (user_id, movie_id, review_text)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                review_text = VALUES(review_text),
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, movie_id, review_text.strip()),
        )
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()
