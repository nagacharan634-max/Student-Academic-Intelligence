import sqlite3
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from pwdlib import PasswordHash

password_hash = PasswordHash.recommended()

# Database file location
DB_PATH = Path(__file__).parent / "sais.db"


# Connect to database
def get_connection():
    print("DATABASE USED:", DB_PATH.resolve())
    return sqlite3.connect(DB_PATH)


# Create database tables
def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            college TEXT,
            branch TEXT,
            year TEXT,
            roll_number TEXT,
            current_cgpa REAL DEFAULT 0,
            target_cgpa REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            predicted_score REAL,
            academic_health REAL,
            attendance REAL,
            study_hours REAL,
            sleep_hours REAL,
            social_media_hours REAL,
            physical_activity REAL,
            mental_health REAL,
            previous_scores REAL,
            created_at TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    conn.commit()
    conn.close()


# Register a new student
def register_student(
    name,
    email,
    password,
    college,
    branch,
    year,
    roll_number
):
    email = email.strip().lower()
    password = password_hash.hash(password.strip())

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Check if this email already exists
        cursor.execute(
            "SELECT id FROM students WHERE email = ?",
            (email,)
        )

        existing_student = cursor.fetchone()

        if existing_student:
            return False, "Email already registered."

        # Insert new student
        cursor.execute("""
            INSERT INTO students
            (name, email, password, college, branch, year, roll_number)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            email,
            password,
            college,
            branch,
            year,
            roll_number
        ))

        conn.commit()

        return True, "Account created successfully!"

    except sqlite3.Error as e:
        print("DATABASE ERROR:", e)
        return False, f"Database error: {e}"

    finally:
        conn.close()
# Login student
def login_student(email, password):
    conn = get_connection()
    cursor = conn.cursor()

    email = email.strip().lower()
    password = password.strip()

    cursor.execute("""
        SELECT
            id,
            name,
            email,
            password,
            college,
            branch,
            year,
            roll_number,
            current_cgpa,
            target_cgpa
        FROM students
        WHERE LOWER(email) = ?
    """, (email,))

    student = cursor.fetchone()
    conn.close()

    if student is None:
        return None

    stored_password = student[3]

    try:
        password_valid = password_hash.verify(
            password,
            stored_password
        )
    except Exception:
        password_valid = False

    if not password_valid:
        return None

    return (
        student[0],
        student[1],
        student[2],
        student[4],
        student[5],
        student[6],
        student[7],
        student[8],
        student[9]
    )


def get_student(student_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            name,
            email,
            college,
            branch,
            year,
            roll_number,
            current_cgpa,
            target_cgpa,
            created_at
        FROM students
        WHERE id = ?
    """, (student_id,))

    student = cursor.fetchone()
    conn.close()

    return student


def save_prediction(
    student_id,
    predicted_score,
    academic_health,
    attendance,
    study_hours,
    sleep_hours,
    social_media_hours,
    physical_activity,
    mental_health,
    previous_scores
):
    conn = get_connection()
    cursor = conn.cursor()

    created_at = datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO predictions (
            student_id,
            predicted_score,
            academic_health,
            attendance,
            study_hours,
            sleep_hours,
            social_media_hours,
            physical_activity,
            mental_health,
            previous_scores,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        student_id,
        predicted_score,
        academic_health,
        attendance,
        study_hours,
        sleep_hours,
        social_media_hours,
        physical_activity,
        mental_health,
        previous_scores,
        created_at
    ))

    conn.commit()
    prediction_id = cursor.lastrowid
    conn.close()

    return prediction_id


def get_predictions(student_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            predicted_score,
            academic_health,
            attendance,
            study_hours,
            sleep_hours,
            social_media_hours,
            physical_activity,
            mental_health,
            previous_scores,
            created_at
        FROM predictions
        WHERE student_id = ?
        ORDER BY created_at DESC
    """, (student_id,))

    predictions = cursor.fetchall()

    conn.close()

    return predictions

def update_academic_goals(
    student_id,
    current_cgpa,
    target_cgpa
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE students
        SET current_cgpa = ?,
            target_cgpa = ?
        WHERE id = ?
    """, (
        current_cgpa,
        target_cgpa,
        student_id
    ))

    conn.commit()
    conn.close()

    return True