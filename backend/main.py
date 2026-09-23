from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from jose import jwt
from datetime import datetime, timedelta, timezone
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()
# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))

from database.database import (
    register_student,
    login_student,
    get_student,
    get_predictions,
    save_prediction
)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Student Academic Intelligence System API",
    description="Backend API for SAIS mobile application",
    version="1.0.0"
)
SECRET_KEY = "sais-change-this-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ============================================================
# LOAD ML FILES
# ============================================================

MODEL_PATH = BASE_DIR / "best_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "label_encoders.pkl"


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODER_PATH)


# ============================================================
# INPUT DATA MODEL
# ============================================================

class PredictionInput(BaseModel):

    gender: str
    age: int
    attendance: float
    study_hours: float
    sleep_hours: float
    social_media_hours: float
    physical_activity: float
    mental_health: float
    internet_quality: str
    motivation_level: str
    parental_education: str
    family_income: str
    part_time_job: str
    previous_scores: float


class RegisterInput(BaseModel):

    name: str
    email: str
    password: str
    college: str
    branch: str
    year: str
    roll_number: str


class LoginInput(BaseModel):

    email: str
    password: str


def create_access_token(student_id):
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "student_id": student_id,
        "exp": expire
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def get_current_student_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        student_id = payload.get("student_id")

        if student_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )

        return int(student_id)

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token"
        )


# ============================================================
# ENCODER
# ============================================================

def encode_value(column, value):
    return int(
        encoders[column].transform([value])[0]
    )


def predict_score(data):
    features = pd.DataFrame({
        "gender": [encode_value("gender", data.gender)],
        "age": [data.age],
        "attendance": [data.attendance],
        "study_hours": [data.study_hours],
        "sleep_hours": [data.sleep_hours],
        "social_media_hours": [data.social_media_hours],
        "physical_activity": [data.physical_activity],
        "mental_health": [data.mental_health],
        "internet_quality": [encode_value("internet_quality", data.internet_quality)],
        "motivation_level": [encode_value("motivation_level", data.motivation_level)],
        "parental_education": [encode_value("parental_education", data.parental_education)],
        "family_income": [encode_value("family_income", data.family_income)],
        "part_time_job": [encode_value("part_time_job", data.part_time_job)],
        "previous_scores": [data.previous_scores]
    })

    return round(float(np.clip(model.predict(scaler.transform(features))[0], 0, 100)), 2)


@app.post("/register")
def register(data: RegisterInput):

    success, message = register_student(
        data.name,
        data.email,
        data.password,
        data.college,
        data.branch,
        data.year,
        data.roll_number
    )

    if success:
        return {
            "status": "success",
            "message": message
        }

    return {
        "status": "error",
        "message": message
    }


@app.post("/login")
def login(data: LoginInput):

    student = login_student(
        data.email,
        data.password
    )

    if student is None:
        return {
            "status": "error",
            "message": "Invalid email or password"
        }

    token = create_access_token(student[0])

    return {
        "status": "success",
        "message": "Login successful",
        "access_token": token,
        "token_type": "bearer",
        "student": {
            "id": student[0],
            "name": student[1],
            "email": student[2],
            "college": student[3],
            "branch": student[4],
            "year": student[5],
            "roll_number": student[6],
            "current_cgpa": student[7],
            "target_cgpa": student[8]
        }
    }


@app.post("/predict")
def predict_academic_performance(
    data: PredictionInput,
    student_id: int = Depends(get_current_student_id)
):

    score = predict_score(data)

    # Academic Health Score
    health_score = (
        (data.attendance / 100) * 25
        + (data.study_hours / 8) * 20
        + (min(data.sleep_hours, 8) / 8) * 15
        + ((5 - data.social_media_hours) / 5) * 10
        + (data.physical_activity / 5) * 10
        + (data.mental_health / 5) * 10
        + (data.previous_scores / 100) * 10
    )

    health_score = max(0, min(100, health_score))
    health_score = round(health_score, 1)

    # Performance level
    if score >= 85:
        performance = "Excellent"
    elif score >= 70:
        performance = "Good"
    elif score >= 50:
        performance = "Average"
    else:
        performance = "Needs Improvement"

    # Personalized tips
    tips = []

    if data.attendance < 75:
        tips.append(
            "Try to improve your class attendance. Regular attendance can help maintain consistency."
        )

    if data.study_hours < 2:
        tips.append(
            "Increase your focused study time gradually. Start with an additional 30 minutes per day."
        )

    if data.sleep_hours < 6:
        tips.append(
            "Try to maintain at least 6–8 hours of quality sleep every day."
        )

    if data.social_media_hours > 3:
        tips.append(
            "Consider reducing social media usage and use that time for focused study or revision."
        )

    if data.physical_activity < 3:
        tips.append(
            "Add some regular physical activity to your daily routine."
        )

    if data.mental_health < 3:
        tips.append(
            "Take regular breaks and maintain a healthy study routine to manage academic stress."
        )

    if data.previous_scores < 60:
        tips.append(
            "Review your previous exam mistakes and focus more on weak subjects."
        )

    if data.motivation_level.lower() in ["low", "medium"]:
        tips.append(
            "Set small daily academic goals and track your progress consistently."
        )

    if not tips:
        tips.append(
            "Your current academic routine looks balanced. Keep maintaining your consistency."
        )

    prediction_id = save_prediction(
        student_id=student_id,
        predicted_score=score,
        academic_health=health_score,
        attendance=data.attendance,
        study_hours=data.study_hours,
        sleep_hours=data.sleep_hours,
        social_media_hours=data.social_media_hours,
        physical_activity=data.physical_activity,
        mental_health=data.mental_health,
        previous_scores=data.previous_scores
    )

    return {
        "status": "success",
        "prediction_id": prediction_id,
        "student_id": student_id,
        "predicted_score": score,
        "academic_health": health_score,
        "performance": performance,
        "personalized_tips": tips,
        "message": "Academic performance analyzed and saved successfully"
    }

# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Student Academic Intelligence System API is running 🚀"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "online",
        "system": "SAIS",
        "model": "loaded",
        "message": "Backend and ML model are working successfully"
    }


@app.get("/profile")
def get_profile(
    student_id: int = Depends(get_current_student_id)
):

    student = get_student(student_id)

    if student is None:
        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    return {
        "status": "success",
        "student": {
            "id": student[0],
            "name": student[1],
            "email": student[2],
            "college": student[3],
            "branch": student[4],
            "year": student[5],
            "roll_number": student[6],
            "current_cgpa": student[7],
            "target_cgpa": student[8],
            "created_at": student[9]
        }
    }


@app.get("/prediction-history")
def prediction_history(
    student_id: int = Depends(get_current_student_id)
):

    predictions = get_predictions(student_id)

    history = []

    for prediction in predictions:
        history.append({
            "id": prediction[0],
            "predicted_score": prediction[1],
            "academic_health": prediction[2],
            "attendance": prediction[3],
            "study_hours": prediction[4],
            "sleep_hours": prediction[5],
            "social_media_hours": prediction[6],
            "physical_activity": prediction[7],
            "mental_health": prediction[8],
            "previous_scores": prediction[9],
            "created_at": prediction[10]
        })

    return {
        "status": "success",
        "student_id": student_id,
        "total_predictions": len(history),
        "predictions": history
    }