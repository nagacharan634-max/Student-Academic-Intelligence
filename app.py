
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from database.database import (
    create_tables,
    register_student,
    login_student,
    save_prediction,
    get_predictions,
    update_academic_goals
)
create_tables()
# ============================================================
# STUDENT ACADEMIC INTELLIGENCE SYSTEM - V3
# ============================================================

st.set_page_config(
    page_title="Student Academic Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- STYLE ---------------------------
st.markdown("""
<style>
:root{--bg:#070b14;--panel:#0d1422;--panel2:#111b2d;--border:#263650;--text:#edf4ff;--muted:#9aabc4;--accent:#7c5cff;--cyan:#00d4ff;}
.stApp{background:radial-gradient(circle at 5% 0%,rgba(124,92,255,.14),transparent 28%),radial-gradient(circle at 95% 5%,rgba(0,212,255,.08),transparent 25%),var(--bg);color:var(--text);}
.block-container{padding-top:1.1rem;max-width:1500px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#090f1c,#070b14);border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:#dce7f8;}
h1,h2,h3,h4,p,label,span{color:var(--text);}
.hero{background:linear-gradient(135deg,rgba(124,92,255,.96),rgba(49,46,129,.95) 52%,rgba(0,212,255,.72));padding:32px 36px;border-radius:26px;color:white;margin-bottom:22px;border:1px solid rgba(255,255,255,.14);box-shadow:0 22px 60px rgba(0,0,0,.35);}
.hero h1{margin:0 0 8px;font-size:38px;color:white!important;}
.hero p{margin:0;font-size:16px;color:#eef6ff!important;}
.section-title{color:#f4f7ff!important;font-size:25px;font-weight:800;margin:25px 0 13px;}
.card{background:linear-gradient(145deg,rgba(17,27,46,.97),rgba(9,15,27,.97))!important;border:1px solid var(--border);border-radius:20px;padding:21px;box-shadow:0 12px 35px rgba(0,0,0,.22);margin-bottom:14px;}
.card h3,.card h4,.card p,.card li{color:#eaf1ff!important;}
.card p,.card li{line-height:1.6;color:#aebbd0!important;}
.feature-card{min-height:185px}.feature-icon{font-size:32px;}
.profile{background:linear-gradient(135deg,rgba(124,92,255,.14),rgba(0,212,255,.06))!important;border:1px solid #304266;border-radius:20px;padding:22px;}
.profile h2,.profile p{color:#eef4ff!important;}
.info,.tip,.success,.warning,.danger{border-radius:14px;padding:14px 17px;margin:9px 0;}
.info{background:#0d1b31!important;border-left:5px solid #38bdf8;color:#cfeaff!important;}
.tip{background:#211a0d!important;border-left:5px solid #f59e0b;color:#ffe8b0!important;}
.success{background:#0c2117!important;border-left:5px solid #22c55e;color:#c8f7d7!important;}
.warning{background:#24170c!important;border-left:5px solid #f59e0b;color:#ffe0a8!important;}
.danger{background:#260f17!important;border-left:5px solid #fb7185;color:#ffd0d8!important;}
.login{max-width:800px;margin:20px auto;background:linear-gradient(145deg,#101a2c,#0b1220)!important;border:1px solid var(--border);border-radius:24px;padding:35px;box-shadow:0 18px 55px rgba(0,0,0,.35);}
.login h1,.login p{color:#eef4ff!important;}
.stButton>button{border-radius:12px;border:1px solid #35496b;min-height:44px;font-weight:700;}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#6d4aff,#00bde7);color:white;border:0;}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,.stTextInput input,.stNumberInput input{background:#0d1627!important;color:#eef4ff!important;border-color:#2b3c59!important;}
.stDataFrame{border:1px solid var(--border);border-radius:14px;overflow:hidden;}
.footer{text-align:center;color:#667895!important;padding:20px 0;font-size:13px;}
</style>
""", unsafe_allow_html=True)

# --------------------------- FILES ---------------------------
@st.cache_resource
def load_files():
    return (
        joblib.load("best_model.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("label_encoders.pkl")
    )

@st.cache_data
def load_data():
    return pd.read_csv("Final_Project_Dataset.csv")

try:
    model, scaler, encoders = load_files()
    df = load_data()
except Exception as e:
    st.error(f"Project files could not be loaded: {e}")
    st.stop()

# ------------------------- SESSION ---------------------------
for key, default in {
    "logged_in": False,
    "student_id": None,
    "name": "",
    "email": "",
    "college": "",
    "year": "",
    "branch": "CSE",
    "roll_number": "",
    "current_cgpa": 0.0,
    "target_cgpa": 0.0,
    "history": [],
    "last_values": None,
    "last_score": None,
    "last_health": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------- HELPERS ---------------------------
def dark_fig(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce7f8"),
        margin=dict(l=30,r=20,t=55,b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig

def enc(col, value):
    return int(encoders[col].transform([value])[0])

def predict(v):
    X = pd.DataFrame({
        "gender":[enc("gender",v["gender"])],
        "age":[v["age"]],
        "attendance":[v["attendance"]],
        "study_hours":[v["study_hours"]],
        "sleep_hours":[v["sleep_hours"]],
        "social_media_hours":[v["social_media_hours"]],
        "physical_activity":[v["physical_activity"]],
        "mental_health":[v["mental_health"]],
        "internet_quality":[enc("internet_quality",v["internet_quality"])],
        "motivation_level":[enc("motivation_level",v["motivation_level"])],
        "parental_education":[enc("parental_education",v["parental_education"])],
        "family_income":[enc("family_income",v["family_income"])],
        "part_time_job":[enc("part_time_job",v["part_time_job"])],
        "previous_scores":[v["previous_scores"]]
    })
    return round(float(np.clip(model.predict(scaler.transform(X))[0],0,100)),2)

def health(v):
    value = (
        v["attendance"]*.25 +
        min(100,v["study_hours"]/8*100)*.20 +
        max(0,100-abs(v["sleep_hours"]-8)*20)*.15 +
        max(0,100-v["social_media_hours"]*12)*.10 +
        (v["physical_activity"]/5*100)*.10 +
        (v["mental_health"]/5*100)*.10 +
        v["previous_scores"]*.10
    )
    return round(float(np.clip(value,0,100)),1)

def level(score):
    if score >= 85: return "Excellent 🏆"
    if score >= 70: return "Good 👍"
    if score >= 50: return "Average 📚"
    return "Needs Improvement ⚠️"

def tips(v):
    x=[]
    if v["attendance"]<75: x.append("📅 Improve attendance and target at least 75–80%.")
    elif v["attendance"]<85: x.append("📅 Attendance is okay; try to move toward 85%+.")
    if v["study_hours"]<3: x.append("📚 Add one focused 30-minute study session each day.")
    if v["sleep_hours"]<6.5: x.append("😴 Try to maintain around 7–8 hours of sleep.")
    if v["social_media_hours"]>4: x.append("📱 Reduce social-media time during study hours.")
    if v["physical_activity"]<3: x.append("🏃 Add regular physical activity or a daily walk.")
    if v["mental_health"]<3: x.append("🧘 Add breaks and healthy stress-management habits.")
    if v["previous_scores"]<60: x.append("📝 Revise weak topics from previous exams.")
    if v["motivation_level"]=="Low": x.append("🎯 Use small daily goals instead of one large target.")
    return x or ["🌟 Your current habits look balanced. Keep the routine consistent."]

# ============================================================
# REAL LOGIN / SIGNUP
# ============================================================
if not st.session_state.logged_in:

    st.markdown("""
    <div class="hero">
        <h1>🎓 Student Academic Intelligence System</h1>
        <p>Predict performance • Track progress • Improve academically</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login">
        <h1>Welcome 👋</h1>
        <p>Create your student account or login to continue.</p>
    </div>
    """, unsafe_allow_html=True)

    login_tab, signup_tab = st.tabs(["🔐 Login", "📝 Create Account"])

    with login_tab:
        st.markdown(
            "<div class='section-title'>🔐 Student Login</div>",
            unsafe_allow_html=True
        )
        login_email = st.text_input(
            "Email",
            placeholder="Enter your registered email",
            key="login_email"
        )
        login_password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )

        if st.button("🚀 Login to Dashboard", type="primary", use_container_width=True):
            if not login_email.strip() or not login_password.strip():
                st.warning("⚠️ Please enter email and password.")
            else:
                student = login_student(login_email, login_password)
                if student is not None:
                    st.session_state.logged_in = True
                    st.session_state.student_id = student[0]
                    st.session_state.name = student[1]
                    st.session_state.email = student[2]
                    st.session_state.college = student[3] or ""
                    st.session_state.branch = student[4] or "CSE"
                    st.session_state.year = student[5] or ""
                    st.session_state.roll_number = student[6] or ""
                    st.session_state.current_cgpa = float(student[7] or 0)
                    st.session_state.target_cgpa = float(student[8] or 0)
                    st.success(f"🎉 Welcome back, {st.session_state.name}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password.")

    with signup_tab:
        st.markdown(
            "<div class='section-title'>📝 Create Student Account</div>",
            unsafe_allow_html=True
        )
        s1, s2 = st.columns(2)
        with s1:
            signup_name = st.text_input("Full Name", placeholder="Example: Charan", key="signup_name")
            signup_email = st.text_input("Email", placeholder="Example: charan@gmail.com", key="signup_email")
            signup_password = st.text_input("Password", type="password", placeholder="Create a password", key="signup_password")
        with s2:
            signup_college = st.text_input("College", placeholder="Example: Aditya University", key="signup_college")
            signup_branch = st.selectbox("Branch", ["CSE", "ECE", "EEE", "MECH", "CIVIL", "Other"], key="signup_branch")
            signup_year = st.selectbox("Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"], key="signup_year")
            signup_roll = st.text_input("Roll Number", placeholder="Example: 25B11CS625", key="signup_roll")

        st.markdown(
            '<div class="info">🔒 Your account information will be stored in the SAIS database.</div>',
            unsafe_allow_html=True
        )
        if st.button("✨ Create My Account", type="primary", use_container_width=True):
            if not signup_name.strip():
                st.warning("⚠️ Please enter your name.")
            elif not signup_email.strip():
                st.warning("⚠️ Please enter your email.")
            elif not signup_password:
                st.warning("⚠️ Please create a password.")
            elif not signup_college.strip():
                st.warning("⚠️ Please enter your college.")
            elif not signup_roll.strip():
                st.warning("⚠️ Please enter your roll number.")
            else:
                success, message = register_student(
                    signup_name.strip(), signup_email.strip(), signup_password,
                    signup_college.strip(), signup_branch, signup_year, signup_roll.strip()
                )
                if success:
                    st.success("🎉 Account created successfully! Please login using your email and password.")
                else:
                    st.error(f"❌ {message}")

    st.markdown(
        '<div class="footer">🎓 Student Academic Intelligence • Secure Student Login</div>',
        unsafe_allow_html=True
    )
    st.stop()
# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## 🎓 Student AI")

st.sidebar.markdown(
    f"""
    <div class="profile">
        <h3>👤 {st.session_state.name}</h3>
        <p><b>Roll:</b> {st.session_state.roll_number}</p>
        <p><b>Branch:</b> {st.session_state.branch}</p>
        <p><b>Year:</b> {st.session_state.year}</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "👤 My Profile",
        "🤖 My Prediction",
        "🔮 What-If Analysis",
        "📊 Understand My Data",
        "🧠 How AI Decides",
        "🕘 Prediction History"
    ]
)

st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout", use_container_width=True):

    st.session_state.logged_in = False
    st.session_state.student_id = None
    st.session_state.name = ""
    st.session_state.email = ""
    st.session_state.college = ""
    st.session_state.branch = "CSE"
    st.session_state.year = ""
    st.session_state.roll_number = ""

    st.rerun()


# ============================================================
# MY PROFILE
# ============================================================
if page == "👤 My Profile":

    # --------------------------------------------------------
    # PROFILE HERO
    # --------------------------------------------------------
    st.markdown("""
    <div class="hero">
        <h1>👤 My Student Profile</h1>
        <p>Your personal academic identity inside SAIS.</p>
    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>🪪 Personal Information</div>",
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)

    with p1:

        st.markdown(
            f"""
            <div class="profile">
                <h2>👤 {st.session_state.name}</h2>
                <p><b>📧 Email:</b> {st.session_state.email}</p>
                <p><b>🎓 College:</b> {st.session_state.college}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:

        st.markdown(
            f"""
            <div class="profile">
                <h2>📚 Academic Details</h2>
                <p><b>🆔 Roll Number:</b> {st.session_state.roll_number}</p>
                <p><b>💻 Branch:</b> {st.session_state.branch}</p>
                <p><b>📅 Year:</b> {st.session_state.year}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # ACADEMIC GOALS
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>🎯 Academic Goals</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="tip">
            💡 Set your current CGPA and the CGPA you want to achieve.
            SAIS will use these values to track your academic progress.
        </div>
        """,
        unsafe_allow_html=True
    )

    g1, g2 = st.columns(2)

    with g1:

        current_cgpa = st.number_input(
            "📚 Current CGPA",
            min_value=0.0,
            max_value=10.0,
            value=float(
                st.session_state.get(
                    "current_cgpa",
                    0.0
                )
            ),
            step=0.01
        )

    with g2:

        target_cgpa = st.number_input(
            "🎯 Target CGPA",
            min_value=0.0,
            max_value=10.0,
            value=float(
                st.session_state.get(
                    "target_cgpa",
                    0.0
                )
            ),
            step=0.01
        )


    # --------------------------------------------------------
    # SAVE ACADEMIC GOALS
    # --------------------------------------------------------
    if st.button(
        "💾 Save Academic Goals",
        type="primary",
        use_container_width=True
    ):

        if target_cgpa < current_cgpa:

            st.warning(
                "⚠️ Target CGPA should normally be equal to "
                "or higher than your current CGPA."
            )

        else:

            # Update session
            st.session_state.current_cgpa = current_cgpa
            st.session_state.target_cgpa = target_cgpa

            # Update database
            update_academic_goals(
                st.session_state.student_id,
                current_cgpa,
                target_cgpa
            )

            st.success(
                "✅ Academic goals updated successfully!"
            )

            st.rerun()


    # --------------------------------------------------------
    # CURRENT GOAL STATUS
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>📊 Goal Status</div>",
        unsafe_allow_html=True
    )

    current = float(
        st.session_state.get(
            "current_cgpa",
            0.0
        )
    )

    target = float(
        st.session_state.get(
            "target_cgpa",
            0.0
        )
    )

    if target > 0:

        progress = min(
            current / target,
            1.0
        )

        percentage = progress * 100

        st.progress(progress)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "📚 Current",
                f"{current:.2f}"
            )

        with c2:
            st.metric(
                "🎯 Target",
                f"{target:.2f}"
            )

        with c3:
            st.metric(
                "📈 Progress",
                f"{percentage:.0f}%"
            )

    else:

        st.info(
            "🎯 Set your target CGPA above to start "
            "tracking your academic progress."
        )


    # --------------------------------------------------------
    # ACCOUNT INFORMATION
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>🔐 Account Information</div>",
        unsafe_allow_html=True
    )

    st.info(
        "Your student account is connected to the SAIS database. "
        "Predictions and academic progress are linked to this account."
    )

    st.markdown(
        """
        <div class="success">
            ✅ <b>Account Status:</b> Active<br>
            🗄️ <b>Database:</b> Connected<br>
            🎓 <b>Student Identity:</b> Verified
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":

    # --------------------------------------------------------
    # GET LATEST PREDICTION
    # --------------------------------------------------------
    predictions = get_predictions(
        st.session_state.student_id
    )

    latest_prediction = None

    if predictions:
        latest_prediction = predictions[0]

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------
    st.markdown(f"""
    <div class="hero">
      <h1>Welcome back, {st.session_state.name}! 👋</h1>
      <p>Your personalized academic intelligence dashboard.</p>
    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # ACADEMIC OVERVIEW
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>🎓 Academic Overview</div>",
        unsafe_allow_html=True
    )

    current_cgpa = float(
        st.session_state.get("current_cgpa", 0)
    )

    target_cgpa = float(
        st.session_state.get("target_cgpa", 0)
    )


    # Latest prediction values
    if latest_prediction:

        predicted_score = float(
            latest_prediction[1]
        )

        academic_health = float(
            latest_prediction[2]
        )

    else:

        predicted_score = 0
        academic_health = 0


    # --------------------------------------------------------
    # FOUR MAIN CARDS
    # --------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "📚 Current CGPA",
            f"{current_cgpa:.2f}"
        )

    with c2:

        st.metric(
            "🎯 Target CGPA",
            f"{target_cgpa:.2f}"
        )

    with c3:

        if predicted_score > 0:

            st.metric(
                "🤖 Latest Prediction",
                f"{predicted_score:.2f}"
            )

        else:

            st.metric(
                "🤖 Latest Prediction",
                "Not available"
            )

    with c4:

        if academic_health > 0:

            st.metric(
                "❤️ Academic Health",
                f"{academic_health:.0f}%"
            )

        else:

            st.metric(
                "❤️ Academic Health",
                "Not available"
            )


    # --------------------------------------------------------
    # TARGET PROGRESS
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>🎯 Target Progress</div>",
        unsafe_allow_html=True
    )

    if target_cgpa > 0:

        progress = min(
            current_cgpa / target_cgpa,
            1.0
        )

        progress_percentage = progress * 100

        st.progress(progress)

        st.markdown(
            f"""
            <div style="text-align:center;">
                <h3>{progress_percentage:.0f}%</h3>
                <p>
                    Current CGPA <b>{current_cgpa:.2f}</b>
                    out of target <b>{target_cgpa:.2f}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "🎯 Set your Current CGPA and Target CGPA "
            "from My Profile to track your progress."
        )


    # --------------------------------------------------------
    # PERSONAL FOCUS
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>💡 Your Personal Focus</div>",
        unsafe_allow_html=True
    )

    if latest_prediction:

        attendance = float(
            latest_prediction[3]
        )

        study_hours = float(
            latest_prediction[4]
        )

        sleep_hours = float(
            latest_prediction[5]
        )

        social_media_hours = float(
            latest_prediction[6]
        )

        focus_areas = []

        if attendance < 75:
            focus_areas.append(
                "📅 Improve attendance"
            )

        if study_hours < 2:
            focus_areas.append(
                "📚 Increase study consistency"
            )

        if sleep_hours < 6:
            focus_areas.append(
                "😴 Improve sleep routine"
            )

        if social_media_hours > 4:
            focus_areas.append(
                "📱 Reduce social media usage"
            )

        if not focus_areas:

            focus_areas = [
                "🔥 Maintain your current routine",
                "📚 Continue consistent studying",
                "🎯 Keep working toward your target"
            ]

        for focus in focus_areas[:3]:

            st.markdown(
                f"""
                <div class="tip">
                    {focus}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "💡 Complete your first prediction to get "
            "personalized improvement suggestions."
        )


    # --------------------------------------------------------
    # STUDENT PROFILE
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>👤 Student Profile</div>",
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.markdown(
            f"""
            <div class="card">
                <h3>👤 Student</h3>
                <p><b>Name:</b> {st.session_state.name}</p>
                <p><b>Roll Number:</b> {st.session_state.roll_number}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:

        st.markdown(
            f"""
            <div class="card">
                <h3>🎓 Academic</h3>
                <p><b>Branch:</b> {st.session_state.branch}</p>
                <p><b>Year:</b> {st.session_state.year}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p3:

        st.markdown(
            f"""
            <div class="card">
                <h3>🏫 College</h3>
                <p>{st.session_state.college}</p>
                <p>Student Academic Intelligence System</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # WHAT THE SYSTEM DOES
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>🚀 What This System Does</div>",
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    cards = [
        (
            "🤖",
            "1. Predict Performance",
            "The ML model studies several student features together and estimates an expected exam score.",
            "In simple words: What score might this student achieve?"
        ),
        (
            "📊",
            "2. Understand Student Data",
            "Charts show patterns between habits and marks in the collected dataset.",
            "In simple words: Which habits seem connected with better performance?"
        ),
        (
            "🔮",
            "3. Test What-If Scenarios",
            "Change study, attendance, sleep and other habits and compare the new model estimate.",
            "In simple words: What happens if I change my routine?"
        ),
        (
            "💡",
            "4. Give Improvement Tips",
            "The system checks the entered routine and points out areas that can be improved.",
            "In simple words: What can I improve from today?"
        )
    ]

    for box, (icon, title, text, simple) in zip(
        [a, b, c],
        cards
    ):

        with box:

            st.markdown(
                f"""
                <div class="card feature-card">
                    <div class="feature-icon">{icon}</div>
                    <h3>{title}</h3>
                    <p>{text}</p>
                    <p><b>{simple}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # DATASET AT A GLANCE
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>📈 Dataset at a Glance</div>",
        unsafe_allow_html=True
    )

    x1, x2, x3, x4 = st.columns(4)

    x1.metric(
        "👨‍🎓 Students",
        len(df)
    )

    x2.metric(
        "🎯 Avg Exam Score",
        f"{df.exam_score.mean():.1f}"
    )

    x3.metric(
        "📅 Avg Attendance",
        f"{df.attendance.mean():.1f}%"
    )

    x4.metric(
        "📚 Avg Study Hours",
        f"{df.study_hours.mean():.1f}"
    )


    # --------------------------------------------------------
    # RECOMMENDED FLOW
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="tip">
            💡 <b>Recommended flow:</b>
            My Prediction → Understand My Data →
            How AI Decides → Prediction History
        </div>
        """,
        unsafe_allow_html=True
    )
# ============================================================
# PREDICTION
# ============================================================
elif page=="🤖 My Prediction":
    st.markdown("""
    <div class="hero">
      <h1>🤖 My Academic Prediction</h1>
      <p>Answer simple real-life questions. The system converts your answers into the model's required data format automatically.</p>
    </div>
    """,unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>👤 Student Information</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="profile">
            <h3>👤 {st.session_state.name}</h3>
            <p>
                <b>Roll Number:</b> {st.session_state.roll_number}
                &nbsp;&nbsp; | &nbsp;&nbsp;
                <b>Branch:</b> {st.session_state.branch}
                &nbsp;&nbsp; | &nbsp;&nbsp;
                <b>Year:</b> {st.session_state.year}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='section-title'>📚 Academic Routine</div>",unsafe_allow_html=True)
    q1,q2,q3=st.columns(3)
    with q1:
        attendance=st.slider("How regularly do you attend classes? (%)",0,100,85)
    with q2:
        study_hours=st.slider("How many focused hours do you study daily?",0,16,4)
    with q3:
        previous_scores=st.slider("How were your previous exam scores?",0,100,70)

    st.markdown("<div class='section-title'>😴 Lifestyle & Digital Habits</div>",unsafe_allow_html=True)
    q1,q2,q3,q4=st.columns(4)
    with q1:
        sleep_hours=st.slider("How many hours do you sleep daily?",0.0,12.0,7.0,.5)
    with q2:
        social_media_hours=st.slider("How many hours go to social media?",0.0,10.0,2.5,.5)
    with q3:
        physical_activity=st.slider("How active are you physically? (1–5)",1,5,3)
    with q4:
        mental_health=st.slider("How would you rate your mental well-being? (1–5)",1,5,4)

    st.markdown("<div class='section-title'>👤 Student & Learning Environment</div>",unsafe_allow_html=True)
    q1,q2,q3=st.columns(3)
    with q1:
        gender=st.selectbox("How do you identify your gender?",["Male","Female"])
        age=st.number_input("What is your age?",15,35,20)
    with q2:
        internet_quality=st.selectbox("How good is your internet access?",["Poor","Average","Good"])
        motivation_level=st.selectbox("How motivated are you to study?",["Low","Medium","High"])
    with q3:
        parental_education=st.selectbox("What is the highest parental education level?",["High School","College","Postgraduate"])
        family_income=st.selectbox("How would you describe family income?",["Low","Medium","High"])
        part_time_job=st.selectbox("Do you currently have a part-time job?",["No","Yes"])

    values={
        "gender":gender,"age":age,"attendance":attendance,"study_hours":study_hours,
        "sleep_hours":sleep_hours,"social_media_hours":social_media_hours,
        "physical_activity":physical_activity,"mental_health":mental_health,
        "internet_quality":internet_quality,"motivation_level":motivation_level,
        "parental_education":parental_education,"family_income":family_income,
        "part_time_job":part_time_job,"previous_scores":previous_scores
    }

    if st.button("🚀 Predict My Academic Performance",type="primary",use_container_width=True):
        try:
            score=predict(values)
            h=health(values)
            label=level(score)
                        # Save prediction permanently to database
            save_prediction(
                st.session_state.student_id,
                score,
                h,
                values["attendance"],
                values["study_hours"],
                values["sleep_hours"],
                values["social_media_hours"],
                values["physical_activity"],
                values["mental_health"],
                values["previous_scores"]
            )
            now=datetime.now().strftime("%d-%m-%Y %I:%M %p")
            st.session_state.history.insert(0,{"Time":now,"Student":st.session_state.name,"Predicted Score":score,"Academic Health":h,"Performance":label})
            st.session_state.history=st.session_state.history[:10]
            st.session_state.last_values=values.copy()
            st.session_state.last_score=score
            st.session_state.last_health=h

            st.success("🎉 Prediction completed successfully!")

            # ------------------------------------------------
            # RESULT SUMMARY
            # ------------------------------------------------
            st.markdown(
                "<div class='section-title'>🎯 Your Prediction Results</div>",
                unsafe_allow_html=True
            )

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "📚 Predicted Exam Score",
                f"{score:.1f}/100"
            )

            r2.metric(
                "❤️ Academic Health",
                f"{h}%"
            )

            r3.metric(
                "🏆 Performance",
                label
            )

            # ------------------------------------------------
            # VISUAL SCORE GAUGES
            # ------------------------------------------------
            g1, g2 = st.columns(2)

            with g1:

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=score,
                        number={
                            "suffix": "/100"
                        },
                        title={
                            "text": "📚 Expected Exam Score"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100]
                            },
                            "bar": {
                                "thickness": 0.25
                            }
                        }
                    )
                )

                fig.update_layout(
                    height=320
                )

                st.plotly_chart(
                    dark_fig(fig),
                    use_container_width=True
                )

            with g2:

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=h,
                        number={
                            "suffix": "%"
                        },
                        title={
                            "text": "❤️ Academic Health"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100]
                            },
                            "bar": {
                                "thickness": 0.25
                            }
                        }
                    )
                )

                fig.update_layout(
                    height=320
                )

                st.plotly_chart(
                    dark_fig(fig),
                    use_container_width=True
                )

            # ------------------------------------------------
            # RESULT EXPLANATION
            # ------------------------------------------------
            st.markdown(
                "<div class='section-title'>🧠 What Does My Result Mean?</div>",
                unsafe_allow_html=True
            )

            if score >= 85:

                msg = f"""
                🏆 <b>Strong predicted performance.</b><br>
                The model estimates an exam score of approximately
                <b>{score:.1f}/100</b>.
                Keep your current positive academic routine consistent.
                """

                css = "success"

            elif score >= 70:

                msg = f"""
                👍 <b>Good predicted performance.</b><br>
                The model estimates an exam score of approximately
                <b>{score:.1f}/100</b>.
                Improving a few daily habits may help strengthen your routine.
                """

                css = "info"

            else:

                msg = f"""
                ⚠️ <b>There is room for improvement.</b><br>
                The model estimates an exam score of approximately
                <b>{score:.1f}/100</b>.
                Focus on the personalised improvement suggestions below.
                """

                css = "warning"

            st.markdown(
                f'<div class="{css}">{msg}</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # CURRENT HABIT SUMMARY
            # ------------------------------------------------
            st.markdown(
                "<div class='section-title'>📊 Your Current Academic Routine</div>",
                unsafe_allow_html=True
            )

            h1, h2, h3, h4 = st.columns(4)

            h1.metric(
                "📅 Attendance",
                f"{values['attendance']}%"
            )

            h2.metric(
                "📚 Study",
                f"{values['study_hours']} hrs/day"
            )

            h3.metric(
                "😴 Sleep",
                f"{values['sleep_hours']} hrs/day"
            )

            h4.metric(
                "📱 Social Media",
                f"{values['social_media_hours']} hrs/day"
            )

            # ------------------------------------------------
            # PERSONAL IMPROVEMENT PLAN
            # ------------------------------------------------
            st.markdown(
                "<div class='section-title'>💡 Your Personal Improvement Plan</div>",
                unsafe_allow_html=True
            )

            improvement_tips = tips(values)

            if improvement_tips:

                for i, t in enumerate(
                    improvement_tips,
                    start=1
                ):

                    st.markdown(
                        f"""
                        <div class="card">
                            <h3>💡 Step {i}</h3>
                            <p>{t}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ------------------------------------------------
            # NEXT STEP
            # ------------------------------------------------
            st.markdown(
                """
                <div class="card">
                    <h3>🔮 Want to improve the result?</h3>
                    <p>
                    Go to <b>What-If Analysis</b> and experiment with
                    attendance, study time, sleep, social-media usage,
                    physical activity and motivation.
                    </p>
                    <p>
                    You can compare your current prediction with a
                    different routine without changing your original result.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # DOWNLOAD REPORT
            # ------------------------------------------------
            report = pd.DataFrame([
                {
                    **values,
                    "predicted_exam_score": score,
                    "academic_health": h,
                    "performance": label,
                    "prediction_time": now
                }
            ])

            st.download_button(
                "📥 Download Prediction Report",
                data=report.to_csv(index=False),
                file_name="student_prediction_report.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_prediction_report"
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )
# ============================================================
# WHAT-IF ANALYSIS
# ============================================================
elif page=="🔮 What-If Analysis":
    st.markdown("""
    <div class="hero">
      <h1>🔮 What-If Analysis Lab</h1>
      <p>Change a few habits and instantly compare the new model estimate with your current baseline.</p>
    </div>
    """,unsafe_allow_html=True)

    if st.session_state.last_values is None:
        st.markdown('<div class="info">ℹ️ First create a prediction in <b>My Prediction</b>. Your latest answers will automatically appear here.</div>',unsafe_allow_html=True)
        st.stop()

    base=st.session_state.last_values.copy()
    base_score=st.session_state.last_score

    st.markdown("<div class='section-title'>🎯 Current Baseline</div>",unsafe_allow_html=True)
    b1,b2,b3=st.columns(3)
    b1.metric("Current Predicted Score",base_score)
    b2.metric("Attendance",f"{base['attendance']}%")
    b3.metric("Study Time",f"{base['study_hours']} hrs/day")

    st.markdown("<div class='section-title'>🧪 Try a Different Scenario</div>",unsafe_allow_html=True)
    st.caption("Only the values below change. Every other answer stays exactly the same.")

    c1,c2,c3=st.columns(3)
    with c1:
        wf_attendance=st.slider("If attendance becomes (%)",0,100,int(base["attendance"]))
        wf_study=st.slider("If study time becomes (hours/day)",0,16,int(base["study_hours"]))
    with c2:
        wf_sleep=st.slider("If sleep becomes (hours/day)",0.0,12.0,float(base["sleep_hours"]),.5)
        wf_social=st.slider("If social media becomes (hours/day)",0.0,10.0,float(base["social_media_hours"]),.5)
    with c3:
        wf_activity=st.slider("If physical activity becomes (1–5)",1,5,int(base["physical_activity"]))
        wf_motivation=st.selectbox("If motivation changes to",["Low","Medium","High"],
                                    index=["Low","Medium","High"].index(base["motivation_level"]))

    scenario=base.copy()
    scenario.update({
        "attendance":wf_attendance,
        "study_hours":wf_study,
        "sleep_hours":wf_sleep,
        "social_media_hours":wf_social,
        "physical_activity":wf_activity,
        "motivation_level":wf_motivation
    })

    try:
        scenario_score=predict(scenario)
        delta=round(scenario_score-base_score,2)
        scenario_health=health(scenario)

                # ----------------------------------------------------
        # BEFORE → AFTER RESULT
        # ----------------------------------------------------
        st.markdown(
            "<div class='section-title'>📊 Before → After Comparison</div>",
            unsafe_allow_html=True
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "📚 Current Score",
            f"{base_score:.1f}"
        )

        r2.metric(
            "🔮 What-If Score",
            f"{scenario_score:.1f}",
            f"{delta:+.1f}"
        )

        r3.metric(
            "❤️ Scenario Health",
            f"{scenario_health:.1f}%"
        )

        # ----------------------------------------------------
        # CHANGED HABITS
        # ----------------------------------------------------
        changed_habits = []

        if wf_attendance != base["attendance"]:
            changed_habits.append(
                f"📅 Attendance: {base['attendance']}% → {wf_attendance}%"
            )

        if wf_study != base["study_hours"]:
            changed_habits.append(
                f"📚 Study: {base['study_hours']} hrs → {wf_study} hrs"
            )

        if wf_sleep != base["sleep_hours"]:
            changed_habits.append(
                f"😴 Sleep: {base['sleep_hours']} hrs → {wf_sleep} hrs"
            )

        if wf_social != base["social_media_hours"]:
            changed_habits.append(
                f"📱 Social Media: {base['social_media_hours']} hrs → {wf_social} hrs"
            )

        if wf_activity != base["physical_activity"]:
            changed_habits.append(
                f"🏃 Physical Activity: {base['physical_activity']}/5 → {wf_activity}/5"
            )

        if wf_motivation != base["motivation_level"]:
            changed_habits.append(
                f"🔥 Motivation: {base['motivation_level']} → {wf_motivation}"
            )

        if changed_habits:

            st.markdown(
                "<div class='section-title'>🧪 Changes You Made</div>",
                unsafe_allow_html=True
            )

            for change in changed_habits:
                st.info(change)

        else:

            st.markdown(
                """
                <div class="info">
                    ➖ <b>No habits were changed.</b><br>
                    Move one or more sliders to test a different scenario.
                </div>
                """,
                unsafe_allow_html=True
            )

        compare=pd.DataFrame({
            "Version":["Current routine","What-if routine"],
            "Predicted Score":[base_score,scenario_score]
        })
        
        fig=px.bar(compare,x="Version",y="Predicted Score",text="Predicted Score",
                   title="Current vs What-If Prediction",range_y=[0,100])
        fig.update_traces(texttemplate="%{text:.1f}",textposition="outside")
        st.plotly_chart(dark_fig(fig,390),use_container_width=True)

        if delta>0.5:
            st.markdown(f'<div class="success">🚀 <b>Positive scenario:</b> the model estimate is <b>{delta:.2f}</b> points higher.</div>',unsafe_allow_html=True)
        elif delta<-0.5:
            st.markdown(f'<div class="warning">⚠️ <b>Trade-off:</b> the model estimate is <b>{abs(delta):.2f}</b> points lower.</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="info">➖ <b>Small change:</b> the model estimate is almost unchanged.</div>',unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
          <h3>🧠 How to understand this</h3>
          <p><b>Baseline</b> = prediction using your current answers.</p>
          <p><b>What-If</b> = prediction after changing selected habits.</p>
          <p><b>Delta</b> = What-If score − Baseline score.</p>
          <p>This is a scenario experiment, not proof that changing one habit will cause the exact score change in real life.</p>
        </div>
        """,unsafe_allow_html=True)

        st.markdown("<div class='section-title'>💡 Scenario Improvement Tips</div>",unsafe_allow_html=True)
        for t in tips(scenario):
            st.info(t)
    except Exception as e:
        st.error(f"What-If analysis could not be calculated: {e}")

# ============================================================
# UNDERSTAND DATA
# ============================================================
elif page=="📊 Understand My Data":

    st.markdown("""
    <div class="hero">
        <h1>📊 Understand My Data</h1>
        <p>Explore real student patterns and discover how different habits are connected with academic performance.</p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>🔎 Explore the Student Dataset</div>",
        unsafe_allow_html=True
    )

    f1, f2, f3, f4 = st.columns(4)

    genders = sorted(df.gender.dropna().unique())
    motivations = sorted(df.motivation_level.dropna().unique())
    incomes = sorted(df.family_income.dropna().unique())
    internet = sorted(df.internet_quality.dropna().unique())

    with f1:
        gf = st.multiselect(
            "👤 Gender",
            genders,
            default=genders
        )

    with f2:
        mf = st.multiselect(
            "🎯 Motivation",
            motivations,
            default=motivations
        )

    with f3:
        inf = st.multiselect(
            "🏠 Family Income",
            incomes,
            default=incomes
        )

    with f4:
        itf = st.multiselect(
            "🌐 Internet Quality",
            internet,
            default=internet
        )

    d = df[
        df.gender.isin(gf)
        & df.motivation_level.isin(mf)
        & df.family_income.isin(inf)
        & df.internet_quality.isin(itf)
    ].copy()

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SUMMARY CARDS
    # --------------------------------------------------------
    total_students = len(d)

    if total_students == 0:
        st.warning("⚠️ No students match the selected filters.")
        st.stop()

    avg_score = pd.to_numeric(
        d["exam_score"],
        errors="coerce"
    ).mean()

    avg_attendance = pd.to_numeric(
        d["attendance"],
        errors="coerce"
    ).mean()

    avg_study = pd.to_numeric(
        d["study_hours"],
        errors="coerce"
    ).mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👨‍🎓 Students",
        total_students
    )

    c2.metric(
        "📈 Average Exam Score",
        f"{avg_score:.1f}"
    )

    c3.metric(
        "📅 Average Attendance",
        f"{avg_attendance:.1f}%"
    )

    c4.metric(
        "📚 Average Study",
        f"{avg_study:.1f} hrs"
    )

    # --------------------------------------------------------
    # DATASET SNAPSHOT
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>🥧 Dataset Snapshot</div>",
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)

    with p1:

        gcounts = d["gender"].value_counts().reset_index()
        gcounts.columns = ["gender", "count"]

        fig = px.pie(
            gcounts,
            names="gender",
            values="count",
            hole=0.50,
            title="👤 Student Distribution"
        )

        st.plotly_chart(
            dark_fig(fig, 380),
            use_container_width=True
        )

    with p2:

        mcounts = d["motivation_level"].value_counts().reset_index()
        mcounts.columns = ["motivation", "count"]

        fig = px.pie(
            mcounts,
            names="motivation",
            values="count",
            hole=0.50,
            title="🎯 Motivation Distribution"
        )

        st.plotly_chart(
            dark_fig(fig, 380),
            use_container_width=True
        )

    st.markdown(
        """
        <div class="info">
        💡 <b>What does this show?</b><br>
        These charts give a quick overview of the students included in the
        currently selected dataset view.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # HABITS VS PERFORMANCE
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>📚 Habits vs Academic Performance</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <h3>🔍 What are we looking for?</h3>
            <p>
            We compare everyday student habits with exam scores to identify
            patterns in the dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    charts = [

        (
            "attendance",
            "📅 Attendance vs Exam Score",
            "Do students with higher attendance tend to have different exam scores?"
        ),

        (
            "study_hours",
            "📚 Study Hours vs Exam Score",
            "How are daily study hours associated with exam performance?"
        ),

        (
            "sleep_hours",
            "😴 Sleep Hours vs Exam Score",
            "How does daily sleep duration relate to exam scores?"
        ),

        (
            "social_media_hours",
            "📱 Social Media vs Exam Score",
            "How does social-media usage vary with exam scores?"
        ),

        (
            "previous_scores",
            "📝 Previous Scores vs Exam Score",
            "How strongly are previous scores associated with current performance?"
        )

    ]

    for i in range(0, len(charts), 2):

        cols = st.columns(2)

        for col, (feature, title, question) in zip(
            cols,
            charts[i:i+2]
        ):

            with col:

                if (
                    feature not in d.columns
                    or "exam_score" not in d.columns
                ):
                    st.warning(
                        f"Required column '{feature}' is not available."
                    )
                    continue

                chart_d = d[
                    [feature, "exam_score"]
                ].copy()

                chart_d[feature] = pd.to_numeric(
                    chart_d[feature],
                    errors="coerce"
                )

                chart_d["exam_score"] = pd.to_numeric(
                    chart_d["exam_score"],
                    errors="coerce"
                )

                chart_d = chart_d.dropna()

                if chart_d.empty:
                    st.info(
                        "No valid numeric data available for this chart."
                    )
                    continue

                fig = px.scatter(
                    chart_d,
                    x=feature,
                    y="exam_score",
                    title=title,
                    labels={
                        feature: feature.replace(
                            "_", " "
                        ).title(),
                        "exam_score": "Exam Score"
                    }
                )

                st.plotly_chart(
                    dark_fig(fig, 390),
                    use_container_width=True
                )

                st.markdown(
                    f"""
                    <div class="info">
                        <b>💭 Student Question:</b><br>
                        {question}
                        <br><br>
                        <b>⚠️ Remember:</b>
                        A pattern in the dataset does not prove that
                        one factor directly causes better or worse marks.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # --------------------------------------------------------
    # GROUP COMPARISONS
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>👥 Compare Student Groups</div>",
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:

        fig = px.box(
            d,
            x="gender",
            y="exam_score",
            title="👤 Gender vs Exam Score"
        )

        st.plotly_chart(
            dark_fig(fig, 390),
            use_container_width=True
        )

    with b:

        fig = px.box(
            d,
            x="motivation_level",
            y="exam_score",
            title="🎯 Motivation vs Exam Score"
        )

        st.plotly_chart(
            dark_fig(fig, 390),
            use_container_width=True
        )

    a, b = st.columns(2)

    with a:

        fig = px.box(
            d,
            x="family_income",
            y="exam_score",
            title="🏠 Family Income vs Exam Score"
        )

        st.plotly_chart(
            dark_fig(fig, 390),
            use_container_width=True
        )

    with b:

        fig = px.box(
            d,
            x="internet_quality",
            y="exam_score",
            title="🌐 Internet Quality vs Exam Score"
        )

        st.plotly_chart(
            dark_fig(fig, 390),
            use_container_width=True
        )

    st.markdown(
        """
        <div class="info">
        📖 <b>How to read these charts:</b>
        The box plots help compare the typical exam score and the spread
        of scores between different student groups.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CORRELATION HEATMAP
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>🔥 Relationship Between Numeric Factors</div>",
        unsafe_allow_html=True
    )

    numeric_d = d.select_dtypes(
        include=np.number
    )

    if numeric_d.shape[1] >= 2:

        corr = numeric_d.corr()

        heat = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="🔥 Correlation Heatmap"
        )

        st.plotly_chart(
            dark_fig(heat, 450),
            use_container_width=True
        )

        st.markdown(
            """
            <div class="tip">
                🔥 <b>How to understand the heatmap:</b><br><br>

                <b>+1</b> → strong positive relationship<br>
                <b>0</b> → weak or no linear relationship<br>
                <b>-1</b> → strong negative relationship<br><br>

                ⚠️ Correlation shows that two variables move together.
                It does <b>not</b> prove that one variable causes the other.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "Not enough numeric columns are available for the heatmap."
        )

    # --------------------------------------------------------
    # FINAL TAKEAWAY
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>💡 What Should a Student Take Away?</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

            <h3>🎓 The goal is not to compare students.</h3>

            <p>
            The purpose of this analysis is to understand patterns in
            student behaviour and academic performance.
            </p>

            <p>
            📅 Attendance, 📚 study habits, 😴 sleep,
            📱 digital usage and 📝 previous performance
            can all be explored together.
            </p>

            <p>
            These insights help the system provide more meaningful
            academic guidance to individual students.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )
# ============================================================
# HOW AI DECIDES
# ============================================================
elif page=="🧠 How AI Decides":

    st.markdown("""
    <div class="hero">
      <h1>🧠 How Does the AI Decide?</h1>
      <p>Understand how your answers are converted into an academic prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # SIMPLE AI EXPLANATION
    # --------------------------------------------------------
    st.markdown("""
    <div class="card">
      <h3>🤖 In Simple Words</h3>

      <p>
      You answer questions about your academic routine, lifestyle,
      previous performance and learning environment.
      </p>

      <p>
      The trained machine learning model studies patterns from the
      training data and uses those patterns to estimate your expected
      exam score.
      </p>

      <p>
      <b>Your answers → Learned patterns → Prediction</b>
      </p>

      <p>
      ⚠️ The prediction is an estimate, not a guarantee of your actual
      future marks.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # HOW THE PROCESS WORKS
    # --------------------------------------------------------
    st.markdown(
        "<div class='section-title'>🔄 How Your Prediction Is Created</div>",
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)

    with a:
        st.markdown("""
        <div class="card">
            <h3>1️⃣ You Answer</h3>
            <p>
            Enter your attendance, study time, sleep,
            previous scores and other details.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="card">
            <h3>2️⃣ Data Processing</h3>
            <p>
            Your answers are converted into the format
            required by the trained model.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="card">
            <h3>3️⃣ AI Prediction</h3>
            <p>
            The machine learning model uses patterns
            learned from training data.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with d:
        st.markdown("""
        <div class="card">
            <h3>4️⃣ Result</h3>
            <p>
            The system gives an estimated exam score
            and academic health information.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------
    try:

        imp = pd.read_csv("feature_importance.csv")

        fc = next(
            (
                c for c in
                ["Feature", "feature", "feature_name"]
                if c in imp.columns
            ),
            imp.columns[0]
        )

        ic = next(
            (
                c for c in
                ["Importance", "importance", "importance_score"]
                if c in imp.columns
            ),
            imp.columns[1]
        )

        imp = imp.rename(
            columns={
                fc: "Feature",
                ic: "Importance"
            }
        )

        imp = imp.sort_values(
            "Importance",
            ascending=False
        )

        names = {
            "attendance": "📅 Attendance",
            "study_hours": "📚 Study Hours",
            "sleep_hours": "😴 Sleep Hours",
            "social_media_hours": "📱 Social Media",
            "previous_scores": "📝 Previous Scores",
            "physical_activity": "🏃 Physical Activity",
            "mental_health": "🧠 Mental Health",
            "motivation_level": "🎯 Motivation",
            "internet_quality": "🌐 Internet Quality",
            "family_income": "🏠 Family Income",
            "parental_education": "👨‍👩‍👦 Parental Education",
            "part_time_job": "💼 Part-Time Job",
            "age": "🎂 Age",
            "gender": "👤 Gender"
        }

        # ----------------------------------------------------
        # MODEL FACTORS
        # ----------------------------------------------------
        st.markdown(
            "<div class='section-title'>🎯 What Information Does the Model Use?</div>",
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="info">
        The model considers multiple student-related features together.
        Feature importance shows which inputs the trained model relied on
        more heavily for its predictions.
        </div>
        """, unsafe_allow_html=True)

        chart = px.bar(
            imp.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Model Feature Importance"
        )

        st.plotly_chart(
            dark_fig(chart, 500),
            use_container_width=True
        )

        st.markdown("""
        <div class="info">
        💡 <b>Important:</b> A higher feature importance means the model
        relied more on that feature in this trained model.
        It does <b>not</b> prove that the feature directly causes
        higher or lower marks.
        </div>
        """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # TOP FACTORS
        # ----------------------------------------------------
        st.markdown(
            "<div class='section-title'>🏆 Important Factors in This Model</div>",
            unsafe_allow_html=True
        )

        top_factors = imp.head(5)

        for rank, (_, row) in enumerate(
            top_factors.iterrows(),
            start=1
        ):

            nm = names.get(
                str(row["Feature"]),
                str(row["Feature"]).replace("_", " ").title()
            )

            importance_value = float(
                row["Importance"]
            )

            st.markdown(
                f"""
                <div class="card">
                    <h3>#{rank} &nbsp; {nm}</h3>
                    <p>
                        Model importance:
                        <b>{importance_value:.3f}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:

        st.warning(
            f"Feature importance could not be displayed: {e}"
        )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------
    try:

        res = pd.read_csv(
            "model_results.csv"
        )

        st.markdown(
            "<div class='section-title'>📈 How Well Did the Model Perform?</div>",
            unsafe_allow_html=True
        )

        st.dataframe(
            res,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        <div class="card">

          <h3>📖 Understanding the Numbers</h3>

          <p>
          <b>R²:</b>
          Shows how much of the variation in the target values
          is explained by the model.
          </p>

          <p>
          <b>MAE:</b>
          Shows the average absolute difference between
          predicted and actual values.
          Lower values generally indicate smaller errors.
          </p>

          <p>
          <b>RMSE:</b>
          Measures prediction error while giving more weight
          to larger errors.
          </p>

        </div>
        """, unsafe_allow_html=True)

    except Exception as e:

        st.info(
            f"Model results are not available: {e}"
        )

    # --------------------------------------------------------
    # FINAL NOTE
    # --------------------------------------------------------
    st.markdown("""
    <div class="success">
        🎓 <b>Remember:</b> AI does not decide your future.
        It analyzes patterns in data and gives an estimate.
        Your habits, effort, learning strategy and real-world
        circumstances can change your actual academic outcome.
    </div>
    """, unsafe_allow_html=True) 

# ============================================================
# PERMANENT PREDICTION HISTORY
# ============================================================
else:

    st.markdown("""
    <div class="hero">
        <h1>🕘 My Academic Journey</h1>
        <p>Track your predictions, monitor your academic health, and see how your progress changes over time.</p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # GET STUDENT PREDICTIONS
    # --------------------------------------------------------
    predictions = get_predictions(
        st.session_state.student_id
    )

    # --------------------------------------------------------
    # NO HISTORY
    # --------------------------------------------------------
    if not predictions:

        st.markdown("""
        <div class="info">
            ℹ️ <b>No predictions yet.</b><br><br>
            Go to <b>My Prediction</b> and create your first academic prediction.
        </div>
        """, unsafe_allow_html=True)

    else:

        # ----------------------------------------------------
        # DATABASE → DATAFRAME
        # ----------------------------------------------------
        history_df = pd.DataFrame(
            predictions,
            columns=[
                "ID",
                "Predicted Score",
                "Academic Health",
                "Attendance",
                "Study Hours",
                "Sleep Hours",
                "Social Media Hours",
                "Physical Activity",
                "Mental Health",
                "Previous Scores",
                "Created At"
            ]
        )

        # ----------------------------------------------------
        # BASIC VALUES
        # ----------------------------------------------------
        latest_score = float(
            history_df.iloc[0]["Predicted Score"]
        )

        latest_health = float(
            history_df.iloc[0]["Academic Health"]
        )

        total_predictions = len(history_df)

        if total_predictions > 1:

            first_score = float(
                history_df.iloc[-1]["Predicted Score"]
            )

            score_change = round(
                latest_score - first_score,
                2
            )

        else:
            score_change = 0

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------
        st.markdown(
            "<div class='section-title'>📊 Your Journey Summary</div>",
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "🔢 Predictions",
            total_predictions
        )

        c2.metric(
            "🎯 Latest Score",
            f"{latest_score:.1f}"
        )

        c3.metric(
            "❤️ Academic Health",
            f"{latest_health:.1f}%"
        )

        c4.metric(
            "📈 Overall Change",
            f"{score_change:+.1f}"
        )

        # ----------------------------------------------------
        # PROGRESS MESSAGE
        # ----------------------------------------------------
        if total_predictions == 1:

            st.markdown("""
            <div class="info">
                🆕 <b>This is your first recorded prediction.</b><br>
                Keep using the system regularly to build your academic journey.
            </div>
            """, unsafe_allow_html=True)

        elif score_change > 0:

            st.markdown(
                f"""
                <div class="success">
                    📈 <b>Your prediction trend is moving upward.</b><br>
                    Your latest predicted score is
                    <b>{score_change:.1f} points higher</b>
                    than your earliest recorded prediction.
                </div>
                """,
                unsafe_allow_html=True
            )

        elif score_change < 0:

            st.markdown(
                f"""
                <div class="warning">
                    ⚠️ <b>Your latest prediction is lower than your earliest one.</b><br>
                    The difference is
                    <b>{abs(score_change):.1f} points</b>.
                    Check your current routine in
                    <b>My Prediction</b> and
                    <b>What-If Analysis</b>.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown("""
            <div class="info">
                ➖ <b>Your predicted score has remained relatively stable.</b><br>
                Keep tracking your routine to identify meaningful changes.
            </div>
            """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # PREPARE CHART DATA
        # ----------------------------------------------------
        chart_df = history_df.copy()

        chart_df["Created At"] = pd.to_datetime(
            chart_df["Created At"]
        )

        # Oldest → newest
        chart_df = chart_df.sort_values(
            "Created At"
        )

        # ----------------------------------------------------
        # PREDICTION TREND
        # ----------------------------------------------------
        st.markdown(
            "<div class='section-title'>📈 Prediction Trend</div>",
            unsafe_allow_html=True
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=chart_df["Created At"],
                y=chart_df["Predicted Score"],
                mode="lines+markers",
                name="Predicted Score",
                line=dict(width=3),
                marker=dict(size=9),
                hovertemplate=
                    "<b>Predicted Score:</b> %{y:.1f}<br>" +
                    "<b>Date:</b> %{x}<extra></extra>"
            )
        )

        fig.update_layout(
            title="How Your Predicted Performance Changes",
            xaxis_title="Prediction Date",
            yaxis_title="Predicted Score",
            yaxis=dict(range=[0, 100]),
            height=430
        )

        st.plotly_chart(
            dark_fig(fig, 430),
            use_container_width=True
        )

        # ----------------------------------------------------
        # ACADEMIC HEALTH TREND
        # ----------------------------------------------------
        st.markdown(
            "<div class='section-title'>❤️ Academic Health Trend</div>",
            unsafe_allow_html=True
        )

        health_chart = go.Figure()

        health_chart.add_trace(
            go.Scatter(
                x=chart_df["Created At"],
                y=chart_df["Academic Health"],
                mode="lines+markers",
                name="Academic Health",
                line=dict(width=3),
                marker=dict(size=9),
                hovertemplate=
                    "<b>Academic Health:</b> %{y:.1f}%<br>" +
                    "<b>Date:</b> %{x}<extra></extra>"
            )
        )

        health_chart.update_layout(
            title="How Your Academic Health Changes",
            xaxis_title="Prediction Date",
            yaxis_title="Academic Health (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )

        st.plotly_chart(
            dark_fig(health_chart, 400),
            use_container_width=True
        )

        # ----------------------------------------------------
        # LATEST ROUTINE
        # ----------------------------------------------------
        latest = history_df.iloc[0]

        st.markdown(
            "<div class='section-title'>🧠 Your Latest Recorded Routine</div>",
            unsafe_allow_html=True
        )

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "📅 Attendance",
            f"{float(latest['Attendance']):.1f}%"
        )

        r2.metric(
            "📚 Study",
            f"{float(latest['Study Hours']):.1f} hrs"
        )

        r3.metric(
            "😴 Sleep",
            f"{float(latest['Sleep Hours']):.1f} hrs"
        )

        r4.metric(
            "📱 Social Media",
            f"{float(latest['Social Media Hours']):.1f} hrs"
        )

        # ----------------------------------------------------
        # HISTORY TABLE
        # ----------------------------------------------------
        st.markdown(
            "<div class='section-title'>📋 Prediction History</div>",
            unsafe_allow_html=True
        )

        display_df = history_df.copy()

        display_df["Created At"] = pd.to_datetime(
            display_df["Created At"]
        ).dt.strftime(
            "%d-%m-%Y %I:%M %p"
        )

        display_df = display_df[
            [
                "Created At",
                "Predicted Score",
                "Academic Health",
                "Attendance",
                "Study Hours",
                "Sleep Hours",
                "Social Media Hours",
                "Previous Scores"
            ]
        ]

        display_df.columns = [
            "Date",
            "Predicted Score",
            "Academic Health",
            "Attendance %",
            "Study Hours",
            "Sleep Hours",
            "Social Media Hours",
            "Previous Scores"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DOWNLOAD HISTORY
        # ----------------------------------------------------
        csv_data = display_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download My Prediction History",
            data=history_df.to_csv(index=False),
            file_name="my_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_prediction_history"
        )

        # ----------------------------------------------------
        # SIMPLE INSIGHT
        # ----------------------------------------------------
        if total_predictions >= 2:

            if score_change > 0:
                st.markdown(
                    f"""
                    <div class="success">
                        📈 <b>Great progress!</b><br>
                        Your latest predicted score is
                        <b>{score_change:.1f} points higher</b>
                        than your earliest recorded prediction.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif score_change < 0:
                st.markdown(
                    f"""
                    <div class="warning">
                        ⚠️ Your latest predicted score is
                        <b>{abs(score_change):.1f} points lower</b>
                        than your earliest recorded prediction.
                        Check your current routine in
                        <b>My Prediction</b> and
                        <b>What-If Analysis</b>.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    """
                    <div class="info">
                        ➖ Your predicted score is currently
                        close to your earliest recorded prediction.
                        Keep tracking your routine to identify trends.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")
st.markdown("<div class='footer'>🎓 Student Academic Intelligence • Python • Pandas • Scikit-Learn • Streamlit • Plotly</div>",unsafe_allow_html=True)
