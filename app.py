
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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
    "name": "",
    "year": "",
    "history": [],
    "last_values": None,
    "last_score": None,
    "last_health": None,
    "roll_number": "",
    "branch": "CSE"
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
# LANDING / DEMO LOGIN
# ============================================================
if not st.session_state.logged_in:
    st.markdown("""
    <div class="hero">
      <h1>🎓 Student Academic Intelligence System</h1>
      <p>Predict performance • Understand habits • Get improvement guidance</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login">
      <h1>Welcome 👋</h1>
      <p>Enter a student name and year to create a project-demo profile. This is not a real password/login system.</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1: name=st.text_input("Student Name",placeholder="Example: Rahul")
    with c2: year=st.text_input("Class / Year",placeholder="Example: 2nd Year CSE")

    a,b,c=st.columns(3)
    with a:
        st.markdown('<div class="card feature-card"><div class="feature-icon">🤖</div><h3>AI Prediction</h3><p>Estimate expected exam performance from student inputs.</p></div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card feature-card"><div class="feature-icon">📊</div><h3>Easy Analytics</h3><p>Understand attendance, study, sleep and other factors through simple charts.</p></div>',unsafe_allow_html=True)
    with c:
        st.markdown('<div class="card feature-card"><div class="feature-icon">💡</div><h3>Smart Guidance</h3><p>Get practical suggestions based on the student routine.</p></div>',unsafe_allow_html=True)

    if st.button("🚀 Enter Student Dashboard",type="primary",use_container_width=True):
        if not name.strip():
            st.warning("Please enter the student name.")
        else:
            st.session_state.name=name.strip()
            st.session_state.year=year.strip() or "Student"
            st.session_state.logged_in=True
            st.rerun()
    st.markdown('<div class="info">🔒 <b>Demo note:</b> This is a project-style profile screen; it does not store or verify passwords.</div>',unsafe_allow_html=True)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## 🎓 Student AI")
st.sidebar.caption(f"Welcome, {st.session_state.name}")
st.sidebar.markdown("---")
page=st.sidebar.radio("Navigation",[
    "🏠 Home","🤖 My Prediction","📊 Understand My Data",
    "🧠 How AI Decides","🕘 Prediction History"
])
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Back to Welcome"):
    st.session_state.logged_in=False
    st.rerun()

# ============================================================
# HOME
# ============================================================
if page=="🏠 Home":
    st.markdown(f"""
    <div class="hero">
      <h1>Welcome back, {st.session_state.name}! 👋</h1>
      <p>Your academic intelligence dashboard is ready.</p>
    </div>
    """,unsafe_allow_html=True)

    st.markdown(f"""
    <div class="profile">
      <h2>👤 Student Profile</h2>
      <p><b>Name:</b> {st.session_state.name}</p>
      <p><b>Class / Year:</b> {st.session_state.year}</p>
      <p><b>System:</b> Student Academic Intelligence System</p>
    </div>
    """,unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🚀 What This System Does</div>",unsafe_allow_html=True)
    a,b,c=st.columns(3)
    cards=[
        ("🤖","1. Predict Performance","The ML model studies several student features together and estimates an expected exam score.","In simple words: What score might this student achieve?"),
        ("📊","2. Understand Student Data","Charts show patterns between habits and marks in the collected dataset.","In simple words: Which habits seem connected with better performance?"),
        ("🔮","3. Test What-If Scenarios","Change study, attendance, sleep and other habits and compare the new model estimate.","In simple words: What happens if I change my routine?"),
        ("💡","4. Give Improvement Tips","The system checks the entered routine and points out areas that can be improved.","In simple words: What can I improve from today?")
    ]
    for box,(icon,title,text,simple) in zip([a,b,c],cards):
        with box:
            st.markdown(f'<div class="card feature-card"><div class="feature-icon">{icon}</div><h3>{title}</h3><p>{text}</p><p><b>{simple}</b></p></div>',unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📈 Dataset at a Glance</div>",unsafe_allow_html=True)
    x1,x2,x3,x4=st.columns(4)
    x1.metric("👨‍🎓 Students",len(df))
    x2.metric("🎯 Avg Exam Score",f"{df.exam_score.mean():.1f}")
    x3.metric("📅 Avg Attendance",f"{df.attendance.mean():.1f}%")
    x4.metric("📚 Avg Study Hours",f"{df.study_hours.mean():.1f}")

    st.markdown('<div class="tip">💡 <b>Recommended flow:</b> My Prediction → Understand My Data → How AI Decides → Prediction History.</div>',unsafe_allow_html=True)

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

    # Student information — data-collection flow kept in the same simple form style
    st.markdown("<div class='section-title'>📝 Student Information</div>",unsafe_allow_html=True)
    info1,info2=st.columns(2)
    with info1:
        student_name=st.text_input("Student Name",value=st.session_state.name,placeholder="Enter your full name")
        roll_number=st.text_input("Roll Number",placeholder="Enter your roll number")
    with info2:
        branch=st.selectbox("Branch",["CSE","ECE","EEE","MECH","CIVIL","Other"])
        year=st.selectbox("Year",["1st Year","2nd Year","3rd Year","4th Year"])

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
            now=datetime.now().strftime("%d-%m-%Y %I:%M %p")
            if student_name.strip():
                st.session_state.name=student_name.strip()
            st.session_state.roll_number=roll_number.strip()
            st.session_state.branch=branch
            st.session_state.year=year
            st.session_state.history.insert(0,{"Time":now,"Student":st.session_state.name,"Predicted Score":score,"Academic Health":h,"Performance":label})
            st.session_state.history=st.session_state.history[:10]
            st.session_state.last_values=values.copy()
            st.session_state.last_score=score
            st.session_state.last_health=h

            st.success("🎉 Prediction completed successfully!")
            r1,r2,r3=st.columns(3)
            r1.metric("📚 Predicted Exam Score",score)
            r2.metric("❤️ Academic Health",f"{h}%")
            r3.metric("🏆 Performance",label)

            g1,g2=st.columns(2)
            with g1:
                fig=go.Figure(go.Indicator(mode="gauge+number",value=score,title={"text":"Expected Exam Score"},gauge={"axis":{"range":[0,100]}}))
                fig.update_layout(height=300)
                st.plotly_chart(dark_fig(fig),use_container_width=True)
            with g2:
                fig=go.Figure(go.Indicator(mode="gauge+number",value=h,title={"text":"Academic Health Index"},gauge={"axis":{"range":[0,100]}}))
                fig.update_layout(height=300)
                st.plotly_chart(dark_fig(fig),use_container_width=True)

            st.markdown("<div class='section-title'>🧠 What Does My Result Mean?</div>",unsafe_allow_html=True)
            if score>=85:
                msg=f"🏆 Excellent predicted performance. The model estimates around <b>{score}/100</b>."
                css="success"
            elif score>=70:
                msg=f"👍 Good predicted performance. The model estimates around <b>{score}/100</b>. A few habit improvements may help."
                css="info"
            else:
                msg=f"⚠️ There is room for improvement. The model estimates around <b>{score}/100</b>. Focus on the plan below."
                css="warning"
            st.markdown(f'<div class="{css}">{msg}</div>',unsafe_allow_html=True)

            st.markdown("<div class='section-title'>💡 Personal Improvement Plan</div>",unsafe_allow_html=True)
            for t in tips(values): st.info(t)

            report=pd.DataFrame([{**values,"predicted_exam_score":score,"academic_health":h,"performance":label,"prediction_time":now}])
            st.download_button("📥 Download Prediction Report",report.to_csv(index=False),"student_prediction_report.csv","text/csv",use_container_width=True)
        except Exception as e:
            st.error(f"Prediction Error: {e}")

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

        r1,r2,r3=st.columns(3)
        r1.metric("Baseline",base_score)
        r2.metric("What-If",scenario_score,delta)
        r3.metric("Scenario Health",f"{scenario_health}%")

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
      <h1>📊 Understand Student Data</h1>
      <p>Simple questions, simple charts. No statistics background required.</p>
    </div>
    """,unsafe_allow_html=True)

    f1,f2,f3,f4=st.columns(4)
    genders=sorted(df.gender.dropna().unique())
    motivations=sorted(df.motivation_level.dropna().unique())
    incomes=sorted(df.family_income.dropna().unique())
    internet=sorted(df.internet_quality.dropna().unique())
    with f1: gf=st.multiselect("Gender",genders,default=genders)
    with f2: mf=st.multiselect("Motivation",motivations,default=motivations)
    with f3: inf=st.multiselect("Family Income",incomes,default=incomes)
    with f4: itf=st.multiselect("Internet Quality",internet,default=internet)

    d=df[df.gender.isin(gf)&df.motivation_level.isin(mf)&df.family_income.isin(inf)&df.internet_quality.isin(itf)].copy()
    st.metric("👨‍🎓 Students in selected view",len(d))
    if len(d)==0:
        st.warning("No students match these filters.")
        st.stop()

    st.markdown("<div class='section-title'>🥧 Easy Dataset Snapshot</div>",unsafe_allow_html=True)
    p1,p2=st.columns(2)
    with p1:
        gcounts=d["gender"].value_counts().reset_index()
        gcounts.columns=["gender","count"]
        fig=px.pie(gcounts,names="gender",values="count",hole=.48,title="Who is in the dataset?")
        st.plotly_chart(dark_fig(fig),use_container_width=True)
    with p2:
        mcounts=d["motivation_level"].value_counts().reset_index()
        mcounts.columns=["motivation","count"]
        fig=px.pie(mcounts,names="motivation",values="count",hole=.48,title="How motivated are students?")
        st.plotly_chart(dark_fig(fig),use_container_width=True)

    st.markdown("<div class='section-title'>📚 Habit vs Marks</div>",unsafe_allow_html=True)
    charts=[
        ("attendance","📅 Attendance vs Exam Score","Does attending more classes tend to go with better marks?"),
        ("study_hours","📚 Study Hours vs Exam Score","Do students who study more tend to score higher?"),
        ("sleep_hours","😴 Sleep Hours vs Exam Score","How does sleep duration relate to exam scores?"),
        ("social_media_hours","📱 Social Media vs Exam Score","Is higher social-media usage associated with different scores?"),
        ("previous_scores","📝 Previous Scores vs Exam Score","Does previous performance help explain current performance?")
    ]
    for i in range(0,len(charts),2):
        cols=st.columns(2)
        for col,(feature,title,question) in zip(cols,charts[i:i+2]):
            with col:
                fig=px.scatter(d,x=feature,y="exam_score",trendline="ols",title=title,hover_data=["gender","motivation_level"])
                st.plotly_chart(dark_fig(fig),use_container_width=True)
                st.markdown(f'<div class="info"><b>Student question:</b> {question}<br><b>Remember:</b> This is an association in the dataset, not proof that one factor causes marks.</div>',unsafe_allow_html=True)

    st.markdown("<div class='section-title'>👥 Group Comparisons</div>",unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        st.plotly_chart(dark_fig(px.box(d,x="gender",y="exam_score",title="👨‍🎓 Gender vs Exam Score")),use_container_width=True)
        st.markdown('<div class="info"><b>Read it like this:</b> compare the typical score and spread of the groups.</div>',unsafe_allow_html=True)
    with b:
        st.plotly_chart(dark_fig(px.box(d,x="motivation_level",y="exam_score",title="🎯 Motivation vs Exam Score")),use_container_width=True)
        st.markdown('<div class="info"><b>Read it like this:</b> compare Low, Medium and High motivation groups.</div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        st.plotly_chart(dark_fig(px.box(d,x="family_income",y="exam_score",title="🏠 Family Income vs Exam Score")),use_container_width=True)
    with b:
        st.plotly_chart(dark_fig(px.box(d,x="internet_quality",y="exam_score",title="🌐 Internet Quality vs Exam Score")),use_container_width=True)

    st.markdown("<div class='section-title'>🔥 Correlation Heatmap</div>",unsafe_allow_html=True)
    corr=d.select_dtypes(include=np.number).corr()
    st.plotly_chart(px.imshow(corr,text_auto=".2f",aspect="auto",title="Which numeric factors move together?"),use_container_width=True)
    st.markdown('<div class="tip">🔥 <b>Heatmap:</b> +1 means strong positive movement, -1 means strong negative movement, and 0 means weak linear relationship. Correlation does not prove causation.</div>',unsafe_allow_html=True)

# ============================================================
# MODEL INSIGHTS
# ============================================================
elif page=="🧠 How AI Decides":
    st.markdown("""
    <div class="hero">
      <h1>🧠 How Does the AI Decide?</h1>
      <p>Understand the model without needing a Data Science background.</p>
    </div>
    """,unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <h3>🤖 The simple explanation</h3>
      <p>The model learned patterns from the training dataset. It looks at several features together and estimates an exam score.</p>
      <p><b>Prediction ≠ guarantee.</b> It is an estimate based on the patterns learned from the available data.</p>
    </div>
    """,unsafe_allow_html=True)

    try:
        imp=pd.read_csv("feature_importance.csv")
        fc=next((c for c in ["Feature","feature","feature_name"] if c in imp.columns),imp.columns[0])
        ic=next((c for c in ["Importance","importance","importance_score"] if c in imp.columns),imp.columns[1])
        imp=imp.rename(columns={fc:"Feature",ic:"Importance"}).sort_values("Importance",ascending=False)

        st.markdown("<div class='section-title'>🎯 Which Factors Matter Most to the Model?</div>",unsafe_allow_html=True)
        st.plotly_chart(px.bar(imp.sort_values("Importance"),x="Importance",y="Feature",orientation="h",title="Feature Importance"),use_container_width=True)
        st.markdown('<div class="info"><b>How to read:</b> a larger importance means the model relied more on that feature in this trained model. It does not mean the feature directly causes higher marks.</div>',unsafe_allow_html=True)

        names={"attendance":"📅 Attendance","study_hours":"📚 Study Hours","sleep_hours":"😴 Sleep Hours","social_media_hours":"📱 Social Media","previous_scores":"📝 Previous Scores","physical_activity":"🏃 Physical Activity","mental_health":"🧠 Mental Health","motivation_level":"🎯 Motivation","internet_quality":"🌐 Internet Quality","family_income":"🏠 Family Income","parental_education":"👨‍👩‍👦 Parental Education","part_time_job":"💼 Part-Time Job","age":"🎂 Age","gender":"👤 Gender"}
        st.markdown("<div class='section-title'>🏆 Top 5 Factors</div>",unsafe_allow_html=True)
        for _,row in imp.head(5).iterrows():
            nm=names.get(str(row.Feature),str(row.Feature).replace("_"," ").title())
            st.markdown(f'<div class="card"><h3>{nm}</h3><p>Model importance: <b>{float(row.Importance):.3f}</b></p></div>',unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Feature importance could not be displayed: {e}")

    try:
        res=pd.read_csv("model_results.csv")
        st.markdown("<div class='section-title'>📈 Model Performance</div>",unsafe_allow_html=True)
        st.dataframe(res,use_container_width=True,hide_index=True)
        st.markdown("""
        <div class="card">
          <h3>📖 What do these numbers mean?</h3>
          <ul>
            <li><b>R²:</b> higher is generally better; it describes how much variation the model explains.</li>
            <li><b>MAE:</b> average absolute prediction error; lower is better.</li>
            <li><b>RMSE:</b> error metric that penalizes large mistakes more; lower is better.</li>
          </ul>
        </div>
        """,unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Model results are not available: {e}")

# ============================================================
# HISTORY
# ============================================================
else:
    st.markdown("""
    <div class="hero">
      <h1>🕘 Prediction History</h1>
      <p>Recent predictions made during this browser session.</p>
    </div>
    """,unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No predictions yet. Make a prediction first.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history),use_container_width=True,hide_index=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history=[]
            st.rerun()

st.markdown("---")
st.markdown("<div class='footer'>🎓 Student Academic Intelligence • Python • Pandas • Scikit-Learn • Streamlit • Plotly</div>",unsafe_allow_html=True)
