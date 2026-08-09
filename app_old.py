
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
.stApp { background:#f4f7fb; }
.block-container { padding-top:1.3rem; max-width:1450px; }

.hero {
    background:linear-gradient(135deg,#4f46e5,#7c3aed,#9333ea);
    padding:34px 38px; border-radius:24px; color:white;
    margin-bottom:24px; box-shadow:0 15px 35px rgba(79,70,229,.20);
}
.hero h1 { margin:0 0 8px 0; font-size:38px; }
.hero p { margin:0; font-size:17px; opacity:.94; }

.section-title { color:#172033 !important; font-size:27px; font-weight:800; margin:24px 0 13px; }
.card {
    background:#ffffff !important; color:#172033 !important;
    border:1px solid #e5e7eb; border-radius:18px; padding:22px;
    box-shadow:0 5px 18px rgba(15,23,42,.06); margin-bottom:14px;
}
.card h3,.card h4,.card p,.card li { color:#172033 !important; }
.card p,.card li { line-height:1.55; color:#475569 !important; }
.feature-card { min-height:185px; }
.feature-icon { font-size:32px; }
.profile {
    background:linear-gradient(135deg,#eef2ff,#f5f3ff) !important;
    border:1px solid #ddd6fe; border-radius:20px; padding:22px;
}
.profile h2,.profile p { color:#172033 !important; }

.info {
    background:#eff6ff !important; border-left:5px solid #3b82f6;
    border-radius:12px; padding:14px 17px; color:#1e3a8a !important;
}
.tip {
    background:#fff7ed !important; border-left:5px solid #f97316;
    border-radius:12px; padding:14px 17px; color:#7c2d12 !important;
}
.success {
    background:#ecfdf5 !important; border-left:5px solid #10b981;
    border-radius:12px; padding:14px 17px; color:#065f46 !important;
}
.warning {
    background:#fffbeb !important; border-left:5px solid #f59e0b;
    border-radius:12px; padding:14px 17px; color:#78350f !important;
}
.login {
    max-width:760px; margin:20px auto; background:#fff !important;
    border:1px solid #e5e7eb; border-radius:24px; padding:35px;
    box-shadow:0 18px 45px rgba(15,23,42,.10);
}
.login h1,.login p { color:#172033 !important; }
.footer { text-align:center; color:#64748b !important; padding:18px 0; }
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
    "history": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------- HELPERS ---------------------------
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
        ("💡","3. Give Improvement Tips","The system checks the entered routine and points out areas that can be improved.","In simple words: What can I improve from today?")
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
      <p>Enter student details and get an ML prediction plus an easy-to-understand action plan.</p>
    </div>
    """,unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    with c1:
        gender=st.selectbox("Gender",["Male","Female"])
        age=st.number_input("Age",15,35,20)
        attendance=st.slider("Attendance (%)",0,100,85)
        study_hours=st.slider("Study Hours / Day",0,40,4)
        previous_scores=st.slider("Previous Scores",0,100,70)
    with c2:
        sleep_hours=st.slider("Sleep Hours / Day",0.0,12.0,7.0,.5)
        social_media_hours=st.slider("Social Media Hours / Day",0.0,10.0,2.5,.5)
        physical_activity=st.slider("Physical Activity (1–5)",1,5,3)
        mental_health=st.slider("Mental Health (1–5)",1,5,4)
        internet_quality=st.selectbox("Internet Quality",["Poor","Average","Good"])
    with c3:
        motivation_level=st.selectbox("Motivation Level",["Low","Medium","High"])
        parental_education=st.selectbox("Parental Education",["High School","College","Postgraduate"])
        family_income=st.selectbox("Family Income",["Low","Medium","High"])
        part_time_job=st.selectbox("Part-Time Job",["No","Yes"])

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
            st.session_state.history.insert(0,{"Time":now,"Student":st.session_state.name,"Predicted Score":score,"Academic Health":h,"Performance":label})
            st.session_state.history=st.session_state.history[:10]

            st.success("🎉 Prediction completed successfully!")
            r1,r2,r3=st.columns(3)
            r1.metric("📚 Predicted Exam Score",score)
            r2.metric("❤️ Academic Health",f"{h}%")
            r3.metric("🏆 Performance",label)

            g1,g2=st.columns(2)
            with g1:
                fig=go.Figure(go.Indicator(mode="gauge+number",value=score,title={"text":"Expected Exam Score"},gauge={"axis":{"range":[0,100]}}))
                fig.update_layout(height=300)
                st.plotly_chart(fig,use_container_width=True)
            with g2:
                fig=go.Figure(go.Indicator(mode="gauge+number",value=h,title={"text":"Academic Health Index"},gauge={"axis":{"range":[0,100]}}))
                fig.update_layout(height=300)
                st.plotly_chart(fig,use_container_width=True)

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
                st.plotly_chart(fig,use_container_width=True)
                st.markdown(f'<div class="info"><b>Student question:</b> {question}<br><b>Remember:</b> This is an association in the dataset, not proof that one factor causes marks.</div>',unsafe_allow_html=True)

    st.markdown("<div class='section-title'>👥 Group Comparisons</div>",unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        st.plotly_chart(px.box(d,x="gender",y="exam_score",title="👨‍🎓 Gender vs Exam Score"),use_container_width=True)
        st.markdown('<div class="info"><b>Read it like this:</b> compare the typical score and spread of the groups.</div>',unsafe_allow_html=True)
    with b:
        st.plotly_chart(px.box(d,x="motivation_level",y="exam_score",title="🎯 Motivation vs Exam Score"),use_container_width=True)
        st.markdown('<div class="info"><b>Read it like this:</b> compare Low, Medium and High motivation groups.</div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        st.plotly_chart(px.box(d,x="family_income",y="exam_score",title="🏠 Family Income vs Exam Score"),use_container_width=True)
    with b:
        st.plotly_chart(px.box(d,x="internet_quality",y="exam_score",title="🌐 Internet Quality vs Exam Score"),use_container_width=True)

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
st.markdown("<div class='footer'>🎓 Student Academic Intelligence System • Python • Pandas • Scikit-Learn • Streamlit • Plotly</div>",unsafe_allow_html=True)
