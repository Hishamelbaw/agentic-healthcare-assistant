import streamlit as st
from src.agent import HealthcareAssistant

st.set_page_config(
    page_title="Agentic Healthcare Assistant",
    page_icon="🩺",
    layout="wide"
)

assistant = HealthcareAssistant()

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #123;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            color: #5b6470;
            margin-bottom: 1.5rem;
        }
        .result-box {
            padding: 1rem;
            border-radius: 12px;
            background: #f7f9fc;
            border: 1px solid #dfe6ee;
            margin-top: 0.75rem;
            white-space: pre-wrap;
        }
        .small-note {
            color: #6b7280;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🩺 Agentic Healthcare Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-powered medical task automation for appointments, patient records, history summarization, and trusted medical retrieval.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Quick Demo Prompts")
    st.caption("Use these during your presentation.")
    st.code("show patient record for Rahul", language=None)
    st.code("summarize medical history for Rahul", language=None)
    st.code("book appointment for Rahul with Dr. Smith tomorrow morning", language=None)
    st.code("list all appointments", language=None)
    st.code("search disease information about PCOS", language=None)

    st.divider()
    st.markdown("**Project Features**")
    st.markdown("""
    - Appointment scheduling  
    - Record retrieval and updates  
    - Medical history summarization  
    - Trusted disease information search  
    - RAG over patient reports  
    """)

tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Patient Records",
    "📅 Appointments",
    "🔎 Medical Search",
    "🤖 Assistant"
])

with tab1:
    st.subheader("Patient Record Lookup")
    patient_name = st.text_input("Patient name", key="record_name", placeholder="Rahul")
    col1, col2 = st.columns([1, 5])

    with col1:
        get_record = st.button("Show Record", use_container_width=True)

    if get_record and patient_name.strip():
        query = f"show patient record for {patient_name.strip()}"
        response = assistant.run(query)
        st.session_state.history.append(("Record Lookup", query, response))
        st.markdown(f'<div class="result-box">{response}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Medical History Summary")
    summary_name = st.text_input("Patient name for summary", key="summary_name", placeholder="Rahul")
    if st.button("Summarize History", use_container_width=True):
        query = f"summarize medical history for {summary_name.strip()}"
        response = assistant.run(query)
        st.session_state.history.append(("History Summary", query, response))
        st.markdown(f'<div class="result-box">{response}</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Book Appointment")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        appt_patient = st.text_input("Patient", key="appt_patient", placeholder="Rahul")
    with c2:
        appt_doctor = st.text_input("Doctor", key="appt_doctor", placeholder="Dr. Smith")
    with c3:
        appt_day = st.selectbox("Day", ["today", "tomorrow"], index=1)
    with c4:
        appt_time = st.selectbox("Preferred time", ["morning", "afternoon", "evening"], index=0)

    if st.button("Book Appointment", use_container_width=True):
        query = f"book appointment for {appt_patient.strip()} with {appt_doctor.strip()} {appt_day} {appt_time}"
        response = assistant.run(query)
        st.session_state.history.append(("Book Appointment", query, response))
        st.success("Appointment request processed.")
        st.markdown(f'<div class="result-box">{response}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("View Appointments")
    if st.button("List All Appointments", use_container_width=True):
        query = "list all appointments"
        response = assistant.run(query)
        st.session_state.history.append(("List Appointments", query, response))
        st.markdown(f'<div class="result-box">{response}</div>', unsafe_allow_html=True)

with tab3:
    st.subheader("Trusted Medical Information Search")
    condition = st.text_input("Condition or topic", key="condition_search", placeholder="PCOS")
    if st.button("Search Medical Information", use_container_width=True):
        query = f"search disease information about {condition.strip()}"
        response = assistant.run(query)
        st.session_state.history.append(("Medical Search", query, response))
        st.info("Results may include trusted references and relevant patient/report excerpts.")
        st.markdown(f'<div class="result-box">{response}</div>', unsafe_allow_html=True)

with tab4:
    st.subheader("General Assistant")
    user_input = st.text_area(
        "Ask the assistant",
        placeholder="Example: what does David Thompson's report say about diabetes?",
        height=120
    )

    if st.button("Submit Query", use_container_width=True):
        if user_input.strip():
            response = assistant.run(user_input.strip())
            st.session_state.history.append(("Assistant", user_input, response))
            st.markdown(f'<div class="result-box">{response}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a request.")

    st.divider()
    st.subheader("Session Activity")
    if st.session_state.history:
        for i, (kind, q, r) in enumerate(reversed(st.session_state.history[-8:]), start=1):
            with st.expander(f"{kind}: {q}"):
                st.write(r)
    else:
        st.markdown('<div class="small-note">No activity yet in this session.</div>', unsafe_allow_html=True)