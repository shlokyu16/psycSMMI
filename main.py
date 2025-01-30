import streamlit as st
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from fn import *



def main():
    st.title("Social Media & Mental Health Survey")
    
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    
    if not st.session_state.submitted:
        with st.form("survey_form"):
            name = st.text_area("Name")
            age = st.number_input("Age", min_value=10, max_value=100, step=1)
            sex = st.selectbox("Sex", ["Male", "Female", "Other"])
            relationship_status = st.selectbox("Relationship Status", ["Single", "In a relationship", "Married", "Divorced"])
            occupation = st.selectbox("Occupation", ["University Student", "School Student", "Salaried Worker", "Retired"])
            social_media_user = st.radio("Are you a social media user?", ["Yes", "No"])
            platforms_used = st.text_area("Platforms Used (comma-separated)")
            time_spent = st.number_input("Time Spent on Social Media (hours per day)", min_value=0, max_value=24, step=1)
            
            st.write("### Mental Health Questions (Scale: 1-5)")
            adhd_q1 = st.slider("Purposeless use of Social Media", 1, 5, 3)
            adhd_q2 = st.slider("Distracted by Social Media", 1, 5, 3)
            adhd_q3 = st.slider("Ease of Distraction by Social Media", 1, 5, 3)
            adhd_q4 = st.slider("Difficulty in concentrating", 1, 5, 3)
            anxiety_q1 = st.slider("Restlessness if Social Media not used", 1, 5, 3)
            anxiety_q2 = st.slider("Bothered by worries", 1, 5, 3)
            self_esteem_q1 = st.slider("Comparison of self to peers", 1, 5, 3)
            self_esteem_q2 = st.slider("Feelings about above comparison", 1, 5, 3)
            self_esteem_q3 = st.slider("Validation sought from Social Media", 1, 5, 3)
            depression_q1 = st.slider("Feelings of Depression", 1, 5, 3)
            depression_q2 = st.slider("Fluctuation of interest", 1, 5, 3)
            depression_q3 = st.slider("Sleep Issues", 1, 5, 3)
            
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                data = {
                    "Time Spent": time_spent,
                    "ADHD Q1": adhd_q1,
                    "ADHD Q2": adhd_q2,
                    "ADHD Q3": adhd_q3,
                    "ADHD Q4": adhd_q4,
                    "Anxiety Q1": anxiety_q1,
                    "Anxiety Q2": anxiety_q2,
                    "Self Esteem Q1": self_esteem_q1,
                    "Self Esteem Q2": self_esteem_q2,
                    "Self Esteem Q3": self_esteem_q3,
                    "Depression Q1": depression_q1,
                    "Depression Q2": depression_q2,
                    "Depression Q3": depression_q3
                }
                df = pd.DataFrame([data])
                df = refine(df)
                outcome = predict(df)
                
                st.session_state.submitted = True
                st.session_state.outcome = outcome
                st.rerun()
                
    if st.session_state.submitted:
        outcome = st.session_state.outcome
        st.write("### Result")
        if outcome == 1:
            st.error("Alert: Seems like you might have mental illness. Please contact a doctor.")
        else:
            st.info("Yay! You are totally fit mentally.")
            
if __name__ == "__main__":
    main()

