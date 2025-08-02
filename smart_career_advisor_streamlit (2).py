#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Dummy data for model training
data = pd.DataFrame({
    'skills': ["Python SQL", "Java HTML", "Excel Communication", "Design Photoshop", "AI ML Python"],
    'qualification': ["B.Tech", "B.Sc", "BBA", "BA", "M.Tech"],
    'experience': [2, 1, 3, 0, 4],
    'interests': ["Data Science", "Web Dev", "Management", "UI UX", "AI ML"],
    'career': ["Data Analyst", "Web Developer", "Project Manager", "UX Designer", "ML Engineer"]
})
data['combined'] = data['skills'] + ' ' + data['qualification'] + ' ' + data['interests']

# Vectorization
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(data['combined'])
X = pd.DataFrame(X_text.toarray())
X.columns = X.columns.astype(str)  # Convert column names to strings
X['experience'] = data['experience'].values

# Encode target
le = LabelEncoder()
y = le.fit_transform(data['career'])

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("Smart Career Advisor")
st.write("Upload a JSON file or fill in the form to get a career recommendation.")

# JSON uploader
uploaded_file = st.file_uploader("Upload JSON file", type="json")
if uploaded_file:
    user_data = json.load(uploaded_file)
    st.json(user_data)
    skills = user_data.get('skills', '')
    qualification = user_data.get('qualification', '')
    try:
        experience = int(user_data.get('experience', 0))
    except:
        experience = 0
    interests = user_data.get('interests', '')
    input_text = skills + ' ' + qualification + ' ' + interests
    input_vec = vectorizer.transform([input_text]).toarray()[0].tolist()
    input_vec.append(experience)
    # Build input DataFrame with correct column names
    input_df = pd.DataFrame([input_vec], columns=list(vectorizer.get_feature_names_out()) + ['experience'])
    prediction = model.predict(input_df)[0]

    career = le.inverse_transform([prediction])[0]
    st.success(f"Recommended Career: {career}")
else:
    # Form input
    with st.form("career_form"):
        skills = st.text_input("Enter your skills (comma-separated)", "Python, SQL")
        qualification = st.text_input("Enter your qualification", "B.Tech")
        experience = st.slider("Years of experience", 0, 10, 2)
        interests = st.text_input("Enter your interests", "AI ML")
        submit = st.form_submit_button("Get Recommendation")

    if submit:
        input_text = skills + ' ' + qualification + ' ' + interests
        input_vec = vectorizer.transform([input_text]).toarray()[0].tolist()
        input_vec.append(experience)
        prediction = model.predict([input_vec])[0]
        career = le.inverse_transform([prediction])[0]
        st.success(f"Recommended Career: {career}")


# In[ ]:




