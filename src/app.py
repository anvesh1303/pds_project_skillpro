from flask import Flask, render_template, request, jsonify
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import PyPDF2
from io import BytesIO
import re


from data_preprocessing import cleanResume, extract_text_from_pdf, preprocess_resume_text
from data_transformation import load_skills_dataset
from modeling import create_pipeline
from results import missing_skills, get_top_skills, calculate_shortlisting_chances, extract_skills, skill_match_percentage

warnings.filterwarnings('ignore')




resumeDataSet = pd.read_csv('/Users/anveshkumar/Desktop/pds_resume_pro/data/resume_dataset.csv', encoding='utf-8')
skills_set = load_skills_dataset('/Users/anveshkumar/Desktop/pds_resume_pro/data/skills_30k.csv')

resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])

X_train, X_test, y_train, y_test = train_test_split(resumeDataSet['cleaned_resume'], resumeDataSet['Category'], test_size=0.25, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2, ngram_range=(1, 2), token_pattern=r'(?u)\b[\w\#.-]+\b')),
    ('clf', LogisticRegression(max_iter=1000, n_jobs=-1))
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test.astype(str), predictions.astype(str), target_names=le.classes_.astype(str)))

### this block can raise some issues. Only use it by removing comments for visualization
# visualize top skills for each category
'''categories = resumeDataSet['Category'].unique()
for category_id in range(len(categories)):
    category_name = le.inverse_transform([category_id])[0]
    top_skills = get_top_skills(pipeline, category_id, 20, skills_set=skills_set)
    skills = [skill for skill, _ in top_skills]
    importance = [percentage for _, percentage in top_skills]

    plt.figure(figsize=(12,6))
    sns.barplot(x=skills, y=importance)
    plt.xlabel('Skill')
    plt.ylabel('Importance Percentage')
    plt.title(f'Top Skills for {category_name} Category')
    plt.xticks(rotation=90)
    plt.show() '''



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', categories=[(idx, name) for idx, name in enumerate(le.classes_)])




@app.route('/analyze', methods=['POST'])
def analyze():
    resume_file = request.files['resume']
    resume_text = extract_text_from_pdf(io.BytesIO(resume_file.read()))

    predicted_category_id = pipeline.predict(preprocess_resume_text(resume_text))[0]
    predicted_category = le.inverse_transform([predicted_category_id])[0]

    top_skills = get_top_skills(pipeline, predicted_category_id, 15, skills_set=skills_set)

    missing_skills_list = missing_skills(pipeline, predicted_category_id, resume_text, skills_set)
    resume_skills = extract_skills(pipeline, predicted_category_id, resume_text, skills_set)
    
    shortlisting_chances = calculate_shortlisting_chances(pipeline, predicted_category_id, resume_text, skills_set)

    return jsonify({
        'predicted_category': predicted_category,
        'top_skills': top_skills,
        'resume_skills': resume_skills,
        'missing_skills': missing_skills_list,
        'shortlisting_chances': shortlisting_chances
    })


@app.route('/top_skills/<int:category_id>')
def top_skills(category_id):
    top_skills_data = get_top_skills(pipeline, category_id, 15, skills_set=skills_set)
    return jsonify(top_skills_data)

if __name__ == '__main__':
    app.run(debug=True)

