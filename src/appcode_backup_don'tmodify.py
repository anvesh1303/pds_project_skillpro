#go to code with skill2vec dataset
from flask import Flask, render_template, request, jsonify
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import PyPDF2
from io import BytesIO

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

warnings.filterwarnings('ignore')

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    resumeText = re.sub('RT|cc', ' ', resumeText)
    resumeText = re.sub('#\S+', ' ', resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

def extract_text_from_pdf(pdf_file):
    pdfReader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdfReader.pages)
    text = ''
    for page in range(num_pages):
        pdfPage = pdfReader.pages[page]
        text += pdfPage.extract_text()
    return text

def preprocess_resume_text(text):
    cleaned_text = cleanResume(text)
    return [cleaned_text]

def predict_resume_category(pipeline, resume_text):
    cleaned_resume = preprocess_resume_text(resume_text)
    return pipeline.predict(cleaned_resume)[0]

def load_skills_dataset(filename):
    skills_df = pd.read_csv(filename)
    skills_set = set()

    for key_skills in skills_df['Key Skills']:
        if pd.isnull(key_skills):
            continue

        skills = key_skills.split('|')
        for skill in skills:
            skills_set.add(normalize_skill_name(skill))

    return skills_set

def normalize_skill_name(skill):
    return skill.lower().strip()


def missing_skills(pipeline, category_id, resume_text, skills_set):
    top_skills = get_top_skills(pipeline, category_id, 10, skills_set=skills_set)
    missing_skills_list = []

    for skill, importance in top_skills:
        # Create a regex pattern that matches the skill (case-insensitive)
        skill_pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)

        # Check if the skill is not in the resume_text
        if not skill_pattern.search(resume_text):
            missing_skills_list.append((skill, importance))

    return missing_skills_list


from difflib import SequenceMatcher

def skill_similarity(a, b):
    a_words = set(a.split())
    b_words = set(b.split())
    return len(a_words.intersection(b_words)) / len(a_words.union(b_words))

def merge_duplicate_skills(skills, similarity_threshold=0.5):
    merged_skills = []
    for skill, importance in skills:
        found = False
        for i, (merged_skill, merged_importance) in enumerate(merged_skills):
            if skill_similarity(skill, merged_skill) >= similarity_threshold:
                merged_skills[i] = (merged_skill, merged_importance + importance)
                found = True
                break
        if not found:
            merged_skills.append((skill, importance))
    return sorted(merged_skills, key=lambda x: x[1], reverse=True)



def get_top_skills(pipeline, category_id, top_n=15, skills_set=None):
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    category_coef = clf.coef_[category_id]
    feature_names = tfidf.get_feature_names_out()
    
    if skills_set is not None:
        feature_indices = [i for i, feature in enumerate(feature_names) if normalize_skill_name(feature) in skills_set]
        category_coef = category_coef[feature_indices]
        feature_names = np.array(feature_names)[feature_indices]
    
    avg_importance = np.mean(clf.coef_[:, feature_indices], axis=0) if skills_set is not None else np.mean(clf.coef_, axis=0)
    
    relative_importance_threshold = 1.5
    
    important_indices = np.where(category_coef > avg_importance * relative_importance_threshold)[0]
    top_skills_indices = important_indices[np.argsort(category_coef[important_indices])[-top_n:]]
    
    top_skills = [(feature_names[i], category_coef[i]) for i in top_skills_indices]

    # Normalize the importance values to percentages
    total_importance = sum([importance for _, importance in top_skills])
    top_skills = [(skill, importance / total_importance * 100) for skill, importance in top_skills]

    # Merge duplicate skills
    top_skills = merge_duplicate_skills(top_skills)

    return top_skills


""" def calculate_shortlisting_chances(pipeline, category_id, resume_text, skills_set=None):
    missing_importance = sum([importance for _, importance in missing_skills(pipeline, category_id, resume_text, skills_set)])
    shortlisting_percentage = max(0, 100 - missing_importance)
    return shortlisting_percentage """

def calculate_shortlisting_chances(pipeline, category_id, resume_text, skills_set=None):
    top_skills = get_top_skills(pipeline, category_id, 10, skills_set=skills_set)
    resume_skills = extract_skills(pipeline, category_id, resume_text, skills_set)
    missing_skills_list = missing_skills(pipeline, category_id, resume_text, skills_set)
    
    total_importance_top_skills = sum([importance for _, importance in top_skills])
    missing_skills_importance = sum([importance for _, importance in missing_skills_list])
    resume_skills_importance = sum([importance for _, importance in resume_skills])

    combined_importance = missing_skills_importance + resume_skills_importance
    shortlisting_percentage = max(0, 100 * (resume_skills_importance / combined_importance))
    return shortlisting_percentage



def extract_skills(pipeline, category_id, resume_text, skills_set):
    top_skills = get_top_skills(pipeline, category_id, 20, skills_set=skills_set)
    resume_skills = []

    for skill, importance in top_skills:
        # Create a regex pattern that matches the skill (case-insensitive)
        skill_pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)

        # Check if the skill is in the resume_text
        if skill_pattern.search(resume_text):
            resume_skills.append((skill, importance))

    return resume_skills

def skill_match_percentage(resume_skills, top_skills):
    matched_skills = set([skill for skill, _ in top_skills]).intersection(resume_skills)
    return len(matched_skills) / len(top_skills) * 100


resumeDataSet = pd.read_csv('/Users/anveshkumar/Desktop/resume_dataset.csv', encoding='utf-8')
skills_set = load_skills_dataset('/Users/anveshkumar/Desktop/skill2vec_50K.csv')

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

categories = resumeDataSet['Category'].unique()
""" for category_id in range(len(categories)):
    print(f"Top 10 skills for {le.inverse_transform([category_id])[0]}:")
    for skill, importance in get_top_skills(pipeline, category_id, 10, skills_set=skills_set):
        print(f"{skill}: {importance:.4f}")
    print("\n") """


# Visualize the distribution of resume categories in the resume_dataset
plt.figure(figsize=(12, 6))
resumeDataSet['Category_names'] = resumeDataSet['Category'].map(lambda x: le.inverse_transform([x])[0])
sns.countplot(data=resumeDataSet, x='Category_names', order=resumeDataSet['Category_names'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Distribution of Resume Categories')
plt.show()



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', categories=[(idx, name) for idx, name in enumerate(le.classes_)])


@app.route('/analyze', methods=['POST'])
def analyze():
    resume_file = request.files['resume']
    resume_text = extract_text_from_pdf(io.BytesIO(resume_file.read()))

    predicted_category_id = predict_resume_category(pipeline, resume_text)
    predicted_category = le.inverse_transform([predicted_category_id])[0]

    top_skills = get_top_skills(pipeline, predicted_category_id, 10, skills_set=skills_set)
    

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
    top_skills_data = get_top_skills(pipeline, category_id, 10, skills_set=skills_set)
    return jsonify(top_skills_data)

if __name__ == '__main__':
    app.run(debug=True)