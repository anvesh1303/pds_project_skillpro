import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

def split_data(resumeDataSet):
    X_train, X_test, y_train, y_test = train_test_split(resumeDataSet['cleaned_resume'], resumeDataSet['Category'], test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test