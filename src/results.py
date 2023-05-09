import re
from data_transformation import normalize_skill_name
import numpy as np



def missing_skills(pipeline, category_id, resume_text, skills_set):
    top_skills = get_top_skills(pipeline, category_id, 20, skills_set=skills_set)
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



def get_top_skills(pipeline, category_id, top_n=20, skills_set=None):
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


def calculate_shortlisting_chances(pipeline, category_id, resume_text, skills_set=None):
    missing_importance = sum([importance for _, importance in missing_skills(pipeline, category_id, resume_text, skills_set)])
    shortlisting_percentage = max(0, 100 - missing_importance)
    return shortlisting_percentage 

""" def calculate_shortlisting_chances(pipeline, category_id, resume_text, skills_set=None):
    top_skills = get_top_skills(pipeline, category_id, 10, skills_set=skills_set)
    resume_skills = extract_skills(pipeline, category_id, resume_text, skills_set)
    missing_skills_list = missing_skills(pipeline, category_id, resume_text, skills_set)
    
    total_importance_top_skills = sum([importance for _, importance in top_skills])
    missing_skills_importance = sum([importance for _, importance in missing_skills_list])
    resume_skills_importance = sum([importance for _, importance in resume_skills])

    combined_importance = missing_skills_importance + resume_skills_importance
    shortlisting_percentage = max(0, 100 * (resume_skills_importance / combined_importance))
    return shortlisting_percentage """



""" def extract_skills(pipeline, category_id, resume_text, skills_set):
    top_skills = get_top_skills(pipeline, category_id, 20, skills_set=skills_set)
    resume_skills = []

    for skill, importance in top_skills:
        # Create a regex pattern that matches the skill (case-insensitive)
        skill_pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)

        # Check if the skill is in the resume_text
        if skill_pattern.search(resume_text):
            resume_skills.append((skill, importance))

    return resume_skills """
    
def extract_skills(pipeline, category_id, resume_text, skills_set):
    top_skills = get_top_skills(pipeline, category_id, 20, skills_set=skills_set)
    resume_skills = []

    for skill, importance in top_skills:
        # Create a regex pattern that matches the skill (case-insensitive)
        skill_pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)

        # Check if the skill is not in the resume_text
        if skill_pattern.search(resume_text):
            resume_skills.append((skill, importance))

    return resume_skills

def skill_match_percentage(resume_skills, top_skills):
    matched_skills = set([skill for skill, _ in top_skills]).intersection(resume_skills)
    return len(matched_skills) / len(top_skills) * 100
