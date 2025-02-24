import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import re
from difflib import SequenceMatcher

file_path = "/resume_data.csv"
data = pd.read_csv(file_path)

# Drop unnecessary columns
drop_cols = [
    'address', 'passing_years', 'result_types', 'company_urls',
    'start_dates', 'end_dates', 'extra_curricular_organization_links',
    'online_links', 'issue_dates', 'expiry_dates',
    'age_requirement', 'matched_score'
]
data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True)

# Handle missing values
data.fillna("", inplace=True)

# Combine all text into one field for embedding
data['all_text'] = data.apply(lambda row: " ".join(row.values.astype(str)), axis=1)

print("Data Preprocessing Completed")

# Load Pre-Trained Sentence Transformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Convert Resume Text into Embeddings
resume_embeddings = np.array([embedder.encode(text) for text in data['all_text']])

print("Resume Embeddings Created! Shape:", resume_embeddings.shape)

import re
from difflib import SequenceMatcher

# Extract years of experience
def extract_experience(text):
    match = re.search(r'(\d+)', str(text))
    return int(match.group(1)) if match else 0

# Extract GPA (Normalized to 0-4 scale)
def extract_gpa(text):
    match = re.search(r'(\d+(\.\d+)?)', str(text))
    gpa = float(match.group(1)) if match else 0
    return min(gpa, 4)

# Fuzzy skill similarity function
def skill_similarity(skill1, skill2):
    return SequenceMatcher(None, skill1, skill2).ratio()

# Compute skill match score with a minimum threshold
def skill_match_score(job_query, resume_skills):
    job_words = job_query.lower().split()
    resume_words = resume_skills.lower().split()

    # Compute similarity for each skill in resume against job query
    match_scores = [max(skill_similarity(skill, job_word) for job_word in job_words) for skill in resume_words]

    score = sum(match_scores) / len(job_words) if job_words else 0  # Avoid division by zero

    # Apply a minimum skill relevance threshold
    return score if score > 0.3 else 0  # If the match is weak, return 0

# Strict major filtering for technical roles
def major_relevance(job_query, major):
    relevant_majors = {"computer science", "software engineering", "data science", "artificial intelligence", "information technology"}

    if any(major.lower() in relevant_majors for major in major.split()):
        return 1  # Give full credit if the major is relevant
    else:
        return 0  # Exclude non-relevant majors

# Experience boost with stronger weight
def experience_boost(years):
    return (years / 5) ** 1.8  # Exponential scaling to favor experienced candidates