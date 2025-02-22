import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer ##for Later
from sklearn.model_selection import train_test_split #splits dataset into training and test data 
# Load the dataset
file_path = "/Users/behruzazimov/Downloads/resume_data.csv"

# Read dataset
data = pd.read_csv(file_path)

# Show basic info
print("Dataset loaded successfully!")
data.info()
data.head()

#removing unnecessary rows
data.drop(columns=['address', 'passing_years', 'result_types', 'company_urls', 'start_dates', 'end_dates', 'extra_curricular_organization_links',
'online_links', 'issue_dates', 'expiry_dates', 'age_requirement', 'matched_score'], inplace = True)

#double check that the columns were dropped
data.info()
data.head()

#fills null values with "Missing"
data.fillna("Missing", inplace=True)
data.info()

def score_resumes(resumes, keywords):
    keyword_list = [kw.strip().lower() for kw in keywords.split(",")]
    scored_resumes = []

    for resume in resumes:
        score = sum(resume["content"].count(keyword) for keyword in keyword_list)  
        if score > 0:  
            scored_resumes.append({"filename": resume["filename"], "score": score})

    # Sort resumes by score in descending order
    scored_resumes.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_resumes
