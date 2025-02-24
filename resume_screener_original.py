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

# Load Resume Data
file_path = "resume_data.csv"
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


# In[ ]:


# Load Pre-Trained Sentence Transformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Convert Resume Text into Embeddings
resume_embeddings = np.array([embedder.encode(text) for text in data['all_text']])

print("Resume Embeddings Created! Shape:", resume_embeddings.shape)


# In[ ]:


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


# In[ ]:


data['experience'] = data['experiencere_requirement'].apply(extract_experience)
data['gpa'] = data['educational_results'].apply(extract_gpa)

print("Experience and GPA extracted!")


# In[ ]:


# Convert to PyTorch Tensors
X_tensor = torch.tensor(resume_embeddings, dtype=torch.float32)
y_tensor = torch.tensor(data['gpa'].values, dtype=torch.float32).unsqueeze(1)

# Split into Train and Test Sets (80/20 Split)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create PyTorch Datasets and Dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Data Split into Training and Testing Sets")


# In[ ]:


# Define Neural Network Model
class ResumeMatchModel(nn.Module):
    def __init__(self, input_size):
        super(ResumeMatchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Output a probability score

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize Model
model = ResumeMatchModel(input_size=X_train.shape[1])
criterion = nn.MSELoss()  # MSE for ranking
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model Initialized")


# In[ ]:


def predict_best_resumes(job_description, model, embedder, data,
                         skill_weight=8.0, experience_weight=10.0, major_weight=2.0, gpa_weight=1.0,
                         top_n=20, bottom_n=10):
    """
    Predicts the best and worst resumes for a given job description.
    - Uses improved skill matching.
    - Excludes irrelevant majors from high ranking.
    - Experience now has a stronger boost.
    """

    job_embedding = torch.tensor(embedder.encode([job_description]), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(resume_embeddings, dtype=torch.float32)).numpy().flatten()

    # Normalize scores to 0-10 range
    min_score, max_score = scores.min(), scores.max()
    scores = 10 * (scores - min_score) / (max_score - min_score)

    # Adjust scores based on weighted factors
    for idx, row in data.iterrows():
        exp_score = experience_boost(row['experience']) * experience_weight  # Stronger boost for experience
        gpa_score = (row['gpa'] / 4) * gpa_weight  # Lowest weight for GPA

        # Skill Match Boost (Improved)
        skill_score = skill_match_score(job_description, row['skills']) * skill_weight

        # Major Match Boost (Strict Filtering)
        major_score = major_relevance(job_description, row['major_field_of_studies']) * major_weight

        # Apply a minimum required skill score to eliminate bad matches
        if skill_score == 0:
            scores[idx] = 0  # Force bad matches to have the lowest score
        else:
            scores[idx] += (exp_score + gpa_score + skill_score + major_score)

    # Ensure max score is 10
    scores = np.clip(scores, 0, 10)

    # Add scores to data
    data['match_score'] = scores
    ranked_resumes = data.sort_values(by='match_score', ascending=False)

    # Get top `top_n` resumes and bottom `bottom_n` resumes
    top_resumes = ranked_resumes.head(top_n)
    bottom_resumes = ranked_resumes.tail(bottom_n)

    return top_resumes, bottom_resumes


# In[ ]:


query = "Looking for an Accountant with Math skills"
top_results, bottom_results = predict_best_resumes(query, model, embedder, data, top_n=20, bottom_n=10)

# Print Top 20 Matching Resumes
print("\n🔹🔹🔹 TOP 20 BEST MATCHING RESUMES 🔹🔹🔹\n")
for idx, row in top_results.iterrows():
    print(f"📌 Resume Index: {idx}")
    print("=" * 50)
    print(f"🟢 Match Score: {round(row['match_score'], 2)} / 10\n")
    print(f"👨‍🎓 **Major:** {row['major_field_of_studies']}")
    print(f"💼 **Experience:** {extract_experience(row['experiencere_requirement'])} years")
    print(f"📊 **GPA:** {extract_gpa(row['educational_results'])}\n")
    print(f"🛠 **Top Skills:** {row['skills']}")
    print("=" * 50 + "\n")

# Print Bottom 10 Least Matching Resumes
print("\n🔻🔻🔻 BOTTOM 10 LOWEST MATCHING RESUMES 🔻🔻🔻\n")
for idx, row in bottom_results.iterrows():
    print(f"📌 Resume Index: {idx}")
    print("=" * 50)
    print(f"🔴 Match Score: {round(row['match_score'], 2)} / 10\n")
    print(f"👨‍🎓 **Major:** {row['major_field_of_studies']}")
    print(f"💼 **Experience:** {extract_experience(row['experiencere_requirement'])} years")
    print(f"📊 **GPA:** {extract_gpa(row['educational_results'])}\n")
    print(f"🛠 **Top Skills:** {row['skills']}")
    print("=" * 50 + "\n")


# In[ ]:




