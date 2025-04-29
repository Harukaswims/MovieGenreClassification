# MovieGenreClassification
# 🎬 Movie Genre Classification (IMDb Dataset)

## **Objective**

The objective of this project is to build a **multi-label classification system** that predicts the **genres of movies** based on their **plot descriptions**. By leveraging **Natural Language Processing (NLP)** and various machine learning models, the system can accurately classify movies into one or more genres.

---

## **Dataset**

The dataset is sourced from **IMDb** and contains:

- **Training Data:**
  - Movie ID
  - Movie Title
  - Movie Genres (multi-label)
  - Movie Description (plot summary)

- **Test Data:**
  - Movie ID
  - Movie Title
  - Movie Description (without genres)

---

## **Models Implemented**

1. **Logistic Regression** (Baseline)
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**
4. **BERT (Transformer-based Model)** – **Best Performing**

---

## **Model Comparison (Validation Accuracy)**

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | *34.91%* |
| Random Forest        | *1.03%*  |
| SVM                  | *40.02%* |
| BERT                 | **XX.XX%** (Best) |

*BERT outperformed traditional models due to its contextual understanding of movie descriptions.*

---

## **Project Structure**

movie-genre-classification/ │ ├── notebooks/ # Kaggle notebook (.ipynb) with all steps ├── models/ # Saved models (Logistic, RF, SVM, BERT) ├── outputs/ # Evaluation results, plots ├── bert_genre_classifier/ # BERT model & tokenizer ├── README.md # Project documentation └── requirements.txt # Python dependencies

## 📝 Steps to Run the Project

### 1. Clone the Repository

git clone https://github.com/your-username/movie-genre-classification.git
cd movie-genre-classification

### 2. Install Dependencies
Copy
Edit
pip install -r requirements.txt
The requirements.txt includes:

pandas

numpy

scikit-learn

nltk

matplotlib

seaborn

torch

transformers

datasets

3. Prepare the Dataset
Download the IMDb Genre Classification Dataset manually (train/test files).

Place them in a data/ folder like this:

kotlin
Copy
Edit
movie-genre-classification/
└── data/
    ├── train_data.txt
    ├── test_data.txt
    ├── test_data_solution.txt
    └── description.txt
4. Run the Notebook
Open the provided notebook under notebooks/ using:

bash
Copy
Edit
jupyter notebook
Or upload it back to Kaggle for easy GPU access.

Follow steps inside the notebook:

Data Preprocessing

TF-IDF Feature Engineering

Model Training (Logistic Regression, Random Forest, SVM, BERT)

Model Evaluation and Visual Comparison

5. Download Trained Models (if running on Kaggle)
After training in Kaggle:

Use download links generated at notebook end

OR

Go to Kaggle > Output Files tab → Download

Models:

logistic_regression_model.pkl

random_forest_model.pkl

svm_model.pkl

bert_genre_classifier.zip

6. Predict Using Saved Models
✅ Traditional Models (Logistic Regression / SVM / Random Forest):

python
Copy
Edit
import joblib

# Load TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load model
model = joblib.load('logistic_regression_model.pkl')

# Predict
description = ["A heartwarming story about a young boy and his dog."]
description_clean = clean_text(description[0])
description_vectorized = vectorizer.transform([description_clean])
predictions = model.predict(description_vectorized)
✅ BERT Model:

python
Copy
Edit
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert_genre_classifier')
model = BertForSequenceClassification.from_pretrained('bert_genre_classifier')

# Predict
inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=256)
outputs = model(**inputs)
preds = torch.sigmoid(outputs.logits)
📋 Evaluation Criteria
Functionality:

Successfully trained, evaluated, and compared multiple models.

Best performance achieved using BERT.

Code Quality:

Clean, modular, well-commented

Innovation & Creativity:

Compared traditional ML techniques with deep learning (BERT).

👨‍💻 Author
Ritesh Kashyap
