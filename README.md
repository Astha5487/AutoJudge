# AutoJudge – Predicting Programming Problem Difficulty

AutoJudge is a **machine learning–based system** that automatically predicts the **difficulty class** and **difficulty score** of competitive programming problems using **only textual descriptions** (problem, input, output).

It is designed for problems similar to those on platforms like **Codeforces**, **CodeChef**, and **Kattis**.

---

##  Project Overview

###  Tasks

* **Classification** → Predict problem class: `Easy / Medium / Hard`
* **Regression** → Predict problem difficulty score: continuous value in **[1, 10]**

### Input Used (Text Only)

* Problem description
* Input description
* Output description

No user statistics, submissions, or solution code are used.

---

## Dataset

The dataset is stored in **`data.jsonl`**. Each data sample contains:

* `title`
* `description`
* `input_description`
* `output_description`
* `problem_class` → `easy / medium / hard`
* `problem_score` → float value in roughly **[1, 10]**

The dataset is **already labeled** — no manual labeling is required.

---

## Approach

###  Data Preprocessing

All text fields are combined into a single string:

```python
full_text = title + " " + description + " " + input_description + " " + output_description
```

**Text Cleaning Steps:**

* Convert to lowercase
* Remove HTML tags
* Remove URLs
* Collapse multiple spaces/newlines into a single space

---

### Feature Engineering

#### a) Text Features — TF–IDF

* **Vectorizer:** `TfidfVectorizer`
* **N-grams:** Unigrams + Bigrams
* **max_features:** 30,000
* **min_df:** 5
* **sublinear_tf:** Enabled (log-scaled term frequency)

---

#### b) Handcrafted Numeric Features

For each `full_text`:

##### Text Statistics

* `log(1 + text_length)`
* `log(1 + math_symbol_count)`
  (symbols such as `+ - * / = < > ^`)

##### Constraint Indicators

* `has_constraints` → contains **"constraints"**, `<=`, or `≤`
* `has_big_n` → contains **10^k** (e.g., 10^5, 10^6)
* `has_time_limit` → contains **"time limit"** or **"seconds"**

##### Algorithm Keyword Log-Counts

Keywords grouped by topic:

* **Dynamic Programming:** `dp`, `knapsack`, `memoization`
* **Graphs:** `bfs`, `dfs`, `dijkstra`, `max flow`
* **Data Structures:** `segment tree`, `fenwick`, `union find`
* **Math:** `gcd`, `modulo`, `combinatorics`
* **Geometry:** `convex hull`
* **Strings:** `kmp`, `suffix array`, `rolling hash`
* **Greedy:** `two pointers`, `sliding window`

 Numeric features are concatenated with TF–IDF vectors using:

```python
scipy.sparse.hstack
```

---

## Models Used

### Classification

**Final Model:** `LogisticRegression (multinomial)`

**Models Tried:**

* Logistic Regression (best)
* Random Forest Classifier
* Linear SVM (LinearSVC)

Logistic Regression performed best based on **cross-validation accuracy**.

---

### Regression

**Final Model:** `GradientBoostingRegressor`

**Models Tried:**

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Gradient Boosting was selected due to **stable predictions** and **low RMSE**.

Deep learning models are **not used**, in line with project requirements.

---

## Evaluation Metrics

###  Classification (Logistic Regression)

* **Test Accuracy:** ~53.21%

**Observations:**

* **Hard** problems are predicted most reliably
* **Medium** problems are hardest to classify due to overlap with Easy and Hard

---

### Regression (Gradient Boosting Regressor)

* **Test RMSE:** ~2.01
* **Test MAE:** ~1.68

**Observations:**

* Gradient Boosting averages many weak learners
* Predictions are slightly smoothed, which is expected for text-based regression

---

## How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Astha5487/AutoJudge.git
cd AutoJudge
```

---

### 2. Create & Activate Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present:

```bash
pip install streamlit scikit-learn pandas numpy scipy joblib
```

---

### 4. (Optional) Re-train the Models

* Open `ACM_project.ipynb` in Jupyter or Google Colab
* Update base paths if needed
* Run all cells

Generated files inside `models/`:

* `tfidf.pkl`
* `logreg_classifier.pkl`
* `gb_regressor.pkl`
* `label_encoder.pkl`

---

### 5. Run the Streamlit Web App

```bash
streamlit run app.py
```

The app will start at:

```
http://localhost:8501
```

---

##  Web Interface (Streamlit)

### Workflow

1. Enter text into three fields:

   * Problem Description
   * Input Description
   * Output Description
2. Click **🔍 Predict**

### Output

* Predicted difficulty class (**Easy / Medium / Hard**) shown as a colored badge
* Predicted difficulty score (e.g., **6.73 / 10**)

The app runs fully **locally** — no database or authentication required.

---

## Demo Video

 **Demo Video:**
👉 [Click here to watch the demo](YOUR_DEMO_VIDEO_LINK)

> Replace `YOUR_DEMO_VIDEO_LINK` with a YouTube or Google Drive link.

---

## Web Interface Screenshots

### Input Interface
Users can paste the problem description, input description, and output description into the text fields.

![Input Interface](<img width="1434" height="859" alt="Image" src="https://github.com/user-attachments/assets/4afd6661-abc7-4be2-8a87-d04963f0d829" />)

---

### Prediction Output
After clicking the **Predict** button, the model displays:
- Predicted difficulty class (Easy / Medium / Hard)
- Predicted difficulty score (0–10)

![Prediction Output](<img width="1436" height="863" alt="Image" src="https://github.com/user-attachments/assets/35a64799-9230-4b5f-aab4-b8673f69d579" />)

---

## Project Structure

```text
AutoJudge/
│
├── ACM_project.ipynb           # Preprocessing, feature engineering, training, evaluation
├── app.py                      # Streamlit web app
├── data.jsonl                  # Dataset (problem statements + labels)
├── models/
│   ├── tfidf.pkl               # TF–IDF vectorizer
│   ├── logreg_classifier.pkl   # Classification model
│   ├── gb_regressor.pkl        # Regression model
│   └── label_encoder.pkl       # Label encoder
├── Report.pdf                  # Detailed project report (4–8 pages)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Author

**Name:** Astha Jaiswal

**Program:** B.Tech

**Institute:** Indian Institute of Technology, Roorkee

**GitHub:** [@Astha5487](https://github.com/Astha5487)

---

