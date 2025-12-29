#  ⚖️ AutoJudge

AutoJudge is a Streamlit web app that predicts the difficulty level and difficulty score of programming problems using NLP and machine learning.

It is perfect for competitive programmers, educators, and coding platforms to quickly assess problem complexity.

## 🚀 Features

- Difficulty Level Prediction: Easy, Medium, Hard, etc.

- Difficulty Score Prediction: Numerical score representing problem complexity.

- Supports full problem statements: problem description, input, and output.

- Built with Random Forest models for classification and regression.

- Interactive Streamlit interface for easy use.

📂 Project Structure
```bash
AutoJudge/
├── app.py                  # Streamlit app
├── models/                 # Trained ML models (tfidf.pkl, classifier.pkl, regressor.pkl)
├── data.jsonl              # Dataset used to train models
├── project.ipynb           # Notebook for training and preprocessing
├── requirements.txt        # Python dependencies
└── .gitignore
```

## 🛠 Installation

1️⃣ Clone the repository:
```bash
git clone https://github.com/Astha5487/AutoJudge.git
cd AutoJudge
```


2️⃣ Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

4️⃣ Run the app locally:
```bash
streamlit run app.py
```
- The app will open in your browser at http://localhost:8501.

## 📊 How It Works

1. User Input: Paste problem description, input format, and output format.

2. Preprocessing: Cleans text and keeps relevant symbols.

3. Feature Extraction: Converts text into TF-IDF vectors.

4. Prediction:

  - Classifier predicts the difficulty level.

  - Regressor predicts a numerical difficulty score.

5. Results: Displays predicted difficulty level and score instantly.

## 💾 Models

The models/ folder contains pre-trained ML models:
```bash
- tfidf.pkl — TF-IDF vectorizer

- classifier.pkl — Random Forest classifier for difficulty level

- regressor.pkl — Random Forest regressor for difficulty score
```
- Note: The models are trained on problem statements from data.jsonl.

## 🌐 Deployment on Streamlit Cloud

1. Go to Streamlit Cloud
 and log in.

2. Connect your GitHub repository AutoJudge.

3. Deploy the app.

4. Make sure models/ folder and requirements.txt are included in the repo.

5. Done! Your app will be live online.

## 📝 Usage

- Open the AutoJudge app.

- Paste the Problem Description, Input Description, and Output Description.

- Click Predict.

- View the predicted Difficulty Level and Difficulty Score instantly.

## 🛠 Technologies Used

- Python 3.11

- Streamlit — Web frontend

- scikit-learn — ML models

- Pandas & NumPy — Data processing

- Pickle — Save/load models

- TF-IDF — Text vectorization

## 📈 Future Improvements

- Add support for multiple languages.

- Add real-time problem difficulty feedback from users.

- Enhance models with deep learning NLP architectures (BERT, GPT, etc.).

📄 License

## MIT License – Open source

Author: Astha Jaiswal

GitHub: https://github.com/Astha5487
