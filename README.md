⚖️ AutoJudge

AutoJudge is a Streamlit-based web application that predicts the difficulty level and difficulty score of programming problems using Natural Language Processing (NLP) and machine learning. It helps competitive programmers, educators, and coding platforms quickly gauge problem complexity.

🚀 Features

Predicts Difficulty Level: Easy, Medium, Hard, etc.

Predicts Difficulty Score: Numerical score representing problem complexity.

Supports full problem descriptions: Problem statement, input format, and output format.

Built with Random Forest models for classification and regression.

Lightweight and interactive Streamlit frontend for easy user experience.

📂 Project Structure
AutoJudge/
├── app.py                  # Streamlit app
├── models/                 # Trained ML models (tfidf.pkl, classifier.pkl, regressor.pkl)
├── data.jsonl              # Dataset used to train models
├── project.ipynb           # Jupyter notebook for model training & preprocessing
├── requirements.txt        # Python dependencies
└── .gitignore

🛠 Installation

Clone the repository:

git clone https://github.com/Astha5487/AutoJudge.git
cd AutoJudge


Create a virtual environment (optional but recommended):

python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows


Install dependencies:

pip install -r requirements.txt


Run the app locally:

streamlit run app.py


The app will open in your default browser at http://localhost:8501.

📊 How It Works

User Inputs: Paste the problem description, input, and output format.

Preprocessing: The app cleans the text, removing unwanted characters while preserving key symbols.

Feature Extraction: Converts text to TF-IDF vectors.

Prediction:

Classifier predicts the difficulty level.

Regressor predicts a numerical difficulty score.

Result: Displays predicted difficulty and score interactively.

💾 Models

The models/ folder contains pre-trained machine learning models:

tfidf.pkl — TF-IDF vectorizer.

classifier.pkl — Random Forest classifier for difficulty level.

regressor.pkl — Random Forest regressor for difficulty score.

Note: These models are trained on problem statements in data.jsonl.

🌐 Deployment

You can deploy AutoJudge easily on Streamlit Cloud:

Go to Streamlit Cloud
 and log in.

Connect your GitHub repo AutoJudge.

Deploy the app.

Ensure that the models/ folder and requirements.txt are included in the repo.

Done! Your app will be live online.

📝 Usage

Open the AutoJudge app.

Paste the Problem Description, Input Description, and Output Description in the respective text areas.

Click Predict.

View the predicted Difficulty Level and Difficulty Score instantly.

🛠 Technologies Used

Python 3.11

Streamlit — for web app frontend

scikit-learn — for machine learning models

Pandas & NumPy — for data manipulation

Pickle — to load/save models

TF-IDF — text vectorization

📈 Future Improvements

Add support for multiple languages.

Implement real-time problem difficulty feedback from users.

Enhance models with deep learning NLP architectures (BERT, GPT, etc.).

📄 License

This project is open source and available under the MIT License.
