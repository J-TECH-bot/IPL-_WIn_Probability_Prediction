# 🏏 IPL Winning Probability Predictor

This project is a **Machine Learning-powered** web application that predicts the **winning probability** of an IPL cricket team in real time, based on current match statistics. It uses past IPL match data to train a model that understands match dynamics like run rate, wickets, and target.

---

## 📌 Features

- Predicts **real-time winning probability** between two IPL teams.
- Uses historical match data (`matches.csv`, `deliveries.csv`).
- Built with **scikit-learn** and deployed using **Streamlit**.
- Interactive user interface for entering match details.
- Pre-trained ML model for instant predictions.

---

## 📂 Repository Structure

.
├── .idea/ # Project configuration files
├── IPL_Winning_probability.ipynb # Model training & analysis notebook
├── app.py # Main Streamlit app
├── deliveries.csv # Ball-by-ball IPL match data
├── joblib_model.pkl # Pre-trained ML model
├── matches.csv # IPL match summaries
├── most_runs_average_strikerate.csv # Player performance stats
├── pipe.pkl # Preprocessing pipeline
├── Players.xlsx # Players information
├── requirements.txt # Dependencies list
├── teams.csv # IPL team information
├── teamwise_home_and_away.csv # Home/away match records
├── test_model.py # Model testing script


---

## 📊 Dataset

The project uses multiple IPL datasets:

- **matches.csv** – Match results and summary.
- **deliveries.csv** – Ball-by-ball match data.
- **teamwise_home_and_away.csv** – Team home/away performance.
- **most_runs_average_strikerate.csv** – Player batting stats.

Source: [Kaggle IPL Dataset](https://www.kaggle.com/manasgarg/ipl)

---

## 🧠 Model Details

- **Algorithm**: Logistic Regression (via scikit-learn pipeline).
- **Features Used**:  
  - Current Score  
  - Overs Completed  
  - Wickets Lost  
  - Target Score  
  - Remaining Balls  
  - Required Run Rate  
  - Team Batting & Bowling

The model outputs probabilities for **Batting Team Win** and **Bowling Team Win**.

---

## 🚀 Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/J-TECH-bot/IPL_Winning_Probability.git
cd IPL_Winning_Probability

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app locally
streamlit run app.py

🌐 Deployment

The project can be deployed on Streamlit Cloud or any Python hosting service.
Example:

streamlit run app.py

📸 Screenshots

1. Web App Interface


📜 License

Licensed under the MIT License. You are free to use and modify it for educational purposes.

🙌 Acknowledgements

IPL datasets from Kaggle

Built with Python, Pandas, scikit-learn, Streamlit


---

If you want, I can also make you a **flow diagram** showing:  
`User Inputs → Preprocessing Pipeline → ML Model → Winning Probability Output`  
which will make the README more visually appealing.  

Do you want me to create that diagram?
