#  Fake Job Posting Detector

A Machine Learning–based application that classifies job postings as **Real**, **Fake**, or **Uncertain** using Natural Language Processing (NLP).

##  Overview
This project analyzes job description text to identify potentially fraudulent postings. It addresses real-world ML challenges such as **class imbalance** and **prediction uncertainty**, and presents results through an easy-to-use web interface.

## Features
- NLP-based text classification (TF-IDF + Logistic Regression)
- Handles class imbalance using class-weighted learning
- Probability-based decision thresholds
- Interactive Streamlit web application
- Pretrained model included for instant use

## Tech Stack
Python, Scikit-learn, Pandas, NumPy, Streamlit

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
