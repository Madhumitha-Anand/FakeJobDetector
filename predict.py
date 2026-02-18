import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# User input
text = input("Paste job description here:\n")

prediction = model.predict([text])[0]

if prediction == 1:
    print("⚠️ This job posting looks FAKE")
else:
    print("✅ This job posting looks REAL")
