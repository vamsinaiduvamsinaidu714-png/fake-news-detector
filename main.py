import tkinter as tk
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to check news
def check_news():
    news = entry.get("1.0", tk.END).strip()

    if news == "":
        result_label.config(text="Please enter some news text")
        return

    # Transform input
    transformed_news = vectorizer.transform([news])

    # Predict
    prediction = model.predict(transformed_news)

    print("Prediction:", prediction[0])  # Debug

    # Result
    if prediction[0] == 1:
        result_label.config(text="REAL NEWS ✅")
    else:
        result_label.config(text="FAKE NEWS ❌")

# Clear function
def clear_text():
    entry.delete("1.0", tk.END)
    result_label.config(text="")

# Create window
window = tk.Tk()
window.title("Fake News Detection App")
window.geometry("500x400")

# Title
title_label = tk.Label(window, text="Fake News Detection App",
                       font=("Arial", 18, "bold"))
title_label.pack(pady=10)

# Text box
entry = tk.Text(window, height=6, width=60)
entry.pack(pady=10)

# Check button
check_button = tk.Button(window, text="Check News",
                         command=check_news)
check_button.pack(pady=5)

# Clear button
clear_button = tk.Button(window, text="Clear",
                         command=clear_text)
clear_button.pack(pady=5)

# Result label
result_label = tk.Label(window, text="", font=("Arial", 14))
result_label.pack(pady=20)

# Run app
window.mainloop()