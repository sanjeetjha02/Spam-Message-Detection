# 📧 Spam Message Detection using LSTM & Flask

## 📌 Overview
This project is a **Spam Message Detection System** using **Deep Learning (LSTM)** and a **Flask Web Application**. It classifies messages as **Spam** or **Not Spam** based on a large dataset of text messages.

### 🚀 Key Features:
- **Uses an LSTM-based Neural Network** for text classification.
- **Trained on a large spam dataset** for better accuracy.
- **Interactive Flask Web App** for real-time spam detection.
- **User-friendly UI** with a modern design.
- **Works with real-world spam messages** (SMS/Emails, etc.).

---
## 📂 Project Structure
```
spam-detection-flask/
│── static/              # Static files (CSS, Images, JS)
│── templates/           # HTML templates
│   │── home.html        # Input form UI
│   │── result.html      # Prediction result UI
│── model.h5             # Trained LSTM model
│── tokenizer.pkl        # Tokenizer for text processing
│── spam.csv             # Spam dataset
│── train_model.py       # Train the LSTM model
│── app.py               # Flask web application
│── requirements.txt     # Dependencies list
│── README.md            # Project Documentation
```

---
## 📥 Dataset
We use the **SMS Spam Collection Dataset**, which contains **5,574 messages** labeled as **spam or ham (not spam)**.

📌 **Download Dataset:**  
- [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [UCI Machine Learning Repository](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

📌 **Example Entries:**  
```
label,message
ham,"Hey, are you coming to the party?"
spam,"Congratulations! You have won $1000. Claim now!"
ham,"Please call me back."
spam,"Click this link to get a free iPhone!"
```

---
## 🛠 Installation & Setup
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Train the LSTM Model**
```bash
python train_model.py
```
This will generate:
- `model.h5` → Trained Spam Detection Model
- `tokenizer.pkl` → Tokenizer for text processing

### **3️⃣ Run the Flask App**
```bash
python app.py
```
✅ Open browser → **http://127.0.0.1:5000/**

---
## 🎯 How It Works
1️⃣ **User enters a message** in the input box.  
2️⃣ **Model processes the text** using the trained LSTM model.  
3️⃣ **Displays result** (Spam or Not Spam) on the webpage.  

### ✅ Example Test Cases:
| **Message** | **Prediction** |
|------------|---------------|
| "Win a free iPhone! Click here." | Spam |
| "Hey, let's meet at 6 PM." | Not Spam |
| "Congratulations! You won $500." | Spam |
| "Please call me when you're free." | Not Spam |

---
## 🔥 Technologies Used
- **Python** 🐍
- **Flask** 🌐 (Web Framework)
- **TensorFlow/Keras** 🤖 (Deep Learning)
- **LSTM (Long Short-Term Memory)** 📊 (Neural Network)
- **HTML, CSS** 🎨 (Frontend UI)

---
## 📝 To-Do List (Future Enhancements)
- ✅ Improve model accuracy with **more layers** in LSTM.
- ✅ Add **Spam Probability Score** for messages.
- ✅ Deploy the app using **Flask + Docker**.
- ✅ Convert to a **mobile app** using Flutter or React Native.

---
## 📜 License
This project is **open-source** and available under the MIT License.

---
## 🙌 Contributing
Feel free to **fork** this repository and improve it. Pull requests are welcome! 🚀

---
## ✉ Contact
📧 Email: [ujjwalkumar0514@gmail.com](mailto:your-email@example.com)  
🔗 LinkedIn: [Ujjwal Kumar](https://www.linkedin.com/in/ujjwalkumarsahni)  
🌍 GitHub: [Ujjwal Kumar](https://github.com/ujjwalkumarsahni)

