Markdown

# ⚡ DataGenius: Synthetic Data Generator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://datageniuses.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gemini API](https://img.shields.io/badge/AI-Google%20Gemini-orange)

> **Simulate reality, safely.** DataGenius uses AI to generate realistic datasets on demand—solving privacy risks while giving you full control over data quality (Clean vs. Dirty).

<img width="1893" height="731" alt="Screenshot 2025-12-09 081843" src="https://github.com/user-attachments/assets/3731a2f3-ba8d-4d0e-a9fb-29cd7adcce64" />

## 📖 About The Project

Data Analysts and Engineers often face a dilemma: **Real data is sensitive, but public datasets are too simple.**

1.  **The Privacy Problem:** You can't just use production customer data for testing or portfolio projects due to PII (Personally Identifiable Information) and privacy laws like GDPR/HIPAA.
2.  **The Quality Problem:** Tutorial datasets (like Titanic) are "clean," but real-world data is full of errors.

I built **DataGenius** to solve both. It is a Python application that uses **Google's Gemini LLM** to generate **synthetic data** that mirrors real-world complexity without the risk.

Whether you need a pristine dataset for a client demo or a broken, messy dataset to test your cleaning scripts, DataGenius lets you simulate the exact scenario you need.   


### Key Features
* **🛡️ Privacy-Safe Simulation:** Generate realistic "dummy" data (names, medical diagnoses, financial transactions) that looks authentic but contains zero real user info. Perfect for public portfolios.
* **🎚️ Choose Your Difficulty:**
    * **Clean Mode:** Generates standardized, formatted data for easy analysis.
    * **Dirty Mode:** Injects realistic chaos (10-20% nulls, typos, outliers) to challenge your data cleaning skills.
* **🤖 AI Auto-Schema:** Type a prompt like *"Startup employee roster with equity vesting"* and watch the AI build the column structure for you.
* **🏭 Industry Presets:** One-click templates for Finance, Healthcare, E-Commerce, and more.
* **📥 CSV Export:** Download raw CSV files ready for SQL import, Python/R analysis, or Excel.

## 🚀 How to Run Locally

You can run this tool on your own machine in 3 simple steps.

### Prerequisites
* Python 3.8 or higher
* A free [Google Gemini API Key](https://aistudio.google.com/)

#### 1. Clone the Repository
``` in bash
git clone [https://github.com/YOUR-USERNAME/gaming-data-generator.git](https://github.com/YOUR-USERNAME/gaming-data-generator.git)
cd gaming-data-generator
```

#### 2. Install Dependencies 
``` in bash 
pip install -r requirements.txt
```

#### 3. Configure API Key

To keep your key safe, this app uses Streamlit Secrets.

Create a folder named .streamlit in the project root.

Inside it, create a file named secrets.toml.

Inside the file, add your key like this:
```
GOOGLE_API_KEY = "AIzaSyD_YOUR_ACTUAL_KEY_HERE"
```

#### 4. Launch the App

Bash
```
streamlit run app.py
```

### 🛠️ Tech Stack
Frontend: Streamlit (Python-based UI)

AI Engine: Google Gemini 2.5 Flash (via google-generativeai)

Data Handling: Pandas

### 📂 Project Structure
```
DataGenius/
├── app.py                 # The main application logic
├── requirements.txt       # Python dependencies
├── .gitignore             # Security rules (hides secrets)
├── README.md              # Documentation
└── .streamlit/
    └── secrets.toml       # API Keys (Local only, NOT on GitHub)
```
### 🧠 Prompt Engineering
A key part of this project was designing the prompts to ensure the AI generates structured CSV data rather than conversational text.

Challenge: LLMs love to chat.

Solution: I used strict system instructions and temperature controls to force the model to output purely valid CSV strings.

Privacy by Design: The prompts explicitly instruct the model to generate fictional entities, ensuring no real PII is ever leaked into the output.

### 🤝 Contributing
Contributions are welcome! If you have ideas for new "Messy Data" types (e.g., specific SQL injection errors or encoding issues), feel free to fork the repo and submit a Pull Request.

Fork the Project

Create your Feature Branch (git checkout -b feature/NewMessiness)

Commit your Changes (git commit -m 'Add new messiness type')

Push to the Branch (git push origin feature/NewMessiness)

Open a Pull Request

### 💎 Acknowledgments
Google Gemini: This project was co-coded with the assistance of Google Gemini. It acted as a "Pair Programmer," helping to generate the initial Streamlit architecture, debug Python errors, and refine the data generation prompts.

Streamlit Community: For providing the framework that makes deploying Python data tools effortless.

### 📄 License
Distributed under the MIT License. See LICENSE for more information.
<hr>

Built with 💻 and ☕ by Zachary Ball
