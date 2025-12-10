Markdown

# ⚡ DataGenius: Synthetic Data Generator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gemini API](https://img.shields.io/badge/AI-Google%20Gemini-orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/zjrball/DataGenius/blob/main/LICENSE.txt)

> **Simulate reality, safely.** DataGenius uses AI to generate realistic datasets on demand—solving privacy risks while giving you full control over data quality (Clean vs. Dirty).

<img width="1893" height="731" alt="Screenshot 2025-12-09 081843" src="https://github.com/user-attachments/assets/3731a2f3-ba8d-4d0e-a9fb-29cd7adcce64" />

---

## Quick Start Guide

Watch the **[YouTube Tutorial](LINK_HERE)** for a step-by-step walkthrough.

---

## AI Implementation Note

This project was built using an **AI-First Workflow**.

* **Role of AI:** I utilized Google Gemini & Github Copilot as "Pair Programmers" to generate the initial Streamlit boilerplate, debug Python errors, and draft the regex patterns for data validation.
* **Role of Human:** I designed the application architecture, engineered the specific prompts to ensure data quality, verified the code for security (API key handling), and managed the deployment pipeline.
* **Goal:** To demonstrate how AI tools can accelerate the development of internal tools for Data Analysis teams.

---

## About The Project

Data Analysts and Engineers often face a dilemma: **Real data is sensitive, but public datasets are too simple.**

1.  **The Privacy Problem:** You can't just use production customer data for testing or portfolio projects due to PII (Personally Identifiable Information) and privacy laws like GDPR/HIPAA.
2.  **The Quality Problem:** Tutorial datasets (like Titanic) are "clean," but real-world data is full of errors.

I built **DataGenius** to solve both. It is a Python application that uses **Google's Gemini LLM** to generate **synthetic data** that mirrors real-world complexity without the risk.

Whether you need a pristine dataset for a client demo or a broken, messy dataset to test your cleaning scripts, DataGenius lets you simulate the exact scenario you need.   

---

### Key Features
* **Privacy-Safe Simulation:** Generate realistic "dummy" data (names, medical diagnoses, financial transactions) that looks authentic but contains zero real user info. Perfect for public portfolios.
* **Choose Your Difficulty:**
    * **Clean Mode:** Generates standardized, formatted data for easy analysis.
    * **Dirty Mode:** Injects realistic chaos (10-20% nulls, typos, outliers) to challenge your data cleaning skills.
* **AI Auto-Schema:** Type a prompt like *"Startup employee roster with equity vesting"* and watch the AI build the column structure for you.
* **Industry Presets:** One-click templates for Finance, Healthcare, E-Commerce, and more.
* **CSV Export:** Download raw CSV files ready for SQL import, Python/R analysis, or Excel.

---

## How to Run Locally
For a compact step-by-step guide, see [RUN_LOCAL.md](https://github.com/zjrball/DataGenius/blob/main/RUN_LOCAL.md).

You can run this tool locally and ask other users to clone the repo and
run it with their own API keys. The app intentionally prefers a local
workflow to avoid accidental public key exposure.

Prerequisites
- Python 3.8 or higher
- A Google Gemini API Key

Minimal steps (Windows PowerShell)

1) Clone the repo
```powershell
git clone https://github.com/zjrball/DataGenius.git
cd DataGenius
```

2) Create and activate a virtual environment 

###### *Windows*
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

###### *Mac/Linux*
```
python3 -m venv .venv
source .venv/bin/activate
```

3) Install dependencies
```powershell
pip install -r .\requirements.txt
```

4) Provide your Gemini API key

Preferred (temporary session variable):
```powershell
$Env:GOOGLE_API_KEY = "your_gemini_key_here"
```

Or paste your API key into the sidebar when the app launches. The app
is designed to avoid using keys placed in `st.secrets` unless you opt-in
explicitly (this prevents accidentally exposing a hosted key in a demo).

Optional: local `secrets.toml` for convenience (only for local use)

Create a `.streamlit\secrets.toml` file with this content for local
development (do NOT commit this file):
```toml
GOOGLE_API_KEY = "your_gemini_key_here"
ACCESS_CODE = "demo123"  # optional: set an access code for demos
```

5) Run the app
```powershell
streamlit run .\app.py
```

Notes
- This project is intended to be run locally. For public demos, ask
    visitors to `git clone` and run with their own keys rather than
    hosting a deployment with a shared `GOOGLE_API_KEY`.
- The app will warn and require explicit consent before using any
    `st.secrets`-provided API key.
---

### Tech Stack
Frontend: Streamlit (Python-based UI)

AI Engine: Google Gemini 2.5 Flash (via google-generativeai)

AI Coding Assistants/Agent Usage:
* [GitHub Copilot](https://code.visualstudio.com/docs/copilot/overview)
* [Google Gemini](https://gemini.google.com/app)

Data Handling: Pandas

---

### 📂 Project Structure
```
DataGenius/
├── app.py                 # The main application logic
├── requirements.txt       # Python dependencies
├── .gitignore             # Security rules (hides secrets)
├── README.md              # Documentation
├── RUN_LOCAL.md           # Documentation
└── .streamlit/
    └── secrets.toml       # API Keys (Local only, NOT on GitHub)
```
--- 

### Prompt Engineering
A key part of this project was designing the prompts to ensure the AI generates structured CSV data rather than conversational text.

Challenge: LLMs love to chat.

Solution: I used strict system instructions and temperature controls to force the model to output purely valid CSV strings.

Privacy by Design: The prompts explicitly instruct the model to generate fictional entities, ensuring no real PII is ever leaked into the output.

---

### Contributing
Contributions are welcome! If you have ideas for new "Messy Data" types (e.g., specific SQL injection errors or encoding issues), feel free to fork the repo and submit a Pull Request.

Fork the Project

Create your Feature Branch (git checkout -b feature/NewMessiness)

Commit your Changes (git commit -m 'Add new messiness type')

Push to the Branch (git push origin feature/NewMessiness)

Open a Pull Request

---

### Acknowledgments
Google Gemini: This project was co-coded with the assistance of Google Gemini. It acted as a "Pair Programmer," helping to generate the initial Streamlit architecture, debug Python errors, and refine the data generation prompts.

Streamlit Community: For providing the framework that makes deploying Python data tools effortless.

---

### License
Distributed under the MIT License. See LICENSE for more information.

---

Built with 💻 and ☕ by Zachary Ball
