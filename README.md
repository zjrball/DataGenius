# ⚡ DataGenius: Synthetic Data Generator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://datageniuses.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gemini API](https://img.shields.io/badge/AI-Google%20Gemini-orange)

> **Simulate reality, safely.** DataGenius uses AI to generate realistic datasets on demand, solving privacy risks while giving you full control over data quality (Clean vs. Dirty).

<img width="1893" height="731" alt="Screenshot 2025-12-09 081843" src="https://github.com/user-attachments/assets/3731a2f3-ba8d-4d0e-a9fb-29cd7adcce64" />

## 📖 About The Project

Data Analysts and Engineers often face a dilemma: **Real data is sensitive, but public datasets are too simple.**

1.  **The Privacy Problem:** You can't just use production customer data for testing or portfolio projects due to PII (Personally Identifiable Information) and privacy laws like GDPR/HIPAA.
2.  **The Quality Problem:** Tutorial datasets are often "clean," but real-world data is full of errors.

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

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR-USERNAME/gaming-data-generator.git](https://github.com/YOUR-USERNAME/gaming-data-generator.git)
cd gaming-data-generator
