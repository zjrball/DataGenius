Run DataGenius locally — Quick guide

This short guide explains, in simple language, how to run the DataGenius Streamlit app on your Windows computer (PowerShell). Follow the steps below.

**Prerequisites**
- Python 3.10+ installed.
- A Gemini API key (Gemini/Google Generative AI). If you don't have one, the app can't call the model.
- Git (optional) if you cloned this repository.

**1) Create and activate a virtual environment (recommended)**
Open PowerShell in the repository root, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by PowerShell policy, run once as admin:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**2) Install dependencies**

```powershell
pip install -r .\requirements.txt
```

**3) Provide your Gemini API key**
You can supply the key in one of two ways:

A) Temporary environment variable (recommended for quick runs):
```powershell
$Env:GOOGLE_API_KEY = "YOUR_GEMINI_KEY_HERE"
streamlit run .\app.py
```

B) Paste into the app sidebar after starting Streamlit (the app will prompt you in the sidebar):
```powershell
streamlit run .\app.py
```
Then paste your key into the "Gemini API Key" field in the sidebar.

Notes:
- Do NOT commit your API key to the repository. Treat it like a password.
- If you prefer a local secrets file (for development only), create `.streamlit\secrets.toml` with:

```toml
# .streamlit/secrets.toml (DO NOT commit)
GOOGLE_API_KEY = "your_key_here"
```
Make sure `.gitignore` excludes `.streamlit/secrets.toml`.

**4) Quick usage**
- Use the sidebar to pick an Industry Preset or edit the schema.
- Click `Auto-Fill` to let the model propose fields (trimmed to 10 suggestions). Edit fields in the editor if needed.
- Click `👁️ Preview (10 rows)` to generate a 10-row preview and validate formatting.
- Click `🚀 Generate Data` to create the full dataset (uses the Row Count slider).
- When results appear, use the `Download CSV` button to save sanitized CSV to your machine.

Security & safety notes (simple):
- The app sanitizes CSV values that could be treated as spreadsheet formulas (values starting with `=`, `+`, `-`, `@`) by prefixing them with a single quote. This helps prevent accidental execution when opening CSVs in Excel/LibreOffice.
- The app was intentionally simplified for local demos: it does not persist telemetry, and it expects you to run it with your own key.

Troubleshooting (common issues)
- "API Key required" shown in the app: Set `$Env:GOOGLE_API_KEY` before running or paste the key in the sidebar.
- Streamlit fails to start: Make sure your virtual environment is active and dependencies are installed.
- PowerShell execution policy blocks activation: run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` as admin.
- Generated CSV parsing errors: Try `Preview` to refine the schema; AI output can sometimes be malformed. Edit the schema or reduce complexity.
- If the app shows an exception, it's visible in the UI (this is expected in local dev mode).

Optional developer steps
- Linting with `ruff`:
```powershell
.\.venv\Scripts\python.exe -m pip install ruff
.\.venv\Scripts\python.exe -m ruff check .\app.py
```
- Run tests (if added): `python -m pytest`

If something doesn't work, tell me what error you see and I can help debug.

---
File: `RUN_LOCAL.md` — short, clear instructions to run DataGenius locally. Keep your API key secret.