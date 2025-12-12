"""
Project: DataGenius
Author: Zachary Ball
Co-Authored By: Google Gemini, Claude AI, and Copilot(AI Assistants)

Description:
This application was developed with AI assistance to demonstrate modern 
rapid-prototyping workflows. While the code logic was augmented by AI, 
all outputs have been manually reviewed and tested for accuracy.
"""

import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import io
import time
import os
from datetime import datetime

# --- Configuration ---
st.set_page_config(
    page_title="DataGenius",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# For local-only usage we do not include a password gate. Keep the UI
# minimal so users can clone and run the app locally with their own key.

# Styling to match the dark aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #050b14;
        color: #e2e8f0;
    }
    .stButton button {
        background: linear-gradient(to right, #7c3aed, #4f46e5);
        color: white;
        border: none;
        font-weight: bold;
    }
    .stDataFrame {
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# --- API Key Handling (Local-first) ---
# This app is intended for local use. Prefer an environment variable
# (`GOOGLE_API_KEY`) or pasting a personal key into the sidebar. Avoid
# relying on framework-backed hosted keys when sharing the repository.
api_key = os.getenv("GOOGLE_API_KEY")

# Ask the user to paste their personal key if not provided via env var.
if not api_key:
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Paste your Gemini API key here for local use.",
    )

# Configure the Google Generative AI client with the provided key.
if api_key:
    genai.configure(api_key=api_key)

# In local mode we display exceptions to help developers iterate quickly.
# Note: telemetry/timing persistence was intentionally removed; the app
# does not write usage logs or telemetry to disk.

# --- Helpers: model cache and schema validation ---
# Cache model instance to avoid re-instantiation across calls.
_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = genai.GenerativeModel('gemini-2.5-flash')
    return _MODEL


def validate_schema(schema):
    """Validate the AI-generated schema structure.

    Returns (True, None) if valid, or (False, message) if invalid.
    Expected schema: list[dict] with keys 'Field Name','Type','Context'.
    'Type' must be one of FIELD_TYPES. 'Messiness' is optional.
    """
    if not isinstance(schema, list):
        return False, "Schema must be a JSON array of field objects."
    if len(schema) == 0:
        return False, "Schema is empty."
    for idx, item in enumerate(schema):
        if not isinstance(item, dict):
            return False, f"Schema item at index {idx} is not an object."
        for key in ["Field Name", "Type", "Context"]:
            if key not in item:
                return False, f"Missing key '{key}' in schema item at index {idx}."
        if not isinstance(item["Field Name"], str) or item["Field Name"].strip() == "":
            return False, f"Invalid Field Name at index {idx}."
        if item["Type"] not in FIELD_TYPES:
            return False, f"Invalid Type '{item['Type']}' at index {idx}. Allowed: {', '.join(FIELD_TYPES)}"
        # Add Messiness key if missing (default to None for no messiness)
        if "Messiness" not in item:
            item["Messiness"] = None
        # Add Messiness % if missing (default to 0)
        if "Messiness %" not in item:
            item["Messiness %"] = 0
    return True, None

# --- Limits ---
# Keep generation limits conservative by default; main cap is `MAX_FIELDS`.

# --- Constants ---
# `MAX_FIELDS` is a hard cap to prevent large schemas that drive up token
# usage and generation cost. Enforced both in the UI and at generation time.
MAX_FIELDS = 15

# Supported type options presented to users in the editor. Keep this list
# in sync with the instructions sent to the AI model so it generates only
# allowed types.
FIELD_TYPES = [
    "String", "Number", "Date", "Boolean", "Currency", 
    "UUID", "Email", "Name", "Category", "Status", "Timestamp"
]

# Messiness options by field type - users can choose specific quality issues per field
MESSINESS_OPTIONS = {
    "Date": ["NaN", "Blank", "Different Formats", "Duplicates", "Future Dates"],
    "Timestamp": ["NaN", "Blank", "Different Formats", "Duplicates", "Timezone Issues"],
    "String": ["NaN", "Blank", "Typos", "Mixed Case", "Extra Whitespace", "Special Characters", "Duplicates", "Extra Values", "Different Formats"],
    "Number": ["NaN", "Blank", "Outliers", "Negative Values", "Decimals as Text"],
    "Currency": ["NaN", "Blank", "Missing Symbols", "Wrong Decimals", "Negative Values"],
    "Boolean": ["NaN", "Blank", "Text Values (Yes/No)", "0/1 Instead"],
    "Email": ["NaN", "Blank", "Invalid Format", "Missing @", "Typos"],
    "Name": ["NaN", "Blank", "Typos", "All Caps", "Numbers in Name", "Duplicates"],
    "Category": ["NaN", "Blank", "Typos", "Inconsistent Labels", "Extra Values"],
    "Status": ["NaN", "Blank", "Typos", "Inconsistent Labels"],
    "UUID": ["NaN", "Blank", "Invalid Format", "Duplicates"],
}

# Pre-generated AI suggestion prompts for quick access
AI_SUGGESTIONS = [
    "Restaurant order history with menu items, prices, and timestamps",
    "Employee performance reviews with ratings, feedback, and salary changes",
    "Social media posts with engagement metrics, hashtags, and timestamps",
    "Inventory management with stock levels, reorder points, and suppliers",
    "Customer support tickets with priority, status, and resolution time",
    "Real estate listings with property details, prices, and locations",
    "Fitness tracker data with workouts, calories, and heart rate",
    "Online course enrollments with students, grades, and completion status",
    "Vehicle fleet maintenance with service records, costs, and mileage",
    "Project task management with assignments, deadlines, and progress"
]

PRESETS = {
    "E-commerce": [
        {"Field Name": "Transaction ID", "Type": "UUID", "Context": "Unique identifier", "Messiness": None, "Messiness %": 0},
        {"Field Name": "Customer", "Type": "Name", "Context": "First and last name", "Messiness": None, "Messiness %": 0},
        {"Field Name": "Product", "Type": "Category", "Context": "Electronics, Home, Fashion", "Messiness": None, "Messiness %": 0},
        {"Field Name": "Amount", "Type": "Currency", "Context": "Min: 10, Max: 500", "Messiness": None, "Messiness %": 0},
        {"Field Name": "Order Date", "Type": "Date", "Context": "2020-2024, realistic distribution", "Messiness": None, "Messiness %": 0},
    ],
    "Healthcare": [
        {"Field Name": "Patient ID", "Type": "UUID", "Context": "Unique identifier", "Messiness": None, "Messiness %": 0},
        {"Field Name": "Diagnosis", "Type": "String", "Context": "ICD-10 codes", "Messiness": None, "Messiness %": 0},
        {"Field Name": "Admission Date", "Type": "Date", "Context": "Last 12 months", "Messiness": None, "Messiness %": 0},
        {"Field Name": "Bill", "Type": "Currency", "Context": "Min: 500, Max: 50000", "Messiness": None, "Messiness %": 0},
    ],
    "Finance": [
        {"Field Name": "Account", "Type": "UUID", "Context": "Unique identifier", "Messiness": None, "Messiness %": 0},
        {"Field Name": "Transaction Type", "Type": "Category", "Context": "Debit, Credit, Transfer", "Messiness": None, "Messiness %": 0},
        {"Field Name": "Amount", "Type": "Currency", "Context": "Min: 50, Max: 5000", "Messiness": None, "Messiness %": 0},
        {"Field Name": "Transaction Date", "Type": "Timestamp", "Context": "Last 3 years", "Messiness": None, "Messiness %": 0},
        {"Field Name": "Fraud Flag", "Type": "Boolean", "Context": "1% True", "Messiness": None, "Messiness %": 0},
    ],
    "CRM (Messy Data Practice)": [
        {"Field Name": "Lead ID", "Type": "UUID", "Context": "Unique identifier", "Messiness": "Blank", "Messiness %": 5},
        {"Field Name": "Company Name", "Type": "String", "Context": "Business names", "Messiness": "Duplicates", "Messiness %": 15},
        {"Field Name": "Contact Name", "Type": "Name", "Context": "First and last name", "Messiness": "Duplicates", "Messiness %": 10},
        {"Field Name": "Email", "Type": "Email", "Context": "Business emails", "Messiness": "Invalid Format", "Messiness %": 20},
        {"Field Name": "Phone", "Type": "String", "Context": "10-digit US format", "Messiness": "Different Formats", "Messiness %": 25},
        {"Field Name": "Lead Status", "Type": "Status", "Context": "New, Contacted, Qualified, Lost", "Messiness": "Inconsistent Labels", "Messiness %": 15},
        {"Field Name": "Deal Value", "Type": "Currency", "Context": "Min: 1000, Max: 100000", "Messiness": "NaN", "Messiness %": 10},
        {"Field Name": "Last Contact Date", "Type": "Date", "Context": "Last 6 months", "Messiness": "Blank", "Messiness %": 20},
    ],
    "Education (Messy Data Practice)": [
        {"Field Name": "Student ID", "Type": "UUID", "Context": "Unique identifier", "Messiness": "Duplicates", "Messiness %": 5},
        {"Field Name": "Student Name", "Type": "Name", "Context": "First and last name", "Messiness": "All Caps", "Messiness %": 30},
        {"Field Name": "Email", "Type": "Email", "Context": "School email addresses", "Messiness": "Typos", "Messiness %": 15},
        {"Field Name": "Grade Level", "Type": "Category", "Context": "Freshman, Sophomore, Junior, Senior", "Messiness": "Typos", "Messiness %": 10},
        {"Field Name": "GPA", "Type": "Number", "Context": "Min: 0.0, Max: 4.0", "Messiness": "NaN", "Messiness %": 15},
        {"Field Name": "Enrollment Date", "Type": "Date", "Context": "Last 4 years", "Messiness": "Different Formats", "Messiness %": 20},
        {"Field Name": "Active Status", "Type": "Boolean", "Context": "90% True", "Messiness": "Text Values (Yes/No)", "Messiness %": 25},
        {"Field Name": "Major", "Type": "String", "Context": "STEM, Business, Liberal Arts, etc", "Messiness": "Extra Whitespace", "Messiness %": 20},
    ],
    "Custom": [
        {"Field Name": "ID", "Type": "UUID", "Context": "", "Messiness": None, "Messiness %": 0},
    ]
}

# --- State Management ---
# Use `st.session_state` so schema and generated data persist across
# reruns. Initialize with a sensible default preset when first run.
if "fields_df" not in st.session_state:
    st.session_state.fields_df = pd.DataFrame(PRESETS["E-commerce"])

if "generated_data" not in st.session_state:
    st.session_state.generated_data = None

if "ai_prompt_text" not in st.session_state:
    st.session_state.ai_prompt_text = ""

if "last_generated" not in st.session_state:
    st.session_state.last_generated = None

# --- Logic ---

def get_context_hint(field_type):
    """Return helpful context examples based on field type."""
    hints = {
        "Date": "💡 Examples: 'Last 5 years', '2020-2024', 'Last 6 months'",
        "Timestamp": "💡 Examples: 'Last 3 years', '2020-2024 with time'",
        "Number": "💡 Examples: 'Min: 1, Max: 100', 'Between 0 and 1000'",
        "Currency": "💡 Examples: 'Min: 10, Max: 500', '$2-$8'",
        "Category": "💡 Examples: 'Electronics, Home, Fashion' (specific options)",
        "Boolean": "💡 Examples: '90% True', '10% False', Binary values",
        "Email": "💡 Examples: 'Business emails', 'Gmail domains'",
        "Name": "💡 Examples: 'First and last name', 'Full names'",
        "Status": "💡 Examples: 'Active, Pending, Closed' (specific statuses)",
        "UUID": "💡 Examples: 'Unique identifier', 'Auto-generated IDs'",
        "String": "💡 Examples: 'Short description', 'Max 50 characters'",
    }
    return hints.get(field_type, "💡 Provide a description for this field")

def get_preset_description(preset_name):
    """Return a brief description of what each preset contains."""
    descriptions = {
        "E-commerce": "🛒 Transaction ID, Customer, Product, Amount, Order Date (5 clean fields)",
        "Healthcare": "🏥 Patient ID, Diagnosis, Admission Date, Bill (4 clean fields)",
        "Finance": "💰 Account, Transaction Type, Amount, Date, Fraud Flag (5 clean fields)",
        "CRM (Messy Data Practice)": "👥 Lead ID, Company, Contact, Email, Phone, Status, Deal Value, Last Contact (8 messy fields)",
        "Education (Messy Data Practice)": "🎓 Student ID, Name, Email, Grade, GPA, Enrollment, Status, Major (8 messy fields)",
        "Custom": "⚙️ Start with a single ID field and customize",
    }
    return descriptions.get(preset_name, "")

def calculate_eta(row_count, field_count=1):
    """Return a human-friendly ETA string for generation.

    This function uses a deterministic formula rather than persisted
    telemetry. The constants (base_time, sec_per_row, field_time) are
    conservative heuristics intended to provide useful feedback without
    storing any local timing data.
    """
    base_time = 3  # Fixed startup overhead in seconds
    # Per-row cost (seconds). This is a heuristic derived from earlier
    # experiments and chosen to be conservative for public releases.
    sec_per_row = 1.26
    row_time = row_count * sec_per_row
    # Add a small per-field penalty to reflect increased prompt complexity
    field_time = field_count * 0.5
    total_seconds = int(base_time + row_time + field_time)

    if total_seconds < 60:
        return f"{total_seconds}s"
    else:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"


def _sanitize_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize dataframe values to mitigate CSV injection attacks.

    Many spreadsheet programs (Excel, LibreOffice) will execute formulas
    in cells that start with characters like '=', '+', '-', or '@'. If
    synthetic data contains attacker-controlled values beginning with
    those characters, a malicious CSV could trigger unsafe behavior when
    opened in a spreadsheet application. To mitigate, prefix such values
    with a single quote so they're treated as text.
    """
    df_safe = df.copy()

    # Characters that can trigger formula evaluation in spreadsheets.
    dangerous_prefixes = ("=", "+", "-", "@")

    for col in df_safe.columns:
        # Only process object/string columns; leave numeric types alone.
        if pd.api.types.is_object_dtype(df_safe[col]) or pd.api.types.is_string_dtype(df_safe[col]):
            def _sanitize_cell(val):
                try:
                    if val is None:
                        return val
                    s = str(val)
                    if s.startswith(dangerous_prefixes):
                        return "'" + s
                    return s
                except Exception:
                    # In case of unexpected types, return the original value.
                    return val

            df_safe[col] = df_safe[col].apply(_sanitize_cell)

    return df_safe

def generate_schema_from_prompt(prompt_text, num_fields=5):
    """Ask the model to propose a small schema from a short description.

    Notes:
    - The model may return code fences or additional commentary; we strip
      common markdown wrappers before attempting to parse JSON.
    - The num_fields parameter allows users to specify how many fields to generate.
    """
    if not api_key:
        st.error("API Key required")
        return
    
    if not prompt_text or prompt_text.strip() == "":
        st.error("Please describe your dataset before using Auto-Fill")
        return

    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🤖 Analyzing prompt... (10%)")
        progress_bar.progress(10)
        time.sleep(0.3)
        
        status_text.text("🧠 Generating schema... (40%)")
        progress_bar.progress(40)
        
        model = get_model()
        # System prompt: be explicit and restrictive. In many cases the model
        # produces helpful output but can also include prose or wrappers; we
        # instruct it to return only a JSON array so parsing is more reliable.
        sys_prompt = f"""
        You are a data architect. Generate a list of fields for a dataset described as: "{prompt_text}".
        IMPORTANT: Generate exactly {num_fields} fields (not more, not less).
        Return ONLY valid JSON array of objects with keys: "Field Name", "Type", "Context".
        Allowed Types: {", ".join(FIELD_TYPES)}.
        
        Context should include:
        - For Date/Timestamp fields: Specify realistic ranges (e.g., "Last 5 years", "2020-2024", "Last 6 months")
        - For Number/Currency: Specify min/max that make sense for the domain (e.g., coffee prices $2-8, not $1-1000)
        - For Category: List specific realistic options (e.g., "Latte, Cappuccino, Espresso" not just "Coffee types")
        - For related fields: Ensure they can correlate logically (e.g., order dates should be older than delivery dates)
        """
        try:
            # Request content from the model. Capture the raw text so we can
            # show a truncated preview when parsing fails (only when debug
            # mode is enabled).
            response = model.generate_content(sys_prompt)
            raw_text = response.text
            
            status_text.text("📝 Parsing JSON... (70%)")
            progress_bar.progress(70)
            time.sleep(0.2)
            
            # Model may include markdown code fences like ```json```; strip them.
            clean_json = raw_text.replace("```json", "").replace("```", "").strip()
            try:
                schema = json.loads(clean_json)
            except json.JSONDecodeError as jde:
                # Parsing failed: show a truncated sample and the JSON error
                # so local developers can quickly iterate on prompts.
                progress_bar.empty()
                status_text.empty()
                sample = (raw_text[:1000] + "...") if len(raw_text) > 1000 else raw_text
                st.error("Failed to parse model JSON output. Sample below (truncated):")
                st.code(sample)
                st.exception(jde)
                return

            status_text.text("✔️ Validating schema... (85%)")
            progress_bar.progress(85)
            time.sleep(0.2)
            
            # Validate schema structure before accepting it. This prevents
            # malformed or unsafe schema objects from being written into
            # session state.
            valid, msg = validate_schema(schema)
            if not valid:
                # Show the failing reason and a short sample to help local
                # debugging and prompt tuning.
                progress_bar.empty()
                status_text.empty()
                sample = (raw_text[:1000] + "...") if len(raw_text) > 1000 else raw_text
                st.error(f"Schema validation failed: {msg}")
                st.code(sample)
                return

            # Safety: trim overly large AI suggestions to requested num_fields.
            if len(schema) > num_fields:
                st.warning(f"⚠️ AI suggested {len(schema)} fields. Trimming to {num_fields} fields as requested.")
                schema = schema[:num_fields]
            
            status_text.text("✔️ Finalizing... (95%)")
            progress_bar.progress(95)
            time.sleep(0.2)

            # Validate trimmed schema as well (in case trimming removed valid
            # fields but left invalid items).
            valid_trim, msg_trim = validate_schema(schema)
            if not valid_trim:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Trimmed schema invalid: {msg_trim}")
                return

            # Record suggested count in-memory only.
            
            status_text.text("✅ Schema ready! (100%)")
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            # Persist the proposed schema to session state and force a
            # rerun so the editor UI reflects the new rows immediately.
            st.session_state.fields_df = pd.DataFrame(schema)
            st.rerun()
        except Exception as e:
            # Network errors or model errors may surface here. Show traceback
            # for local debugging.
            st.exception(e)
            st.error("Failed to generate schema: model request failed.")


def preview_dataset(industry, fields_data):
    """Generate a fast, small preview to validate schema & prompts.

    Important implementation notes:
    - We enforce `MAX_FIELDS` here to avoid expensive previews when a
      schema is accidentally oversized.
    - The AI-generated CSV can be malformed (extra commas, inconsistent
      quoting, or stray text). To make parsing robust we use
      `pd.read_csv(..., on_bad_lines='skip', engine='python')` which will
      skip rows that cannot be parsed instead of crashing the app.
    """
    if not api_key:
        st.error("API Key required")
        return

    # Enforce field limit
    if len(fields_data) > MAX_FIELDS:
        st.error(f"❌ Maximum {MAX_FIELDS} fields allowed. You have {len(fields_data)} fields.")
        return

    model = get_model()
    
    # Build field descriptions with messiness instructions
    field_desc = []
    messiness_instructions = []
    for _, row in fields_data.iterrows():
        field_desc.append(f"{row['Field Name']} ({row['Type']}): {row['Context']}")
        messiness = row.get('Messiness', None)
        if messiness and pd.notna(messiness):
            messiness_pct = row.get('Messiness %', 15)
            messiness_instructions.append(f"  - {row['Field Name']}: Apply '{messiness}' ({messiness_pct}% of rows)")
    
    quality_section = ""
    if messiness_instructions:
        quality_section = "\n\nQUALITY ISSUES TO INJECT:\n" + "\n".join(messiness_instructions)
    else:
        quality_section = "\n\nAll fields should be Clean with standard formats and no nulls."

    prompt = f"""
    Generate a CSV dataset for '{industry}'.
    Rows: 10

    Columns (exactly {len(field_desc)} fields):
    {'; '.join(field_desc)}
    {quality_section}

    IMPORTANT DATA REALISM RULES:
    - Dates/Timestamps: Keep within specified ranges. If range not specified, use last 2-5 years.
    - Related fields must be logically consistent (e.g., order date < delivery date, hire date < termination date).
    - Categorical values should reflect real-world distributions (e.g., popular items appear more often).
    - Numeric ranges should respect Context constraints and industry norms.
    - Apply messiness types ONLY to specified fields at their exact specified percentage.
    
    CRITICAL: Output ONLY valid CSV with exactly {len(field_desc)} columns per row. Include headers. No markdown formatting.
    """

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Request preview from the model and aggressively strip common
        # markdown wrappers. We still protect parsing with pandas.
        status_text.text("🤖 Generating 10-row preview... (20%)")
        progress_bar.progress(20)
        response = model.generate_content(prompt)
        csv_text = response.text.replace("```csv", "").replace("```", "").strip()

        status_text.text("📊 Parsing CSV data... (60%)")
        progress_bar.progress(60)

        # Parse and display with error handling
        try:
            # `on_bad_lines='skip'` prevents a single malformed AI row from
            # crashing the whole preview. We surface an error if the final
            # parsed frame is empty to guide the user to improve prompts.
            df = pd.read_csv(io.StringIO(csv_text), on_bad_lines='skip', engine='python')
            if len(df) == 0:
                st.error("Preview failed: No valid rows generated. Please try again.")
                return

            status_text.text("🔒 Sanitizing data... (80%)")
            progress_bar.progress(80)

            # Sanitize the resulting DataFrame to mitigate CSV injection
            # before storing and presenting it to the user or offering a
            # download.
            df_safe = _sanitize_dataframe_for_csv(df)
            st.session_state.generated_data = df_safe
            
            # Complete UI update
            progress_bar.progress(100)
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()
            # Force UI refresh so the Results panel appears with the preview.
            st.rerun()
        except Exception as parse_err:
            progress_bar.empty()
            status_text.empty()
            st.exception(parse_err)
            st.error("Preview failed: CSV parsing error. The AI output may be malformed — try refining your schema.")
            return

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.exception(e)
        st.error("Preview failed: model request failed. Check API key and network.")

def generate_dataset(industry, rows, fields_data):
    """Generate the full dataset according to the user's schema.

    The implementation mirrors `preview_dataset` but generates `rows` rows.
    Key safety checks are repeated here to avoid accidental large token
    consumption when a schema exceeds `MAX_FIELDS`.
    """
    if not api_key:
        st.error("API Key required")
        return
    
    # Enforce field limit
    if len(fields_data) > MAX_FIELDS:
        st.error(f"❌ Maximum {MAX_FIELDS} fields allowed to manage token usage. You have {len(fields_data)} fields.")
        return

    # No rate limiting in local mode; local users are expected to manage their own API usage.

    model = get_model()
    
    # Build field descriptions with messiness instructions
    field_desc = []
    messiness_instructions = []
    for _, row in fields_data.iterrows():
        field_desc.append(f"{row['Field Name']} ({row['Type']}): {row['Context']}")
        messiness = row.get('Messiness', None)
        if messiness and pd.notna(messiness):
            messiness_pct = row.get('Messiness %', 15)
            messiness_instructions.append(f"  - {row['Field Name']}: Apply '{messiness}' ({messiness_pct}% of rows)")
    
    quality_section = ""
    if messiness_instructions:
        quality_section = "\n\nQUALITY ISSUES TO INJECT:\n" + "\n".join(messiness_instructions)
    else:
        quality_section = "\n\nAll fields should be Clean with standard formats and no nulls."

    prompt = f"""Generate EXACTLY {rows} CSV rows for '{industry}' (not more, not less).

Columns: {'; '.join(field_desc)}
{quality_section}

IMPORTANT DATA REALISM:
- Dates/Timestamps: Stay within specified ranges. If unspecified, use last 2-5 years.
- Related fields must be logically consistent (order date < ship date, start < end, etc.).
- Categories should follow realistic distributions (common items more frequent).
- Numbers must respect Context min/max and industry standards.
- For time-series data, ensure natural temporal patterns (weekday peaks for B2B, weekend for retail, etc.).
- Apply messiness types ONLY to specified fields at their exact specified percentage.

CRITICAL: Generate EXACTLY {rows} data rows (excluding header). Do not generate more or fewer rows.
Output: CSV with headers only, no markdown."""

    # UI affordances: progress and status improve perceived performance.
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 1. Generate Data: request CSV text from the model.
        status_text.text(f"🤖 Generating {rows} rows with {len(field_desc)} fields... (30%)")
        progress_bar.progress(30)
        # Local generation request
        response = model.generate_content(prompt)
        csv_text = response.text.replace("```csv", "").replace("```", "").strip()

        status_text.text("📊 Parsing CSV data... (60%)")
        progress_bar.progress(60)

        # 2. Parse and Store: robust parsing as in preview (skip bad lines).
        try:
            df = pd.read_csv(io.StringIO(csv_text), on_bad_lines='skip', engine='python')
            if len(df) == 0:
                st.error("Generation failed: No valid rows produced. Try adjusting your field descriptions.")
                return

            # Enforce exact row count by truncating or warning
            if len(df) > rows:
                st.warning(f"⚠️ AI generated {len(df)} rows instead of {rows}. Truncating to requested count.")
                df = df.head(rows)
            elif len(df) < rows:
                st.warning(f"⚠️ AI generated only {len(df)} rows instead of {rows}.")

            status_text.text("🔒 Sanitizing data... (80%)")
            progress_bar.progress(80)

            # Sanitize before storing/downloading to mitigate CSV injection.
            df_safe = _sanitize_dataframe_for_csv(df)
            st.session_state.generated_data = df_safe
            # Generation succeeded (local only)
        except Exception as parse_err:
            # Provide a clear message and reveal the exception for local debugging.
            st.exception(parse_err)
            st.error("Generation failed: CSV parsing error. The AI may have generated inconsistent data or used unexpected formatting.")
            return

        # Complete UI update and force a rerun so the Results are rendered.
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        # Record generation timestamp
        st.session_state.last_generated = datetime.now()
        st.rerun()

    except Exception as e:
        st.exception(e)
        st.error("Generation failed: model request failed. Check API key and network.")

# --- UI Layout ---

# Sidebar
with st.sidebar:
    st.title("DataGenius ⚡")
    
    # Preset Preview Cards
    with st.expander("📋 Preset Previews", expanded=False):
        for preset_name in PRESETS.keys():
            description = get_preset_description(preset_name)
            st.markdown(f"**{preset_name}**\n`{description}`")
    
    selected_preset = st.selectbox(
        "Industry Preset", 
        list(PRESETS.keys()),
        index=0
    )
    
    if st.button("Load Preset"):
        st.session_state.fields_df = pd.DataFrame(PRESETS[selected_preset])
        st.rerun()
    
    st.divider()
    
    row_count = st.slider("Row Count", 10, 100, 50)
    
    # Display ETA
    eta = calculate_eta(row_count, len(st.session_state.fields_df))
    st.caption(f"⏱️ Estimated time: {eta}")
    
    st.divider()
    
    # Developer aid: exceptions are shown in the UI for local debugging.
    preview_btn = st.button("👁️ Preview (10 rows)", use_container_width=True)
    generate_btn = st.button("🚀 Generate Data", use_container_width=True, type="primary")
    
    st.divider()
    
    # Watermark / Credits
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.85em;'>
            Designed by <a href='https://www.linkedin.com/in/zacharyjball/' target='_blank' style='text-decoration: none;'>Zachary Ball</a><br>
            <a href='https://github.com/zjrball' target='_blank' style='text-decoration: none;'>GitHub</a> • 
            <a href='https://www.linkedin.com/in/zacharyjball/' target='_blank' style='text-decoration: none;'>LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Content
st.header("Schema Configuration")

# AI Auto-Fill Section
with st.expander("✨ AI Auto-Fill Schema"):
    # Quick suggestions at the top
    st.caption("💡 Quick Ideas:")
    col_sug1, col_sug2 = st.columns(2)
    suggestion_1 = AI_SUGGESTIONS[0]
    suggestion_2 = AI_SUGGESTIONS[1]
    with col_sug1:
        if st.button(f"📋 {suggestion_1[:40]}...", key="sug1", use_container_width=True):
            st.session_state.ai_prompt_text = suggestion_1
            st.rerun()
    with col_sug2:
        if st.button(f"📋 {suggestion_2[:40]}...", key="sug2", use_container_width=True):
            st.session_state.ai_prompt_text = suggestion_2
            st.rerun()
    
    col_ai_1, col_ai_2, col_ai_3 = st.columns([4, 1, 1])
    with col_ai_1:
        ai_prompt = st.text_input("Describe your dataset", value=st.session_state.ai_prompt_text, placeholder="e.g. 'Startup employee roster with equity and vesting schedules'")
    with col_ai_2:
        num_fields = st.number_input("# Fields", min_value=1, max_value=MAX_FIELDS, value=5, step=1, help="Number of fields to generate")
    with col_ai_3:
        st.write("") # Spacer
        st.write("") 
        if st.button("Auto-Fill"):
            generate_schema_from_prompt(ai_prompt, num_fields)

# Field Editor with limit indicator
field_count = len(st.session_state.fields_df)
field_limit_color = "🟢" if field_count <= MAX_FIELDS else "🔴"
st.caption(f"{field_limit_color} Fields: {field_count}/{MAX_FIELDS}")

# Quick Actions
st.caption("⚡ Quick Actions:")
col_qa1, col_qa2, col_qa3 = st.columns(3)
with col_qa1:
    if st.button("🧹 Clear Messiness", help="Reset all fields to clean data (0% messiness)", key="clear_messiness"):
        for idx in range(len(st.session_state.fields_df)):
            st.session_state.fields_df.at[idx, 'Messiness'] = None
            st.session_state.fields_df.at[idx, 'Messiness %'] = 0
        st.success("✅ Clearing all fields to clean...")
        time.sleep(1)
        st.rerun()
with col_qa2:
    if st.button("📊 Add 10% Messiness", help="Apply 10% random messiness to all fields", key="add_messiness"):
        for idx, row in st.session_state.fields_df.iterrows():
            field_type = row.get('Type', 'String')
            valid_options = MESSINESS_OPTIONS.get(field_type, [])
            current_messiness = row.get('Messiness')
            current_pct = row.get('Messiness %', 0) or 0
            
            # Only add messiness if field doesn't already have a messiness type set
            if not current_messiness or pd.isna(current_messiness) or current_messiness == '':
                if valid_options and len(valid_options) > 0:
                    st.session_state.fields_df.at[idx, 'Messiness'] = valid_options[0]
                    st.session_state.fields_df.at[idx, 'Messiness %'] = 10
            else:
                # If already has a messiness type, just increase percentage by 10% (capped at 100%)
                new_pct = min(current_pct + 10, 100)
                st.session_state.fields_df.at[idx, 'Messiness %'] = new_pct
        st.success("✅ Adding 10% messiness to all fields...")
        time.sleep(1)
        st.rerun()
with col_qa3:
    if st.button("🔄 Reset to Preset", help="Reload the current preset", key="reset_preset"):
        st.session_state.fields_df = pd.DataFrame(PRESETS[selected_preset])
        st.success(f"✅ Resetting to {selected_preset} preset...")
        time.sleep(1)
        st.rerun()

# Validation Status Summary
if field_count > 0:
    messiness_count = len(st.session_state.fields_df[st.session_state.fields_df['Messiness'].notna()])
    clean_count = field_count - messiness_count
    status_line = f"✅ {clean_count} clean | ⚠️ {messiness_count} with quality issues"
    st.caption(status_line)

# Show messiness options guide
with st.expander("ℹ️ Messiness Options by Field Type"):
    st.markdown("""
    **Messiness Percentage**: Set 0-100% for each field. Higher percentages = more messy data.
    
    ### Data Types & Valid Messiness Options:
    
    📅 **Date**: NaN, Blank, Different Formats, Duplicates, Future Dates
    
    ⏰ **Timestamp**: NaN, Blank, Different Formats, Duplicates, Timezone Issues
    
    📝 **String**: NaN, Blank, Typos, Mixed Case, Extra Whitespace, Special Characters, Duplicates, Extra Values, Different Formats
    
    🔢 **Number**: NaN, Blank, Outliers, Negative Values, Decimals as Text
    
    💵 **Currency**: NaN, Blank, Missing Symbols, Wrong Decimals, Negative Values
    
    ✓ **Boolean**: NaN, Blank, Text Values (Yes/No), 0/1 Instead
    
    📧 **Email**: NaN, Blank, Invalid Format, Missing @, Typos
    
    👤 **Name**: NaN, Blank, Typos, All Caps, Numbers in Name, Duplicates
    
    🏷️ **Category**: NaN, Blank, Typos, Inconsistent Labels, Extra Values
    
    🔖 **Status**: NaN, Blank, Typos, Inconsistent Labels
    
    🆔 **UUID**: NaN, Blank, Invalid Format, Duplicates
    """)

# Ensure Messiness column exists in session state
if 'Messiness' not in st.session_state.fields_df.columns:
    st.session_state.fields_df['Messiness'] = None
if 'Messiness %' not in st.session_state.fields_df.columns:
    st.session_state.fields_df['Messiness %'] = 0

# Collect all unique messiness options across all field types
all_messiness_options = []
for options in MESSINESS_OPTIONS.values():
    for opt in options:
        if opt not in all_messiness_options:
            all_messiness_options.append(opt)

# Function to validate a single field
def get_field_validation_status(field_name, field_type, messiness, messiness_pct):
    """Return validation status icon and message for a field."""
    # Ensure field_name is a string
    field_name = str(field_name) if field_name and not pd.isna(field_name) else ""
    # Ensure field_type is a string
    field_type = str(field_type) if field_type and not pd.isna(field_type) else ""
    field_type = field_type.strip()
    
    if not field_name or field_name.strip() == "":
        return "⚠️", "Missing field name"
    if not field_type or field_type == "":
        return "⚠️", "Missing field type"
    
    # Ensure messiness is a string (it might be a list or other type from data_editor)
    if isinstance(messiness, list):
        messiness = messiness[0] if messiness else None
    
    # Handle pandas NA values
    if pd.isna(messiness):
        messiness = None
    elif messiness:
        messiness = str(messiness).strip()
    
    # Check if messiness is valid for the field type
    if messiness and messiness != "" and messiness != "None":
        # Find the correct field type key (case-insensitive)
        correct_field_type = None
        for key in MESSINESS_OPTIONS.keys():
            if key.lower() == field_type.lower():
                correct_field_type = key
                break
        
        if correct_field_type:
            valid_options = MESSINESS_OPTIONS[correct_field_type]
            # Case-insensitive comparison for safety
            messiness_match = any(m.lower() == messiness.lower() for m in valid_options)
            if not messiness_match:
                return "⚠️", f"'{messiness}' not valid for {field_type}"
        else:
            return "⚠️", f"Unknown field type: {field_type}"
    
    # If messiness is set but percentage is 0, that's a warning
    if messiness and messiness != "" and messiness != "None" and (messiness_pct is None or messiness_pct == 0):
        return "⚠️", "Messiness set but % is 0"
    
    # All valid
    return "✅", "Valid"

# Display validation summary before editor
st.subheader("Field Validation Status")
validation_data = []
for idx, row in st.session_state.fields_df.iterrows():
    status_icon, status_msg = get_field_validation_status(
        row.get("Field Name", ""),
        row.get("Type", ""),
        row.get("Messiness"),
        row.get("Messiness %", 0)
    )
    validation_data.append({
        "Row": idx + 1,
        "Status": status_icon,
        "Field Name": row.get("Field Name", ""),
        "Type": row.get("Type", ""),
        "Message": status_msg
    })

if validation_data:
    validation_df = pd.DataFrame(validation_data)
    # Display with custom styling - show only if there are warnings
    has_warnings = any("⚠️" in row["Status"] for row in validation_data)
    if has_warnings:
        st.dataframe(
            validation_df[["Row", "Status", "Field Name", "Type", "Message"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("✅ All fields are valid!")

# Clean up dataframe before editor: convert list values to scalar values
df_for_editor = st.session_state.fields_df.copy()
for col in df_for_editor.columns:
    df_for_editor[col] = df_for_editor[col].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
    )

edited_df = st.data_editor(
    df_for_editor,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Type": st.column_config.SelectboxColumn(
            "Type",
            options=FIELD_TYPES,
            required=True,
        ),
        "Context": st.column_config.TextColumn(
            "Context / Description",
            help="e.g. 'Min: 10, Max: 100' or 'Last 5 years'"
        ),
        "Messiness": st.column_config.SelectboxColumn(
            "Messiness",
            options=all_messiness_options,
            help="Choose quality issue for this field (options vary by Type)",
            required=False,
        ),
        "Messiness %": st.column_config.NumberColumn(
            "Messiness %",
            help="Percentage of rows with this quality issue (0-100%). Higher percentages create dirtier data.",
            min_value=0,
            max_value=100,
            step=5,
            format="%d%%",
        )
    }
)

# Only update session state if row count changed (rows added/deleted)
# Don't update on every cell edit to prevent refresh issues
if len(edited_df) != len(st.session_state.fields_df):
    st.session_state.fields_df = edited_df.copy()

# Check if messiness or type changed in any row to trigger rerun for validation update
messiness_changed = False
for idx, row in edited_df.iterrows():
    if idx < len(st.session_state.fields_df):
        old_row = st.session_state.fields_df.iloc[idx]
        # Use a safe comparison that handles pandas NA values
        row_messiness = row.get('Messiness')
        old_messiness = old_row.get('Messiness')
        row_type = row.get('Type')
        old_type = old_row.get('Type')
        
        # Compare safely by converting NA to None
        row_messiness_val = None if pd.isna(row_messiness) else row_messiness
        old_messiness_val = None if pd.isna(old_messiness) else old_messiness
        row_type_val = None if pd.isna(row_type) else row_type
        old_type_val = None if pd.isna(old_type) else old_type
        
        if row_messiness_val != old_messiness_val or row_type_val != old_type_val:
            messiness_changed = True
            break

if messiness_changed:
    st.session_state.fields_df = edited_df.copy()
    st.rerun()

# Update field count after processing
updated_field_count = len(st.session_state.fields_df)
if updated_field_count >= MAX_FIELDS:
    st.info(f"ℹ️ Field limit ({MAX_FIELDS}) reached. Remove fields to add more.")

# Use edited_df for generation (which includes all current edits)
working_df = edited_df

# Validation and processing function - only runs when generate/preview is clicked
def validate_and_process_schema(df):
    """Validate and clean up the schema before generation."""
    processed_df = df.copy()
    
    # Auto-number blank Field Names
    for idx, row in processed_df.iterrows():
        field_type = row.get('Type', 'String')
        current_messiness = row.get('Messiness', None)
        field_name = row.get('Field Name', '')
        
        # Auto-number blank Field Names
        if field_name == '' or pd.isna(field_name):
            processed_df.at[idx, 'Field Name'] = f"Field {idx + 1}"
        
        # Validate messiness is appropriate for field type
        if current_messiness and pd.notna(current_messiness):
            # Find the correct field type key (case-insensitive)
            correct_field_type = None
            for key in MESSINESS_OPTIONS.keys():
                if key.lower() == field_type.lower():
                    correct_field_type = key
                    break
            
            if correct_field_type:
                valid_options = MESSINESS_OPTIONS[correct_field_type]
                # Case-insensitive comparison with whitespace trimming
                current_messiness_str = str(current_messiness).strip().lower()
                messiness_match = any(m.lower() == current_messiness_str for m in valid_options)
                if not messiness_match:
                    st.warning(f"⚠️ '{current_messiness}' is not valid for {field_type} type in row {idx + 1}. Using clean data instead.")
                    processed_df.at[idx, 'Messiness'] = None
                    processed_df.at[idx, 'Messiness %'] = 0
    
    # Enforce field limit
    if len(processed_df) > MAX_FIELDS:
        st.error(f"❌ Maximum {MAX_FIELDS} fields allowed. Truncating to first {MAX_FIELDS} fields.")
        processed_df = processed_df.head(MAX_FIELDS)
    
    return processed_df

# Handle Preview Trigger
if preview_btn:
    validated_df = validate_and_process_schema(working_df)
    if len(validated_df) > 0:
        preview_dataset(selected_preset, validated_df)

# Handle Generation Trigger
if generate_btn:
    validated_df = validate_and_process_schema(working_df)
    if len(validated_df) > 0:
        generate_dataset(selected_preset, row_count, validated_df)

# Results Section
if st.session_state.generated_data is not None:
    st.divider()
    col_header, col_timestamp, col_download = st.columns([2, 2, 1])
    with col_header:
        st.subheader("Results")
    with col_timestamp:
        if st.session_state.last_generated:
            time_diff = (datetime.now() - st.session_state.last_generated).total_seconds()
            if time_diff < 60:
                time_str = f"{int(time_diff)}s ago"
            else:
                time_str = st.session_state.last_generated.strftime("%I:%M %p")
            st.caption(f"⏰ Generated: {time_str}")
    with col_download:
        csv = st.session_state.generated_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{selected_preset.lower()}_data.csv",
            mime="text/csv",
            type="primary"
        )
    
    # Data Preview
    # Reset index to start from 1 instead of 0 for better readability
    display_df = st.session_state.generated_data.copy()
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True)

elif not api_key:
    st.info("🔑 **Getting Started:**\n\n1. Enter your Gemini API Key in the sidebar\n2. Choose an industry preset or customize fields\n3. Click Preview to see 10 rows\n4. Click Generate Data to download your dataset\n\n💡 **Tip:** Start with a preset, then try the Quick Actions to add messiness!")