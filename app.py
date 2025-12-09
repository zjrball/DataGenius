"""
Project: DataGenius
Author: Zachary Ball
Co-Authored By: Google Gemini & Claude AI (AI Assistants)

Description:
This application was developed with AI assistance to demonstrate modern 
rapid-prototyping workflows. While the code logic was augmented by AI, 
all outputs have been manually reviewed and tested for accuracy and security.
"""

import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import io
import time
import os

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
    'Type' must be one of FIELD_TYPES.
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

PRESETS = {
    "E-commerce": [
        {"Field Name": "Transaction ID", "Type": "UUID", "Context": "Unique identifier"},
        {"Field Name": "Customer", "Type": "Name", "Context": ""},
        {"Field Name": "Product", "Type": "Category", "Context": "Electronics, Home, Fashion"},
        {"Field Name": "Amount", "Type": "Currency", "Context": "Min: 10, Max: 500"},
        {"Field Name": "Date", "Type": "Date", "Context": "2024"},
    ],
    "Healthcare": [
        {"Field Name": "Patient ID", "Type": "UUID", "Context": ""},
        {"Field Name": "Diagnosis", "Type": "String", "Context": "ICD-10 codes"},
        {"Field Name": "Admission", "Type": "Date", "Context": "Last 6 months"},
        {"Field Name": "Bill", "Type": "Currency", "Context": "Min: 500, Max: 50000"},
    ],
    "Finance": [
        {"Field Name": "Account", "Type": "UUID", "Context": ""},
        {"Field Name": "Type", "Type": "Category", "Context": "Debit, Credit, Transfer"},
        {"Field Name": "Amount", "Type": "Currency", "Context": "Min: 50, Max: 5000"},
        {"Field Name": "Fraud Flag", "Type": "Boolean", "Context": "1% True"},
    ],
    "Custom": [
        {"Field Name": "ID", "Type": "UUID", "Context": ""},
    ]
}

# --- State Management ---
# Use `st.session_state` so schema and generated data persist across
# reruns. Initialize with a sensible default preset when first run.
if "fields_df" not in st.session_state:
    st.session_state.fields_df = pd.DataFrame(PRESETS["E-commerce"])

if "generated_data" not in st.session_state:
    st.session_state.generated_data = None

# --- Logic ---

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

def generate_schema_from_prompt(prompt_text):
    """Ask the model to propose a small schema from a short description.

    Notes:
    - The model may return code fences or additional commentary; we strip
      common markdown wrappers before attempting to parse JSON.
    - We intentionally cap AI suggestions to 10 fields here (even though
      `MAX_FIELDS` is larger). This is to keep AI suggestions concise and
      useful; the user can always edit/expand the schema manually up to
      `MAX_FIELDS`.
    """
    if not api_key:
        st.error("API Key required")
        return

    with st.spinner("Dreaming up schema..."):
        model = get_model()
        # System prompt: be explicit and restrictive. In many cases the model
        # produces helpful output but can also include prose or wrappers; we
        # instruct it to return only a JSON array so parsing is more reliable.
        sys_prompt = f"""
        You are a data architect. Generate a list of fields for a dataset described as: "{prompt_text}".
        IMPORTANT: Generate maximum 10 fields only (not {MAX_FIELDS}).
        Return ONLY valid JSON array of objects with keys: "Field Name", "Type", "Context".
        Allowed Types: {", ".join(FIELD_TYPES)}.
        Context should include min/max for numbers or category options.
        """
        try:
            # Request content from the model. Capture the raw text so we can
            # show a truncated preview when parsing fails (only when debug
            # mode is enabled).
            response = model.generate_content(sys_prompt)
            raw_text = response.text
            # Model may include markdown code fences like ```json```; strip them.
            clean_json = raw_text.replace("```json", "").replace("```", "").strip()
            try:
                schema = json.loads(clean_json)
            except json.JSONDecodeError as jde:
                # Parsing failed: show a truncated sample and the JSON error
                # so local developers can quickly iterate on prompts.
                sample = (raw_text[:1000] + "...") if len(raw_text) > 1000 else raw_text
                st.error("Failed to parse model JSON output. Sample below (truncated):")
                st.code(sample)
                st.exception(jde)
                return

            # Validate schema structure before accepting it. This prevents
            # malformed or unsafe schema objects from being written into
            # session state.
            valid, msg = validate_schema(schema)
            if not valid:
                # Show the failing reason and a short sample to help local
                # debugging and prompt tuning.
                sample = (raw_text[:1000] + "...") if len(raw_text) > 1000 else raw_text
                st.error(f"Schema validation failed: {msg}")
                st.code(sample)
                return

            # Safety: trim overly large AI suggestions to 10 items. We show a
            # warning so users understand a truncation happened.
            if len(schema) > 10:
                st.warning(f"⚠️ AI suggested {len(schema)} fields. Trimming to 10 fields for AI autofill.")
                schema = schema[:10]

            # Validate trimmed schema as well (in case trimming removed valid
            # fields but left invalid items).
            valid_trim, msg_trim = validate_schema(schema)
            if not valid_trim:
                st.error(f"Trimmed schema invalid: {msg_trim}")
                return

            # Record suggested count in-memory only.

            # Persist the proposed schema to session state and force a
            # rerun so the editor UI reflects the new rows immediately.
            st.session_state.fields_df = pd.DataFrame(schema)
            st.rerun()
        except Exception as e:
            # Network errors or model errors may surface here. Show traceback
            # for local debugging.
            st.exception(e)
            st.error("Failed to generate schema: model request failed.")


def preview_dataset(industry, quality, fields_data, dirty_percentage=10):
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
    
    # Build a compact textual description of each field for the prompt.
    # Example: "Age (Number): Min: 18, Max: 99"
    # Keeping this human-friendly reduces ambiguity in the generated CSV.
    # If `fields_data` is a DataFrame with unexpected columns, the list
    # comprehension will raise; callers should ensure a valid schema format.
    #
    # Note: `fields_data.iterrows()` yields a snapshot of the DataFrame rows.
    # If the user edits the DataFrame in the UI after this snapshot, those
    # edits won't affect the current generation call.
    #
    # Construct schema description
    field_desc = []
    for _, row in fields_data.iterrows():
        field_desc.append(f"{row['Field Name']} ({row['Type']}): {row['Context']}")
    
    quality_prompt = (
        "Strictly clean data. Standard formats. No nulls." 
        if quality == "Clean" 
        else f"Messy data. {dirty_percentage}% nulls/typos. Mixed date formats."
    )

    # The prompt explicitly demands CSV-only output (no markdown/code fences)
    # and the exact number of columns. Explicit instructions reduce the rate of
    # malformed output but do not eliminate it, hence the forgiving parsing
    # above.
    prompt = f"""
    Generate a CSV dataset for '{industry}'.
    Rows: 10
    Quality Rules: {quality_prompt}

    Columns (exactly {len(field_desc)} fields):
    {"; ".join(field_desc)}

    CRITICAL: Output ONLY valid CSV with exactly {len(field_desc)} columns per row. Include headers. No markdown formatting.
    """

    status_text = st.empty()

    try:
        # Request preview from the model and aggressively strip common
        # markdown wrappers. We still protect parsing with pandas.
        status_text.text("Generating preview...")
        response = model.generate_content(prompt)
        csv_text = response.text.replace("```csv", "").replace("```", "").strip()

        # Parse and display with error handling
        try:
            # `on_bad_lines='skip'` prevents a single malformed AI row from
            # crashing the whole preview. We surface an error if the final
            # parsed frame is empty to guide the user to improve prompts.
            df = pd.read_csv(io.StringIO(csv_text), on_bad_lines='skip', engine='python')
            if len(df) == 0:
                st.error("Preview failed: No valid rows generated. Please try again.")
                return

            # Sanitize the resulting DataFrame to mitigate CSV injection
            # before storing and presenting it to the user or offering a
            # download.
            df_safe = _sanitize_dataframe_for_csv(df)
            st.session_state.generated_data = df_safe
            status_text.empty()
            # Force UI refresh so the Results panel appears with the preview.
            st.rerun()
        except Exception as parse_err:
            st.exception(parse_err)
            st.error("Preview failed: CSV parsing error. The AI output may be malformed — try refining your schema.")
            return

    except Exception as e:
        st.exception(e)
        st.error("Preview failed: model request failed. Check API key and network.")

def generate_dataset(industry, rows, quality, fields_data, dirty_percentage=10):
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
    
    # Construct schema description
    # Build the descriptive field list to place in the AI prompt. Keep it
    # brief but informative (min/max, categories, uniqueness hints).
    field_desc = []
    for _, row in fields_data.iterrows():
        field_desc.append(f"{row['Field Name']} ({row['Type']}): {row['Context']}")
    
    quality_prompt = (
        "Clean data. Standard formats. No nulls." 
        if quality == "Clean" 
        else f"Messy data. {dirty_percentage}% nulls/typos. Mixed formats."
    )

    prompt = f"""Generate {rows} CSV rows for '{industry}'. {quality_prompt}
Columns: {"; ".join(field_desc)}
Output: CSV with headers only, no markdown."""

    # UI affordances: progress and status improve perceived performance.
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 1. Generate Data: request CSV text from the model.
        status_text.text("Generating synthetic data...")
        progress_bar.progress(30)
        # Local generation request
        start_ts = time.time()
        response = model.generate_content(prompt)
        csv_text = response.text.replace("```csv", "").replace("```", "").strip()

        progress_bar.progress(50)

        # 2. Parse and Store: robust parsing as in preview (skip bad lines).
        try:
            df = pd.read_csv(io.StringIO(csv_text), on_bad_lines='skip', engine='python')
            if len(df) == 0:
                st.error("Generation failed: No valid rows produced. Try adjusting your field descriptions.")
                return

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
        st.rerun()

    except Exception as e:
        st.exception(e)
        st.error("Generation failed: model request failed. Check API key and network.")

# --- UI Layout ---

# Sidebar
with st.sidebar:
    st.title("DataGenius ⚡")
    
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
    data_quality = st.radio("Data Quality", ["Clean", "Dirty"], index=0)
    
    # Dirty data percentage slider
    if data_quality == "Dirty":
        dirty_percentage = st.slider("Dirty Data %", 0, 80, 10, help="Percentage of rows with issues")
    else:
        dirty_percentage = 0
    
    # Display ETA
    eta = calculate_eta(row_count, len(st.session_state.fields_df))
    st.caption(f"⏱️ Estimated time: {eta}")
    
    st.divider()
    
    # Developer aid: exceptions are shown in the UI for local debugging.
    preview_btn = st.button("👁️ Preview (10 rows)", use_container_width=True)
    generate_btn = st.button("🚀 Generate Data", use_container_width=True, type="primary")

# Main Content
st.header("Schema Configuration")

# AI Auto-Fill Section
with st.expander("✨ AI Auto-Fill Schema"):
    col_ai_1, col_ai_2 = st.columns([4, 1])
    with col_ai_1:
        ai_prompt = st.text_input("Describe your dataset", placeholder="e.g. 'Startup employee roster with equity and vesting schedules'")
    with col_ai_2:
        st.write("") # Spacer
        st.write("") 
        if st.button("Auto-Fill"):
            generate_schema_from_prompt(ai_prompt)

# Field Editor with limit indicator
field_count = len(st.session_state.fields_df)
field_limit_color = "🟢" if field_count <= MAX_FIELDS else "🔴"
st.caption(f"{field_limit_color} Fields: {field_count}/{MAX_FIELDS}")

edited_df = st.data_editor(
    st.session_state.fields_df,
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
            help="e.g. 'Min: 10, Max: 100' or 'Must be unique'"
        )
    }
)

# Update session state when rows are edited or deleted
st.session_state.fields_df = edited_df

if field_count >= MAX_FIELDS:
    st.info(f"ℹ️ Field limit ({MAX_FIELDS}) reached. Remove fields to add more.")

# Handle Preview Trigger
if preview_btn:
    if len(edited_df) > MAX_FIELDS:
        st.error(f"❌ Cannot generate: {len(edited_df)} fields exceed the {MAX_FIELDS} field limit.")
    else:
        preview_dataset(selected_preset, data_quality, edited_df, dirty_percentage)

# Handle Generation Trigger
if generate_btn:
    if len(edited_df) > MAX_FIELDS:
        st.error(f"❌ Cannot generate: {len(edited_df)} fields exceed the {MAX_FIELDS} field limit.")
    else:
        generate_dataset(selected_preset, row_count, data_quality, edited_df, dirty_percentage)

# Results Section
if st.session_state.generated_data is not None:
    st.divider()
    col_header, col_download = st.columns([3, 1])
    with col_header:
        st.subheader("Results")
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
    st.dataframe(st.session_state.generated_data, use_container_width=True)

elif not api_key:
    st.warning("Please enter a Gemini API Key in the sidebar to start generating.")