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
import logging
from datetime import datetime

# --- Configuration ---
st.set_page_config(
    page_title="DataGenius",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🔒 Optional Gatekeeper (password protection) ---
# If you set `ACCESS_CODE` in `st.secrets`, the app will require users to
# enter that code before proceeding. This is useful for public demos to
# reduce bot abuse and accidental API key consumption.
def check_password():
    """Returns True if the user has entered the correct access code.

    Behaviour:
    - If `ACCESS_CODE` is present in `st.secrets`, show a small password
      UI and require it. If absent, the function returns True and the
      app proceeds unguarded (useful for local development).
    """

    # If no access code configured, don't gate the app.
    if "ACCESS_CODE" not in st.secrets:
        return True

    def password_entered():
        # Callback executed when user types the password.
        if st.session_state.get("password") == st.secrets["ACCESS_CODE"]:
            st.session_state["password_correct"] = True
            # Remove the raw password from session state for safety.
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # 1) Already authenticated
    if st.session_state.get("password_correct", False):
        return True

    # 2) Show the input box
    st.markdown("## 🔐 Restricted Access")
    st.markdown("To prevent bot abuse and manage API costs, this live demo is password protected.")
    st.info("💡 **Want to try it?** Ask the repo owner for the access code.")

    st.text_input("Enter Access Code", type="password", on_change=password_entered, key="password")

    # 3) Show error if entered and incorrect
    if st.session_state.get("password_correct", False) is False and "password" in st.session_state:
        st.error("😕 Incorrect access code. Please try again.")

    return False


# Stop early if an access code is configured and the user hasn't authenticated.
if "ACCESS_CODE" in st.secrets and not check_password():
    st.stop()
elif "ACCESS_CODE" not in st.secrets:
    # Helpful hint for maintainers: configure an access code for public demos.
    st.sidebar.info("No access code configured. Set `ACCESS_CODE` in `st.secrets` to enable password protection.")

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

# --- API Key Handling (Cloud vs Local) ---
# We support two ways to supply the Gemini API key:
# 1) `st.secrets["GOOGLE_API_KEY"]` for cloud/hosted deployments (secure).
# 2) Manual entry in the sidebar for local development.
# The app will first prefer the hosted secret and fall back to the sidebar input.
api_key = None

# If deployed to Streamlit Cloud or another host, put the key in `st.secrets`.
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]

# If no hosted key found, ask the developer/user to paste it locally.
if not api_key:
    # Local entry is intentionally a password field so the key isn't shown on-screen.
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
else:
    # Visual cue for hosted key usage. Non-essential UI affordance.
    with st.sidebar:
        st.success("✅ Hosted Key Active")

# Configure the Google Generative AI client with the provided key. Many
# functions below call `genai.GenerativeModel` and require this configuration.
if api_key:
    genai.configure(api_key=api_key)

# `show_debug` will be configured by a sidebar checkbox so operators can
# enable detailed errors during development. Default is False.
show_debug = False

# --- Logging & Telemetry (opt-in) ---
# Configure a console logger for structured, non-sensitive events. We
# intentionally never log secrets or raw model outputs.
logger = logging.getLogger("datagen")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Telemetry persistence is disabled by default. Enable by setting
# `ENABLE_TELEMETRY` in `st.secrets` to 'true' or by setting the
# environment variable `ENABLE_TELEMETRY=true`.
telemetry_enabled = False
telemetry_flag = None
if "ENABLE_TELEMETRY" in st.secrets:
    telemetry_flag = str(st.secrets.get("ENABLE_TELEMETRY")).lower()
elif os.getenv("ENABLE_TELEMETRY"):
    telemetry_flag = os.getenv("ENABLE_TELEMETRY").lower()

if telemetry_flag == "true":
    telemetry_enabled = True

# Telemetry file (only used if telemetry_enabled=True). Keep minimal info.
telemetry_file = os.path.join(os.getcwd(), "telemetry.log")

def log_event(event_name: str, details: dict):
    """Log a structured event to console and optionally persist minimal telemetry.

    This function must never include secrets or raw model outputs in the
    logged data. Only include high-level metadata useful for debugging and
    monitoring (counts, durations, non-sensitive flags).
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    entry = {"ts": timestamp, "event": event_name, "details": details}

    # Console log (safe, non-persistent)
    try:
        logger.info(json.dumps(entry))
    except Exception:
        # Fallback: log a simple message
        logger.info(f"{timestamp} - {event_name}")

    # Optionally persist anonymized telemetry (append-only). Don't store
    # raw model text or API keys.
    if telemetry_enabled:
        try:
            with open(telemetry_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            # Avoid raising errors from telemetry writes.
            logger.warning("Failed to write telemetry entry")

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

# --- Limits & rate limiting ---
# Maximum rows allowed for generation (sensible cap to limit cost).
MAX_ROWS = 500
# Rate limiter: max generation calls per window (seconds)
GEN_CALL_LIMIT = 3
GEN_WINDOW_SECONDS = 300  # 5 minutes

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

# Note: analysis/telemetry features were deliberately removed for public
# release to avoid persisting usage data locally.

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


# Timing logging removed for public release to avoid storing local usage data

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
        # Log the schema request (no sensitive content)
        log_event("schema_request", {"prompt_len": len(prompt_text or ""), "max_suggest": 10})
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
                # Parsing failed: surface a helpful debug message when the
                # operator has enabled debug mode. Show a truncated sample of
                # the model output to help diagnose formatting issues.
                if show_debug:
                    sample = (raw_text[:1000] + "...") if len(raw_text) > 1000 else raw_text
                    st.error("Failed to parse model JSON output. Sample below (truncated):")
                    st.code(sample)
                    st.exception(jde)
                else:
                    st.error("Failed to parse model output. Enable 'Show debug details' to inspect the raw model response.")
                return

            # Validate schema structure before accepting it. This prevents
            # malformed or unsafe schema objects from being written into
            # session state.
            valid, msg = validate_schema(schema)
            if not valid:
                # If debug mode is enabled, show the raw sample to help
                # diagnose the issue; otherwise show a concise message.
                log_event("schema_validation_failed", {"reason": msg})
                if show_debug:
                    sample = (raw_text[:1000] + "...") if len(raw_text) > 1000 else raw_text
                    st.error(f"Schema validation failed: {msg}")
                    st.code(sample)
                else:
                    st.error(f"Schema validation failed: {msg}")
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
                log_event("schema_validation_failed_after_trim", {"reason": msg_trim})
                st.error(f"Trimmed schema invalid: {msg_trim}")
                return

            # Telemetry: record how many suggestions the model provided (no
            # content is stored).
            log_event("schema_suggested", {"suggested_count": len(schema)})

            # Persist the proposed schema to session state and force a
            # rerun so the editor UI reflects the new rows immediately.
            st.session_state.fields_df = pd.DataFrame(schema)
            st.rerun()
        except Exception as e:
            # Network errors or model errors may surface here. Show full
            # traceback only when debug is explicitly enabled.
            log_event("schema_error", {"error": str(type(e).__name__)})
            if show_debug:
                st.exception(e)
            else:
                st.error("Failed to generate schema: model request failed. Enable 'Show debug details' for more info.")


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
        # Emit a safe event: a preview was requested. Do not include keys
        # or raw prompts in logs.
        log_event("preview_request", {"preset": industry, "quality": quality, "fields": len(field_desc), "rows": 10})
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
            # Telemetry: report how many rows parsed successfully (no data).
            log_event("preview_result", {"rows_produced": len(df_safe), "fields": len(df_safe.columns)})
            status_text.empty()
            # Force UI refresh so the Results panel appears with the preview.
            st.rerun()
        except Exception as parse_err:
            log_event("preview_parse_error", {"error": str(type(parse_err).__name__)})
            if show_debug:
                st.exception(parse_err)
            st.error("Preview failed: CSV parsing error. The AI output may be malformed — try refining your schema.")
            return

    except Exception as e:
        if show_debug:
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

    # --- Rate limiting (per-session) ---
    now_ts = time.time()
    gen_calls = st.session_state.get("generate_timestamps", [])
    # Keep only timestamps within the window
    gen_calls = [t for t in gen_calls if now_ts - t < GEN_WINDOW_SECONDS]
    if len(gen_calls) >= GEN_CALL_LIMIT:
        # Too many recent generate calls
        st.error(f"Rate limit: Maximum {GEN_CALL_LIMIT} generation requests allowed every {GEN_WINDOW_SECONDS//60} minutes. Try again later.")
        log_event("rate_limited", {"type": "generate", "recent_calls": len(gen_calls)})
        return
    # Record this attempt now (will be saved after checks)
    gen_calls.append(now_ts)
    st.session_state["generate_timestamps"] = gen_calls

    # Clamp rows to a hard maximum to prevent excessive token use
    if rows > MAX_ROWS:
        st.warning(f"Requested rows ({rows}) exceed maximum allowed ({MAX_ROWS}). Clamping to {MAX_ROWS} rows.")
        rows = MAX_ROWS

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
        # Log a safe generation request event (no secrets or raw output).
        log_event("generate_request", {"preset": industry, "rows": rows, "fields": len(field_desc), "quality": quality})
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
            log_event("generate_result", {"rows_produced": len(df_safe), "fields": len(df_safe.columns)})
        except Exception as parse_err:
            # Provide a clear message and optionally reveal the exception when
            # debugging is enabled.
            if show_debug:
                st.exception(parse_err)
            log_event("generate_parse_error", {"error": str(type(parse_err).__name__)})
            st.error("Generation failed: CSV parsing error. The AI may have generated inconsistent data or used unexpected formatting.")
            return

        # Complete UI update and force a rerun so the Results are rendered.
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        st.rerun()

    except Exception as e:
        if show_debug:
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
    
    # Warn operators if a hosted API key is active. Hosting a key on a
    # public deployment can allow anyone with access to consume the key
    # and incur costs — prefer per-user keys for public services.
    if "GOOGLE_API_KEY" in st.secrets:
        st.warning(
            "Using a hosted API key: ensure this deployment is private or restrict usage."
        )

    # Developer aid: allow showing detailed errors when debugging locally.
    # Keep disabled by default for safety in public-facing deployments.
    show_debug = st.checkbox("Show debug details", value=False, help="Reveal error traces for debugging only")

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