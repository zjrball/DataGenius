import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import io
import time

# --- Configuration ---
st.set_page_config(
    page_title="DataGenius",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# --- API Key Handling (The Update) ---
api_key = None

# Check if the key exists in Streamlit Secrets (Cloud Deployment)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    
# Fallback: Ask user for key if not in secrets (Local Use)
if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
else:
    # Optional: Show a badge if using the hosted key
    with st.sidebar:
        st.success("✅ Hosted Key Active")

# Configure Gemini if key is present
if api_key:
    genai.configure(api_key=api_key)

# --- Constants ---
MAX_FIELDS = 15
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
    "Empty": [
        {"Field Name": "ID", "Type": "UUID", "Context": ""},
    ]
}

# --- State Management ---
if "fields_df" not in st.session_state:
    st.session_state.fields_df = pd.DataFrame(PRESETS["E-commerce"])

if "generated_data" not in st.session_state:
    st.session_state.generated_data = None

if "analysis_ideas" not in st.session_state:
    st.session_state.analysis_ideas = []

# --- Logic ---

def generate_schema_from_prompt(prompt_text):
    """Uses Gemini to hallucinate a schema based on user description."""
    if not api_key:
        st.error("API Key required")
        return

    with st.spinner("Dreaming up schema..."):
        model = genai.GenerativeModel('gemini-2.5-flash')
        sys_prompt = f"""
        You are a data architect. Generate a list of fields for a dataset described as: "{prompt_text}".
        IMPORTANT: Generate maximum 10 fields only (not {MAX_FIELDS}).
        Return ONLY valid JSON array of objects with keys: "Field Name", "Type", "Context".
        Allowed Types: {", ".join(FIELD_TYPES)}.
        Context should include min/max for numbers or category options.
        """
        try:
            response = model.generate_content(sys_prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            schema = json.loads(clean_json)
            
            # Enforce 10-field limit for AI suggestions
            if len(schema) > 10:
                st.warning(f"⚠️ AI suggested {len(schema)} fields. Trimming to 10 fields for AI autofill.")
                schema = schema[:10]
            
            st.session_state.fields_df = pd.DataFrame(schema)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate schema: {e}")

def preview_dataset(industry, quality, fields_data):
    """Generates a quick 10-row preview of the dataset."""
    if not api_key:
        st.error("API Key required")
        return
    
    # Enforce field limit
    if len(fields_data) > MAX_FIELDS:
        st.error(f"❌ Maximum {MAX_FIELDS} fields allowed. You have {len(fields_data)} fields.")
        return

    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Construct schema description
    field_desc = []
    for _, row in fields_data.iterrows():
        field_desc.append(f"{row['Field Name']} ({row['Type']}): {row['Context']}")
    
    quality_prompt = (
        "Strictly clean data. Standard formats. No nulls." 
        if quality == "Clean" 
        else "Messy data. 10% nulls. Occasional typos. Mixed date formats."
    )

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
        status_text.text("Generating preview...")
        response = model.generate_content(prompt)
        csv_text = response.text.replace("```csv", "").replace("```", "").strip()
        
        # Parse and display with error handling
        try:
            df = pd.read_csv(io.StringIO(csv_text), on_bad_lines='skip')
            if len(df) == 0:
                st.error("Preview failed: No valid rows generated. Please try again.")
                return
            st.session_state.generated_data = df
            st.session_state.analysis_ideas = []
            status_text.empty()
            st.rerun()
        except pd.errors.ParserError as parse_err:
            st.error(f"Preview failed: CSV parsing error. {str(parse_err)}")
            return

    except Exception as e:
        st.error(f"Preview failed: {e}")

def generate_dataset(industry, rows, quality, fields_data):
    """Generates the actual CSV data."""
    if not api_key:
        st.error("API Key required")
        return
    
    # Enforce field limit
    if len(fields_data) > MAX_FIELDS:
        st.error(f"❌ Maximum {MAX_FIELDS} fields allowed to manage token usage. You have {len(fields_data)} fields.")
        return

    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Construct schema description
    field_desc = []
    for _, row in fields_data.iterrows():
        field_desc.append(f"{row['Field Name']} ({row['Type']}): {row['Context']}")
    
    quality_prompt = (
        "Strictly clean data. Standard formats. No nulls." 
        if quality == "Clean" 
        else "Messy data. 10% nulls. Occasional typos. Mixed date formats."
    )

    prompt = f"""
    Generate a CSV dataset for '{industry}'.
    Rows: {rows}
    Quality Rules: {quality_prompt}
    
    Columns:
    {"; ".join(field_desc)}
    
    Output ONLY raw CSV content. Include headers.
    """

    analysis_prompt = f"""
    Suggest 3 business questions for a {industry} dataset with columns: {", ".join(fields_data['Field Name'].tolist())}.
    Return JSON string array.
    """

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 1. Generate Data
        status_text.text("Generating synthetic data...")
        progress_bar.progress(30)
        
        response = model.generate_content(prompt)
        csv_text = response.text.replace("```csv", "").replace("```", "").strip()
        
        progress_bar.progress(70)
        
        # 2. Generate Analysis
        status_text.text("Analyzing potential insights...")
        analysis_res = model.generate_content(analysis_prompt)
        
        try:
            ideas = json.loads(analysis_res.text.replace("```json", "").replace("```", ""))
        except:
            ideas = ["Analyze trends", "Check distributions", "Find outliers"]
            
        progress_bar.progress(90)
        
        # 3. Parse and Store
        df = pd.read_csv(io.StringIO(csv_text))
        st.session_state.generated_data = df
        st.session_state.analysis_ideas = ideas
        
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        st.rerun()

    except Exception as e:
        st.error(f"Generation failed: {e}")

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
    
    st.divider()
    
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
        preview_dataset(selected_preset, data_quality, edited_df)

# Handle Generation Trigger
if generate_btn:
    if len(edited_df) > MAX_FIELDS:
        st.error(f"❌ Cannot generate: {len(edited_df)} fields exceed the {MAX_FIELDS} field limit.")
    else:
        generate_dataset(selected_preset, row_count, data_quality, edited_df)

# Results Section
if st.session_state.generated_data is not None:
    st.divider()
    st.subheader("Results")
    
    # Analysis Ideas
    if st.session_state.analysis_ideas:
        cols = st.columns(len(st.session_state.analysis_ideas))
        for i, idea in enumerate(st.session_state.analysis_ideas):
            with cols[i]:
                st.info(idea)

    # Data Preview
    st.dataframe(st.session_state.generated_data, use_container_width=True)
    
    # Download
    csv = st.session_state.generated_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{selected_preset.lower()}_data.csv",
        mime="text/csv",
        type="primary"
    )

elif not api_key:
    st.warning("Please enter a Gemini API Key in the sidebar to start generating.")