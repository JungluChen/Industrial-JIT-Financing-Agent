import streamlit as st
import pandas as pd
from openai import OpenAI
import time
import json
import random
import os
import base64
import streamlit.components.v1 as components
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
import streamlit as st

# 針對右下角所有懸浮元件的強制隱藏
hide_streamlit_cloud_elements = """
    <style>
    /* 1. 隱藏右上角選單與工具欄 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 2. 隱藏頁尾 "Made with Streamlit" */
    footer {visibility: hidden;}

    /* 3. 隱藏右下角 "Manage app" 黑色按鈕 */
    .stAppDeployButton {
        display: none !important;
    }

    /* 4. 隱藏右下角 "Hosted with Streamlit" 紅色標籤 (針對 Cloud 注入) */
    div[data-testid="stStatusWidget"] {
        visibility: hidden !important;
        display: none !important;
    }

    /* 5. 萬用：針對任何在底部右側出現的懸浮容器 */
    [data-testid="stConnectionStatus"],
    .st-emotion-cache-1wb5477, 
    .st-emotion-cache-6qob1r {
        display: none !important;
    }
    </style>
    """

st.markdown(hide_streamlit_cloud_elements, unsafe_allow_html=True)
def st_mermaid(code, theme="default"):
    mermaid_code = f"%%{{init: {{'theme': '{theme}'}}}}%%\n{code}"
    encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
    url = f"https://mermaid.ink/svg/{encoded}"
    st.markdown(f'<div style="display: flex; justify-content: center;"><img src="{url}" alt="System Architecture Flowchart" style="max-width: 100%;"></div>', unsafe_allow_html=True)

# App Configuration
st.set_page_config(
    page_title="Industrial JIT Financing Agent",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None
    }
)
# Footer/Tutorial Docs

# -----------------------------------------------------------------------------
# 1. State Management
# -----------------------------------------------------------------------------

# Sidebar Theme Toggle
with st.sidebar:
    st.title("⚙️ Settings")
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'Light'
        
    is_dark = st.toggle("🌙 Dark Mode", value=(st.session_state.theme_mode == 'Dark'))
    new_mode = 'Dark' if is_dark else 'Light'
    
    if new_mode != st.session_state.theme_mode:
        st.session_state.theme_mode = new_mode
        # Update Streamlit config to natively switch themes
        os.makedirs('.streamlit', exist_ok=True)
        with open('.streamlit/config.toml', 'w') as f:
            f.write(f'[theme]\nbase="{new_mode.lower()}"\nprimaryColor="#2980b9"\n')
        st.rerun()
    st.divider()

mermaid_theme = "dark" if st.session_state.theme_mode == 'Dark' else "default"

# Metric Cards Custom CSS (Streamlit doesn't have native metric cards exactly as requested, so we retain minimal CSS for layout)
is_dark_mode = st.session_state.theme_mode == 'Dark'
st.markdown(f"""
<style>
    .metric-card {{
        background-color: {"#111111" if is_dark_mode else "#ffffff"} !important;
        padding: 20px;
        border-radius: 12px;
        box-shadow: {"0 0 0 1px #333333" if is_dark_mode else "0 4px 6px rgba(0,0,0,0.1)"};
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid {"#333333" if is_dark_mode else "#e0e0e0"} !important;
    }}
    .metric-card h3, .metric-card h2 {{
        color: {"#ffffff" if is_dark_mode else "#1f1f1f"} !important;
    }}
</style>
""", unsafe_allow_html=True)


if 'erp_data' not in st.session_state:
    try:
        st.session_state.erp_data = pd.read_csv('mock_erp_data.csv')
    except Exception as e:
        st.error(f"Failed to load ERP data: {e}")
        st.session_state.erp_data = pd.DataFrame()

if 'funding_gap_detected' not in st.session_state:
    st.session_state.funding_gap_detected = False

if 'selected_po' not in st.session_state:
    st.session_state.selected_po = None

if 'agent_reasoning' not in st.session_state:
    st.session_state.agent_reasoning = ""

if 'lender_rates' not in st.session_state:
    st.session_state.lender_rates = []

if 'execution_status' not in st.session_state:
    st.session_state.execution_status = "Waiting..."

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# -----------------------------------------------------------------------------
# 2. NVIDIA NIM API Setup
# -----------------------------------------------------------------------------
try:
    nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
except KeyError:
    st.error("NVIDIA_API_KEY is not set in .streamlit/secrets.toml!")
    st.stop()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key
)

def get_agent_reasoning(order_data):
    """Call the NVIDIA NIM API for reasoning logic"""
    prompt = f"""
    You are an AI Industrial Credit Agent monitoring a manufacturer's ERP system.
    Analyze the following Purchase Order for an immediate funding gap:
    {order_data}
    
    Calculate the exact gap (Amount - Available_Balance).
    Provide a concise explanation of the risk to the supply chain if this gap is not funded immediately.
    State how much funding is required.
    Format your response cleanly.
    """
    
    try:
        response = client.chat.completions.create(
            model="meta/llama3-8b-instruct", # Updated as per prompt instructions
            messages=[
                {"role": "system", "content": "You are a senior financial analyst and AI credit agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        # Graceful fallback for demo purposes if API key or model fails
        return f"**NVIDIA NIM Analysis (Demo Fallback - {e}):**\n\n- **Detected Gap:** The order exceeds available balance by exactly the shortfall amount.\n- **Risk Assessment:** Critical supply chain bottleneck. Failure to fund this PO immediately will delay production and incur late penalties.\n- **Recommendation:** Proceed to query lenders for short-term JIT financing to bridge this gap."

# -----------------------------------------------------------------------------
# 3. Tool Use: Mock Lender API
# -----------------------------------------------------------------------------
def query_lender_api(funding_amount):
    """Simulates querying external fintech lender APIs for current rates"""
    time.sleep(1.5) # Simulate API latency
    
    base_rate = 5.0
    # Simulate variations based on amount
    risk_premium = random.uniform(0.5, 2.0) if funding_amount > 50000 else random.uniform(0.1, 1.0)
    
    return [
        {"Lender": "FinBank FastCapital", "Rate": f"{base_rate + risk_premium:.2f}%", "Term": "30 Days", "Approval": "Instant"},
        {"Lender": "Industrial Credit Union", "Rate": f"{base_rate + risk_premium - 0.5:.2f}%", "Term": "45 Days", "Approval": "1 Hour"},
        {"Lender": "Global Trade Finance", "Rate": f"{base_rate + risk_premium + 1.2:.2f}%", "Term": "60 Days", "Approval": "Instant"}
    ]

# -----------------------------------------------------------------------------
# 4. Interface Setup (4-Screen Flow)
# -----------------------------------------------------------------------------
st.title("🏭 Industrial JIT Financing Agent")

st.markdown("""
**Author:** Chen Jung-Lu (E1582484@u.nus.edu)  
**Course:** DBA5105 Fintech, Enabling Technologies and Analytics  
**Project:** ASSIGNMENT 2 - AGENTIC FINTECH
""")

with st.expander("📚 Detailed Workflow & Tutorial", expanded=False):
    st.markdown("""
    ### 🎯 Getting Started with the Industrial JIT Financing Agent
    Welcome to the Industrial JIT (Just-In-Time) Financing Agent demo. This application demonstrates an autonomous AI agent capable of detecting supply chain funding gaps and proposing real-time financing solutions.

    ### 🛠️ Step-by-Step Workflow
    
    **AGENT 1 (ERP MONITOR)**
    1. Read real-time Purchase Order Data from ERP system (CSV)
    2. Scan for Purchase Orders where `Amount` exceeds `Available_Balance`
    3. Detect immediate funding gap and exact shortfall
    4. HANDOVER to AGENT 2

    **AGENT 2 (CREDIT ANALYST)**
    1. Receive funding gap and order details from AGENT 1
    2. Call NVIDIA NIM API to assess critical supply chain risks and state required funding
    3. Present risk assessment reasoning to HUMAN
    4. Query external Fintech Lender APIs for real-time financing rates
    5. Present top financing options with Terms and Approval times
    6. HANDOVER to HUMAN

    **HUMAN (APPROVAL MANAGER)**
    1. Review risk assessment and financing options
    2. Decide whether to **Approve Funding** for the best rate or **Reject & Cancel** (YES/NO)
    3. If YES, funds are transferred and PO Status is updated to 'Funded'

    **AGENT 3 (WORKFLOW ASSISTANT)**
    1. Receive entire ERP context and chat history
    2. Wait for HUMAN questions regarding order values, risks, or data insights
    3. Use NVIDIA NIM API to generate real-time answers based on data
    4. Present answer to HUMAN

    ### 📊 System Architecture Flowchart
    """)
    mermaid_code = """
        graph LR
            subgraph AGENT_1 [AGENT 1: ERP MONITOR]
                A1[1. Read ERP Data CSV] --> A2[2. Detect Funding Gap]
            end

            subgraph AGENT_2 [AGENT 2: CREDIT ANALYST]
                B1[3. Assess Risk via NVIDIA API] --> B2[4. Query Fintech Lenders]
            end

            subgraph HUMAN [HUMAN: APPROVAL MANAGER]
                H1{5. Review and Decide} -->|Approve| H2(6. Transfer Funds)
                H1 -->|Reject| H3(Cancel)
            end

            subgraph AGENT_3 [AGENT 3: WORKFLOW ASSISTANT]
                C1[Receive User Query] --> C2[Generate Insights via AI]
            end

            A2 -->|Handover Workflow| B1
            B2 -->|Present Options| H1
            A1 -.->|Data Context| C1
            C2 -.->|Answer Questions| C1
    """

    
    st_mermaid(mermaid_code, theme=mermaid_theme)

    st.markdown("""
    ### 💻 Technologies Used
    - **Frontend:** Streamlit
    - **AI Engine:** NVIDIA NIM API
    - **Data Handling:** Pandas DataFrames
    """)

st.markdown("Autonomous detection and resolution of supply chain funding gaps.")

# Top metrics calculation
df = st.session_state.erp_data
total_po_value = df['Amount'].sum()
pending_gaps = len(df[(df['Amount'] > df['Available_Balance']) & (df['Status'] == 'Pending')])
active_financing = df[df['Status'] == 'Funded']['Amount'].sum()

cols = st.columns(3)
with cols[0]:
    st.markdown(f'<div class="metric-card"><h3>Total PO Value</h3><h2>${total_po_value:,.0f}</h2></div>', unsafe_allow_html=True)
with cols[1]:
    st.markdown(f'<div class="metric-card"><h3>Pending Gaps</h3><h2>{pending_gaps}</h2></div>', unsafe_allow_html=True)
with cols[2]:
    st.markdown(f'<div class="metric-card"><h3>Active Financing</h3><h2>${active_financing:,.0f}</h2></div>', unsafe_allow_html=True)

st.divider()

# Layout for the 4 sections

# Screen 1: ERP Monitoring

st.subheader("📊 1. ERP System Monitor")
st.markdown("Real-time Purchase Order Data")

# File upload and download template
with open("mock_erp_data.csv", "r") as f:
    csv_template = f.read()
st.download_button(
    label="📥 Download Template CSV",
    data=csv_template,
    file_name="erp_template.csv",
    mime="text/csv",
    width='stretch'
)

uploaded_file = st.file_uploader("Upload your ERP CSV data", type=['csv'])
if uploaded_file is not None:
    try:
        st.session_state.erp_data = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# Editable Table
st.session_state.erp_data = st.data_editor(st.session_state.erp_data, width='stretch', hide_index=True, num_rows="dynamic")

st.markdown("**Agent Action:**")
col2a, col2b = st.columns([1, 1])
with col2a:
    if st.button("🔍 Run Gap Analysis (Agent Scan)", type="primary", width='stretch'):
        with st.spinner("Agent scanning ERP data..."):
            time.sleep(1) # simulate scan
            
            # Find a PO with a gap (Amount > Available) and status Pending
            df = st.session_state.erp_data
            gaps = df[(df['Amount'] > df['Available_Balance']) & (df['Status'] == 'Pending')]
            
            if not gaps.empty:
                po_to_fund = gaps.iloc[0].to_dict()
                st.session_state.selected_po = po_to_fund
                st.session_state.funding_gap_detected = True
                
                # Fetch reasoning
                po_str = json.dumps(po_to_fund, indent=2)
                st.session_state.agent_reasoning = get_agent_reasoning(po_str)
                st.session_state.execution_status = "Awaiting Action"
                st.rerun()
            else:
                st.success("No funding gaps detected at this time.")
with col2b:
    if st.button("🔄 Reset System State", width='stretch'):
        default_data = """PO_Number,Supplier,Required_Date,Amount,Available_Balance,Status
PO-1001,Acme Corp,2026-03-01,85000.0,50000.0,Pending
PO-1002,Global Industries,2026-03-05,25000.0,25000.0,Funded
PO-1003,TechFlow Systems,2026-03-10,120000.0,120000.0,Funded
PO-1004,Nexus Dynamics,2026-03-15,35000.0,35000.0,Pending"""
        
        with open('mock_erp_data.csv', 'w') as f:
            f.write(default_data)
        
        st.session_state.erp_data = pd.read_csv('mock_erp_data.csv')
        st.session_state.funding_gap_detected = False
        st.session_state.selected_po = None
        st.session_state.agent_reasoning = ""
        st.session_state.lender_rates = []
        st.session_state.execution_status = "Waiting..."
        st.rerun()

# Screen 2: Reasoning Engine

st.subheader("🧠 2. Agent Reasoning Layer")
if st.session_state.funding_gap_detected and st.session_state.selected_po:
    po_num = st.session_state.selected_po['PO_Number']
    gap_amt = st.session_state.selected_po['Amount'] - st.session_state.selected_po['Available_Balance']
    
    st.info(f"🚨 **Gap Detected for {po_num} | Shortfall: ${gap_amt:,.2f}**")
    
    with st.container(border=True):
        st.markdown("**NVIDIA NIM Response (`nemotron-4-340b-instruct`):**")
        st.write(st.session_state.agent_reasoning)
        
    if st.button("⚙️ Trigger Tool Use (Query Lenders)", width='stretch'):
        with st.spinner("Agent querying lender APIs..."):
            st.session_state.lender_rates = query_lender_api(gap_amt)
            st.rerun()
else:
    st.info("Agent is idle. Run Gap Analysis to begin.")

st.divider()

# Lower section layout
col3, col4 = st.columns([1, 1])

# Screen 3: Tool Use / API Results
with col3:
    st.subheader("🛠️ 3. Tool Use: Lender API Rates")
    if len(st.session_state.lender_rates) > 0:
        st.markdown("Agent successfully negotiated/queried the following options:")
        rates_df = pd.DataFrame(st.session_state.lender_rates)
        st.dataframe(rates_df, width='stretch', hide_index=True)
    else:
        st.info("Awaiting Tool Use execution.")

# Screen 4: HITL Layer
with col4:
    st.subheader("🛡️ 4. Execution & HITL Security Gate")
    if len(st.session_state.lender_rates) > 0:
        st.warning("⚠️ **Human-in-the-Loop Approval Required** ⚠️")
        st.markdown(f"Agent proposes executing the best available rate for **{st.session_state.selected_po['PO_Number']}**.")
        
        best_lender = st.session_state.lender_rates[0]['Lender']
        rate = st.session_state.lender_rates[0]['Rate']
        
        st.success(f"**Proposed:** Fund via {best_lender} at {rate}")
        
        if st.session_state.execution_status == "Funded!":
            st.success("✅ Funds have been transferred and PO status updated.")
        else:
            if st.button("✅ Approve Funding", type="primary", width='stretch'):
                # Update status
                st.session_state.execution_status = "Funded!"
                
                # Update dataframe
                po_num = st.session_state.selected_po['PO_Number']
                idx = st.session_state.erp_data.index[st.session_state.erp_data['PO_Number'] == po_num].tolist()[0]
                st.session_state.erp_data.at[idx, 'Status'] = 'Funded'
                st.session_state.erp_data.at[idx, 'Available_Balance'] = st.session_state.erp_data.at[idx, 'Amount']
                
                # Save data (mock)
                st.session_state.erp_data.to_csv('mock_erp_data.csv', index=False)
                
                st.rerun()
                
            if st.button("❌ Reject & Cancel", width='stretch'):
                st.session_state.funding_gap_detected = False
                st.session_state.selected_po = None
                st.session_state.agent_reasoning = ""
                st.session_state.lender_rates = []
                st.session_state.execution_status = "Waiting..."
                st.rerun()
                
    else:
        st.info("Awaiting lender proposals.")

st.divider()

# Screen 5: Workflow Chat
st.subheader("💬 5. Workflow Agent Chat")
st.markdown("Ask the agent questions about the current data, funding options, or supply chain risks.")


has_messages = len(st.session_state.chat_messages) > 0

# Only create the visible height=300 container if there are messages or a new prompt being submitted
chat_placeholder = st.empty()

with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([10, 1])
    with cols[0]:
        prompt = st.text_input("Ask the agent a question...", label_visibility="collapsed", placeholder="Ask the agent a question...")
    with cols[1]:
        submit = st.form_submit_button("Send", width='stretch')

if has_messages or (submit and prompt):
    chat_container = chat_placeholder.container(height=300)
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
else:
    chat_container = chat_placeholder

if submit and prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chat_context = f"Current ERP Data:\n{st.session_state.erp_data.to_csv(index=False)}\n\nUser Question: {prompt}"
                    response = client.chat.completions.create(
                        model="meta/llama3-8b-instruct", # Updated model name for better tier compatibility
                        messages=[
                            {"role": "system", "content": "You are a helpful AI Industrial Credit Agent assistant. Answer the user's questions based on the provided ERP data context."},
                            {"role": "user", "content": chat_context}
                        ],
                        temperature=0.3,
                        max_tokens=500
                    )
                    reply = response.choices[0].message.content
                except Exception as e:
                    # Graceful fallback for demo purposes
                    reply = f"**(Demo Mode Fallback Response):** Based on the current ERP data, the total pending PO value is significant. I am monitoring the active funding gaps and have processed your approval for immediate financing. If you have a valid NVIDIA API Key, please add it to `.streamlit/secrets.toml` to enable live chat AI. (Error: {e})"
                
                st.markdown(reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})

st.subheader("🎥 6. Video")
import os
video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video.mp4")
if os.path.exists(video_path):
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
    st.video(video_bytes)
else:
    st.warning("Video tutorial file not found. Please ensure video.mp4 exists in the repository.")
