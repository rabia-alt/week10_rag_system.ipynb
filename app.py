import os
try:
    from groq import Groq
except ModuleNotFoundError:
    os.system('pip install groq streamlit chromadb -q')
    from groq import Groq

import streamlit as st
import chromadb

# Page Configuration
st.set_page_config(page_title='Enterprise Knowledge Hub', layout='wide', page_icon='🏢')

# Custom UI Styling (Premium Dark Theme & Clear White Text)
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .css-1d391kg { background-color: #1e293b !important; }
    
    /* Input box and Chat text contrast overrides */
    div[data-testid="stChatMessage"] {
        background-color: #1e293b !important;
        border-radius: 10px;
        margin-bottom: 10px;
        padding: 15px;
    }
    div[data-testid="stChatMessage"] p {
        color: #ffffff !important;
        font-size: 16px !important;
    }
    
    div.stButton > button:first-child {
        background-color: #ef4444; color: white; border-radius: 8px; border: none; width: 100%;
    }
    div.stButton > button:first-child:hover { background-color: #dc2626; color: white; }
    
    .card {
        background-color: #1e293b; padding: 15px; border-radius: 10px; 
        border-left: 5px solid #3b82f6; margin-bottom: 10px; height: 120px; color: #ffffff;
    }
    .status-box {
        background-color: #14532d; color: #4ade80; padding: 10px; 
        border-radius: 8px; font-weight: bold; text-align: center; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# SECURITY UPDATE: GitHub par API key hamesha Streamlit secrets se read ki jati hai
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # Local backup settings (just in case)
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

@st.cache_resource
def init_rag_extended():
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name='company_docs')
    
    documents_list = [
        "Remote Work Guidelines: Employees can work from home up to 3 days per week with manager approval. Core working hours for remote tracking are 10:00 AM to 4:00 PM.",
        "Vacation and Time Off Policies: All full-time employees are entitled to 25 days of annual vacation leave per calendar year. Unused leaves do not carry over to the next year.",
        "Parental Leave Benefits: The company provides 16 weeks of fully paid maternity leave for birth mothers and 4 weeks for fathers.",
        "Office Timings and Punctuality: Standard office hours are 9:00 AM to 5:00 PM, Monday to Friday. A grace period of 15 minutes is allowed for daily check-in.",
        "IT Security and Password Policy: System passwords must be changed every 90 days. Sharing passwords or using unauthorized USB drives on office laptops is strictly prohibited.",
        "Salary Review and Appraisals: Performance appraisals happen annually in December. Salary increments are based on the performance rating, ranging from 5% up to 15%.",
        "Dress Code Policy: Business casuals are required from Monday to Thursday. Casual attire, including jeans and sneakers, is permitted only on Fridays."
    ]
    
    dummy_embeddings = [[0.0] * 1536 for _ in range(len(documents_list))]
    collection.add(
        documents=documents_list,
        embeddings=dummy_embeddings,
        ids=[f"doc_{i}" for i in range(len(documents_list))]
    )
    return collection

if GROQ_API_KEY:
    collection = init_rag_extended()
else:
    st.error("🔑 Groq API Key missing! Please configure it in Streamlit Secrets.")

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    if GROQ_API_KEY:
        st.markdown('<div class="status-box">🔒 Knowledge Base Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box" style="background-color:#7f1d1d; color:#fca5a5;">⚠️ Key Missing</div>', unsafe_allow_html=True)
        
    st.subheader("💡 Testing Examples")
    st.markdown("""
    * *What is the password update policy?*
    * *Can I use a USB drive on my laptop?*
    * *What is the grace period for check-in?*
    * *How much increment can I get?*
    * *When do appraisals happen?*
    """)
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# MAIN INTERFACE
st.title('🏢 Company Knowledge Assistant')
st.caption('Ask any policy question. The AI will retrieve context from ChromaDB and answer instantly.')

# Welcome Cards
if 'messages' not in st.session_state or len(st.session_state.messages) == 0:
    st.markdown("### 🌟 Suggested Categories:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><b>🔐 IT Security</b><br>Try asking about passwords or security hardware guidelines.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><b>💰 Appraisals & Dress Code</b><br>Try asking about performance reviews or corporate dress codes.</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><b>📅 Leave & Timings</b><br>Try asking about work hours, check-in rules, or annual leaves.</div>', unsafe_allow_html=True)
    st.markdown("---")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "👤" if message['role'] == 'user' else "🤖"
    with st.chat_message(message['role'], avatar=avatar):
        st.write(message['content'])

if prompt := st.chat_input('Type your question here...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user', avatar="👤"):
        st.write(prompt)
        
    with st.spinner('🔍 Querying ChromaDB & Llama...'):
        try:
            all_docs = collection.get()
            context = "\n".join(all_docs['documents']) if all_docs['documents'] else "No context found"
            
            system_prompt = (
                f"You are a professional corporate assistant. Answer the user's question using ONLY the context provided below. "
                f"CRITICAL: Your response must be entirely in English language under all circumstances.\n"
                f"If the information is not present in the context, reply exactly with: 'I cannot find this in the company documents.'\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {prompt}"
            )
            
            client = Groq(api_key=GROQ_API_KEY)
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": system_prompt}],
                temperature=0
            )
            answer = completion.choices[0].message.content
            full_response = f"💡 **[Context Retrieved From ChromaDB Memory Successfully]**\n\n{answer}"
        except Exception as e:
            full_response = f"⚠️ Error: {str(e)}"

    st.session_state.messages.append({'role': 'assistant', 'content': full_response})
    with st.chat_message('assistant', avatar="🤖"):
        st.write(full_response)
