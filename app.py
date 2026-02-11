import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
import os

# محاولة استيراد المكتبات المتقدمة للمعالجة الذكية للنصوص (RAG)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

# 1. إعداد مفتاح الـ API بشكل آمن للنشر
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    st.warning("⚙️ نظام الأمان: يرجى إضافة GROQ_API_KEY في إعدادات Secrets بالمنصة.")
    st.stop()

# 2. إعدادات الصفحة الأساسية
st.set_page_config(
    page_title="المنصة الافتراضية للاستشارات النفسية", 
    page_icon="🧠", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- إعدادات الموارد البصرية ---
BACKGROUND_IMAGE_URL = "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&q=80&w=2070"
LOGO_PATH = "my_logo.png" 

# --- تصميم واجهة المستخدم باستخدام CSS ---
st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{ display: none !important; }}
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.4)), url("{BACKGROUND_IMAGE_URL}");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }}
    .login-card {{
        background: rgba(255, 255, 255, 0.9);
        padding: 2.5rem;
        border-radius: 25px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.1);
        backdrop-filter: blur(12px);
        margin: 2rem auto;
        max-width: 550px; 
    }}
    .chat-bubble {{
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 10px;
        display: inline-block;
        width: 100%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }}
    .user-bubble {{ background-color: #e3f2fd; border-right: 5px solid #1e88e5; }}
    .assistant-bubble {{ background-color: #ffffff; border-right: 5px solid #43a047; }}
    #MainMenu, footer, header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# 3. إدارة الجلسة
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

# --- واجهة تسجيل الدخول ---
if st.session_state.user_profile is None:
    st.markdown("<h1 style='text-align:center; font-size:4rem;'>🧠</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center; color: #1e3a8a;'>المنصة الافتراضية للاستشارات النفسية</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        with st.form("admission_form"):
            name = st.text_input("الأسم الكريم")
            gender = st.radio("الجنس", ["ذكر", "أنثى"], horizontal=True)
            age = st.number_input("العمر", min_value=18, max_value=120, value=25)
            education = st.selectbox("المستوى التعليمي", ["", "ثانوي", "بكالوريوس", "ماجستير", "دكتوراه"])
            submit = st.form_submit_button("بدء الجلسة")
            if submit and name and education:
                st.session_state.user_profile = {"name": name, "age": age, "education": education, "gender": gender}
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- وظيفة استخراج النصوص والبحث المطور ---
@st.cache_resource
def get_knowledge_context(user_query=""):
    knowledge_dir = "docs"
    all_text = ""
    if os.path.exists(knowledge_dir):
        for filename in os.listdir(knowledge_dir):
            if filename.endswith(".pdf"):
                try:
                    path = os.path.join(knowledge_dir, filename)
                    reader = PdfReader(path)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text: all_text += text + "\n"
                except: continue 
    
    if not all_text:
        return "لا توجد ملفات في مجلد docs."

    # إذا كانت المكتبات المتقدمة تعمل، نستخدم البحث الذكي
    if RAG_AVAILABLE and user_query:
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(all_text)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embeddings)
            relevant_docs = vector_store.similarity_search(user_query, k=4)
            return "\n".join([doc.page_content for doc in relevant_docs])
        except Exception:
            pass # ننتقل للبحث البسيط في حال الفشل
            
    # البحث البسيط (Fallback): نأخذ أول 10 آلاف حرف لضمان وجود سياق
    return all_text[:12000]

# --- واجهة المحادثة ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role_class = "user-bubble" if message["role"] == "user" else "assistant-bubble"
    st.markdown(f'<div class="chat-bubble {role_class}"><strong>{"أنت" if message["role"]=="user" else "المستشار"}:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)

if user_input := st.chat_input("اسأل مستشارك النفسي..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_input = st.session_state.messages[-1]["content"]
    with st.spinner("جاري مراجعة المراجع العلمية..."):
        try:
            context = get_knowledge_context(user_input)
            client = Groq(api_key=GROQ_API_KEY)
            user_info = st.session_state.user_profile
            
            system_prompt = f"""
            أنت مستشار نفسي خبير. التزم بالتعليمات التالية بدقة:
            1. أجب حصراً وبناءً على المعلومات الواردة في المراجع المرفقة أدناه.
            2. إذا لم تجد الإجابة في المراجع، قل بلباقة أنك لا تملك معلومة حول هذا الأمر في مصادرك الحالية.
            3. خاطب المستخدم ({user_info['name']}) بما يناسب عمره ({user_info['age']}) وجنسه.
            
            المراجع المتاحة:
            {context}
            """
            
            api_messages = [{"role": "system", "content": system_prompt}]
            api_messages.extend(st.session_state.messages[-5:])

            completion = client.chat.completions.create(
                messages=api_messages,
                model="llama-3.3-70b-versatile",
                temperature=0.3 # تقليل العشوائية لضمان الالتزام بالنص
            )
            
            response = completion.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        except Exception as e:
            st.error("حدث خطأ في استرجاع البيانات.")
