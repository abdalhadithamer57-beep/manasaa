import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# تحميل المتغيرات من ملف .env (للتشغيل المحلي أو على خادم VPS)
# ملاحظة: يجب إنشاء ملف باسم .env ووضع GROQ_API_KEY=your_key_here بداخله
load_dotenv()

# محاولة استيراد المكتبات المتقدمة للمعالجة الذكية للنصوص (RAG)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

# 1. إعداد مفتاح الـ API بشكل آمن
# الأولوية لـ Streamlit Secrets (للنشر السحابي) ثم لبيئة النظام (للتشغيل المحلي)
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# تنبيه للمستخدم في حال عدم العثور على المفتاح
if not GROQ_API_KEY:
    st.error("⚠️ خطأ في المصادقة: لم يتم العثور على مفتاح API الخاص بـ Groq.")
    st.info("لحل المشكلة: أضف المفتاح في إعدادات Secrets في Streamlit أو في ملف .env محلياً باسم GROQ_API_KEY.")
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
        margin: 1rem auto;
        max-width: 550px; 
    }}
    .logo-wrapper {{
        display: flex;
        justify-content: center;
        margin-bottom: -20px;
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
    # عرض الشعار في المنتصف
    col_l, col_m, col_r = st.columns([1, 1, 1])
    with col_m:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown("<h1 style='text-align:center; font-size:5rem;'>🧠</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align:center; color: #1e3a8a; margin-top: -10px;'>المنصة الافتراضية للاستشارات النفسية</h2>", unsafe_allow_html=True)
    
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

    if RAG_AVAILABLE and user_query:
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(all_text)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embeddings)
            relevant_docs = vector_store.similarity_search(user_query, k=4)
            return "\n".join([doc.page_content for doc in relevant_docs])
        except Exception:
            pass 
            
    return all_text[:12000]

# --- واجهة المحادثة ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض اسم المستخدم في الأعلى بشكل هادئ
st.markdown(f"### مرحباً {st.session_state.user_profile['name']} | جلسة استشارية آمنة")

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
                temperature=0.3 
            )
            
            response = completion.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        except Exception as e:
            st.error("حدث خطأ في استرجاع البيانات.")
