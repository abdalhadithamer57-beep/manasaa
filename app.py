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
except ImportError:
    RAG_AVAILABLE = False

# 1. إعداد مفتاح الـ API بشكل آمن للنشر
# يتم سحب المفتاح من إعدادات Secrets في Streamlit Cloud لضمان الأمان
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # ملاحظة: إذا كنت تشغل الكود محلياً، يمكنك وضع المفتاح هنا مؤقتاً أو استخدام ملف .env
    GROQ_API_KEY = "" 
    if not GROQ_API_KEY:
        st.error("⚠️ لم يتم العثور على مفتاح API. يرجى إضافته في إعدادات Secrets في Streamlit Cloud باسم GROQ_API_KEY.")
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

# --- تصميم واجهة المستخدم باستخدام CSS المتطور ---
st.markdown(f"""
    <style>
    /* تخصيص مظهر التطبيق بالكامل */
    [data-testid="stSidebar"] {{
        display: none !important;
    }}
    
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
        width: 100%;
        max-width: 550px; 
        border: 1px solid rgba(255,255,255,0.5);
    }}
    
    .logo-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 1.5rem;
    }}

    .stButton>button {{
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        font-weight: bold;
        padding: 14px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(30, 58, 138, 0.3);
    }}

    /* تنسيق فقاعات المحادثة */
    .chat-bubble {{
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 10px;
        display: inline-block;
        width: 100%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.1);
    }}
    .user-bubble {{
        background-color: #e3f2fd;
        border-right: 5px solid #1e88e5;
    }}
    .assistant-bubble {{
        background-color: #ffffff;
        border-right: 5px solid #43a047;
    }}

    #MainMenu, footer, header {{visibility: hidden;}}
    
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# 3. إدارة الجلسة وبيانات المستخدم
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

# --- واجهة تسجيل الدخول ---
if st.session_state.user_profile is None:
    container = st.container()
    with container:
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=180)
        else:
            st.markdown("<h1 style='text-align:center; font-size:4rem;'>🧠</h1>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
            <div style='text-align:center; margin-bottom: 2rem;'>
                <h1 style='color: #1e3a8a; font-weight: 800; font-size: 2.2rem;'>المنصة الافتراضية للاستشارات النفسية</h1>
                <h2 style='color: #3b82f6; font-weight: 600; font-size: 1.3rem;'>مركز البحوث النفسية</h2>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
        with col2:
            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            with st.form("admission_form"):
                st.markdown("<h3 style='text-align:center; color:#1e3a8a;'>استمارة التسجيل الإلزامية</h3>", unsafe_allow_html=True)
                
                name = st.text_input("الأسم الكريم", placeholder="يرجى إدخال اسمك الكامل")
                gender = st.radio("الجنس", ["ذكر", "أنثى"], horizontal=True)
                age = st.number_input("العمر (متاح من 18 سنة فما فوق)", min_value=1, max_value=120, value=20)
                education = st.selectbox("المستوى التعليمي", ["", "ثانوي", "بكالوريوس", "ماجستير", "دكتوراه", "أخرى"])
                
                st.markdown("<p style='font-size:0.85rem; color:#ef4444; text-align:center;'>* جميع الحقول مطلوبة للمتابعة</p>", unsafe_allow_html=True)
                
                submit = st.form_submit_button("إرسال البيانات وبدء الجلسة")
                
                if submit:
                    if not name.strip():
                        st.error("⚠️ يرجى تزويدنا بالاسم.")
                    elif education == "":
                        st.error("⚠️ يرجى اختيار المستوى التعليمي.")
                    elif age < 18:
                        st.error("🛑 نعتذر، المنصة مخصصة للبالغين فقط.")
                    else:
                        st.session_state.user_profile = {
                            "name": name, 
                            "age": age, 
                            "education": education,
                            "gender": gender
                        }
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- واجهة المحادثة (بعد تسجيل الدخول) ---
st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem; background: rgba(255,255,255,0.85); padding: 25px; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
        <h2 style="color: #1e3a8a; font-weight: 700;">أهلاً بك، {st.session_state.user_profile['name']}</h2>
        <p style="color: #1e40af; font-size: 1.1rem;">أنت الآن في محادثة آمنة وخاصة مع المستشار الذكي</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_knowledge_context(user_query=None):
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
    
    if not all_text: return "لا توجد مراجع نصية متاحة."
    
    if RAG_AVAILABLE and user_query:
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = text_splitter.split_text(all_text)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embeddings)
            relevant_docs = vector_store.similarity_search(user_query, k=3)
            return "\n".join([doc.page_content for doc in relevant_docs])
        except: return all_text[:8000]
    return all_text[:8000]

if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض سجل الدردشة
for message in st.session_state.messages:
    role_class = "user-bubble" if message["role"] == "user" else "assistant-bubble"
    role_label = "أنت" if message["role"] == "user" else "المستشار الذكي"
    st.markdown(f"""
        <div class="chat-bubble {role_class}">
            <strong>{role_label}:</strong><br>
            {message["content"]}
        </div>
    """, unsafe_allow_html=True)

# معالجة المدخلات الجديدة
if user_input := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"""
        <div class="chat-bubble user-bubble">
            <strong>أنت:</strong><br>
            {user_input}
        </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("جاري التفكير وصياغة الرد..."):
            context = get_knowledge_context(user_input)
            client = Groq(api_key=GROQ_API_KEY)
            
            user_info = st.session_state.user_profile
            system_prompt = f"""
            أنت مستشار نفسي خبير بمركز البحوث النفسية. 
            المستخدم: {user_info['name']}، جنسه {user_info['gender']}، عمره {user_info['age']}، تعليمه {user_info['education']}.
            خاطبه بلباقة، طمئنه، واستخدم المراجع التالية للرد:
            {context}
            """
            
            api_messages = [{"role": "system", "content": system_prompt}]
            for m in st.session_state.messages[-5:]:
                api_messages.append({"role": m["role"], "content": m["content"]})

            completion = client.chat.completions.create(
                messages=api_messages,
                model="llama-3.3-70b-versatile",
                temperature=0.6
            )
            
            response = completion.choices[0].message.content
            st.markdown(f"""
                <div class="chat-bubble assistant-bubble">
                    <strong>المستشار الذكي:</strong><br>
                    {response}
                </div>
            """, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    except Exception:
        st.error("عذراً، حدث خطأ في الاتصال بالخدمة الذكية. يرجى مراجعة إعدادات المفتاح.")