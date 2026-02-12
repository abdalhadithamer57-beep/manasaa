import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

# تحميل المتغيرات من ملف .env إذا كان موجوداً
load_dotenv()

# 1. إعدادات الصفحة لضمان الظهور على كل الأجهزة
st.set_page_config(page_title="مركز البحوث النفسية", layout="wide")

# 2. كود CSS قوي جداً لإجبار النص على الظهور (Force Display)
st.markdown("""
    <style>
    /* إجبار الحاوية الرئيسية على عرض كل المحتوى */
    .main .block-container {
        display: block !important;
        visibility: visible !important;
        direction: rtl !important;
    }
    
    /* تنسيق الفقاعات لضمان عدم تداخلها أو اختفائها في iOS */
    .answer-box {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: #000000 !important;
        display: block !important;
        min-height: 50px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* إخفاء العناصر التي قد تحجب الرؤية */
    footer, header { visibility: hidden !important; }
    </style>
    """, unsafe_allow_html=True)

# جلب المفتاح من ملف .env أو من إعدادات النظام
api_key = os.getenv("GROQ_API_KEY")

if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل القديمة
for msg in st.session_state.messages:
    role_name = "أنت" if msg["role"] == "user" else "المستشار"
    st.markdown(f'<div class="answer-box"><strong>{role_name}:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

# إدخال المستخدم
if prompt := st.chat_input("تحدث إلينا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="answer-box"><strong>أنت:</strong> {prompt}</div>', unsafe_allow_html=True)

    # التحقق من وجود المفتاح قبل الطلب
    if not api_key:
        st.error("خطأ: لم يتم العثور على GROQ_API_KEY. تأكد من إنشاء ملف .env في السيرفر.")
    else:
        # طلب الإجابة من Groq مع معالجة الأخطاء
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "أنت مستشار نفسي خبير."}] + st.session_state.messages
            )
            full_response = response.choices[0].message.content
            
            # إذا كانت الإجابة فارغة لسبب ما
            if not full_response:
                full_response = "نعتذر، لم أتمكن من صياغة الإجابة حالياً. حاول مرة أخرى."
                
            st.markdown(f'<div class="answer-box"><strong>المستشار:</strong> {full_response}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"حدث خطأ فني: {str(e)}")
