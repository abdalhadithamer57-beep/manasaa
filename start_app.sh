#!/bin/bash
# تفعيل البيئة الوهمية
source venv/bin/activate
# تشغيل التطبيق في الخلفية وإخراج السجل لملف log
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
echo "الموقع يعمل الآن على المنفذ 8501 في الخلفية!"
