
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# إعداد المسارات النسبية داخل مجلد المشروع
base_path = "./"
status_model_path = os.path.join(base_path, "best_lgbm_status_model.pkl")
alarm_model_path = os.path.join(base_path, "best_lgbm_alarm_model.pkl")
scaler_path = os.path.join(base_path, "scaler.pkl")
frame_path = os.path.join(base_path, "frame.xlsx")

# تحميل النماذج
status_model = joblib.load(status_model_path)
alarm_model = joblib.load(alarm_model_path)
scaler = joblib.load(scaler_path)

# ---------- تنسيق CSS احترافي ----------
st.markdown("""
    <style>
        .main {
            background-color: #f5f8fc;
        }
        .title {
            background-color: #003366;
            color: white;
            padding: 16px;
            border-radius: 10px;
            font-size: 28px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .upload-box {
            background-color: #e0ebf5;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .download-button {
            background-color: #005c99;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- واجهة المستخدم ----------
st.markdown('<div class="title">📊 نظام التنبؤ بالحالة العامة ونوع الإنذار للعدادات الذكية</div>', unsafe_allow_html=True)

st.markdown("🔹 يمكنك تحميل النموذج الجاهز للبيانات من الزر أدناه، تعبئته ورفعه مرة أخرى للنظام:")
with open(frame_path, "rb") as f:
    st.download_button("⬇️ تحميل قالب البيانات (frame.xlsx)", f, file_name="frame.xlsx")

st.markdown('<div class="upload-box">📎 قم برفع ملف Excel يحتوي على الأعمدة التالية: `Meter Number`, `V1`, `V2`, `V3`, `A1`, `A2`, `A3`</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 رفع الملف", type=["xlsx"])

# ---------- تحليل البيانات ----------
if uploaded_file:
    try:
        df_input = pd.read_excel(uploaded_file)
        required_columns = ['Meter Number', 'V1', 'V2', 'V3', 'A1', 'A2', 'A3']

        if not all(col in df_input.columns for col in required_columns):
            st.error("❌ الملف لا يحتوي على الأعمدة المطلوبة.")
        else:
            features = ['V1', 'V2', 'V3', 'A1', 'A2', 'A3']
            X = df_input[features]
            X_scaled = scaler.transform(X)

            # نموذج الحالة العامة
            status_proba = status_model.predict_proba(X_scaled)
            status_preds = status_model.predict(X_scaled)
            status_conf = status_proba.max(axis=1) * 100

            df_input['الحالة المتوقعة'] = status_preds
            df_input['نسبة الثقة بالحالة (%)'] = status_conf.round(2)

            # نموذج الإنذار لغير "سليم"
            non_ok_indices = df_input['الحالة المتوقعة'] != 'سليم'
            X_alarm = X[non_ok_indices]
            X_alarm_scaled = scaler.transform(X_alarm)

            alarm_proba = alarm_model.predict_proba(X_alarm_scaled)
            alarm_preds = alarm_model.predict(X_alarm_scaled)
            alarm_conf = alarm_proba.max(axis=1) * 100

            df_input['نوع الإنذار المتوقع'] = "لا يوجد"
            df_input['نسبة الثقة بالإنذار (%)'] = np.nan
            df_input.loc[non_ok_indices, 'نوع الإنذار المتوقع'] = alarm_preds
            df_input.loc[non_ok_indices, 'نسبة الثقة بالإنذار (%)'] = alarm_conf.round(2)

            # عرض النتائج
            st.success("✅ تم تحليل الملف بنجاح")
            st.dataframe(df_input)

            # تحميل النتائج
            output_file = os.path.join(base_path, "predicted_results.xlsx")
            df_input.to_excel(output_file, index=False)
            with open(output_file, "rb") as f:
                st.download_button("📥 تحميل النتائج", f, file_name="predicted_results.xlsx")

    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء التحليل: {e}")
