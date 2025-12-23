import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Dự đoán Bệnh Tim Mạch",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH (Làm đẹp giao diện) ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #e63939;
        border-color: #e63939;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HÀM LOAD MODEL THÔNG MINH ---
@st.cache_resource
def load_model():
    # Danh sách các đường dẫn có thể chứa file model
    possible_paths = [
        'models/heart_disease_model.pkl',           # Đường dẫn chuẩn (khi chạy từ thư mục gốc)
        'notebooks/models/heart_disease_model.pkl', # Đường dẫn nếu file bị lưu nhầm vào notebook
        'heart_disease_model.pkl'                   # Đường dẫn nếu file nằm ngay thư mục gốc
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as file:
                    model = pickle.load(file)
                return model
            except Exception as e:
                st.error(f"⚠️ Lỗi khi đọc file mô hình tại {path}: {e}")
                return None
            
    # Nếu chạy hết vòng lặp mà không return được model nào
    st.error("⚠️ KHÔNG TÌM THẤY FILE MÔ HÌNH! Vui lòng kiểm tra lại thư mục 'models/'.")
    return None

model = load_model()

# --- HÀM XỬ LÝ DỮ LIỆU ĐẦU VÀO (PREPROCESSING) ---
# Hàm này chuyển đổi dữ liệu từ Web thành dạng mà Model Random Forest hiểu (One-Hot Encoding)
def preprocess_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # 1. Định nghĩa các cột chuẩn (PHẢI KHỚP với X_train.columns lúc train)
    # Dựa trên pd.get_dummies(drop_first=True)
    columns = [
        'age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca',
        'cp_1', 'cp_2', 'cp_3',          # cp_0 bị drop
        'thal_1', 'thal_2', 'thal_3',    # thal_0 bị drop
        'slope_1', 'slope_2'             # slope_0 bị drop
    ]
    
    # Tạo DataFrame 1 dòng chứa toàn số 0
    input_df = pd.DataFrame(0, index=[0], columns=columns)
    
    # 2. Gán giá trị số & Nhị phân cơ bản
    input_df['age'] = age
    input_df['sex'] = 1 if sex == "Nam" else 0
    input_df['trestbps'] = trestbps
    input_df['chol'] = chol
    input_df['fbs'] = 1 if fbs == "Đúng (True)" else 0
    input_df['thalach'] = thalach
    input_df['exang'] = 1 if exang == "Có" else 0
    input_df['oldpeak'] = oldpeak
    input_df['ca'] = ca
    
    # Map RestECG (0,1,2)
    ecg_map = {"Bình thường": 0, "Sóng ST-T bất thường": 1, "Phì đại thất trái": 2}
    input_df['restecg'] = ecg_map[restecg]
    
    # 3. Xử lý One-Hot Encoding (Gán số 1 vào đúng cột dummy)
    
    # CP (Loại đau ngực) - Map về 0,1,2,3
    cp_map_val = {
        "Điển hình (Typical Angina)": 0,
        "Không điển hình (Atypical Angina)": 1, 
        "Đau không do tim (Non-anginal)": 2, 
        "Không triệu chứng (Asymptomatic)": 3
    }
    val = cp_map_val[cp]
    if val in [1, 2, 3]: # Nếu là 0 thì tất cả cột cp_1, cp_2, cp_3 đều bằng 0 (Reference)
        if f'cp_{val}' in input_df.columns:
            input_df[f'cp_{val}'] = 1
            
    # Slope (Độ dốc) - Map về 0,1,2
    slope_map_val = {"Lên (Upsloping)": 0, "Bằng (Flat)": 1, "Xuống (Downsloping)": 2}
    val = slope_map_val[slope]
    if val in [1, 2]:
        if f'slope_{val}' in input_df.columns:
            input_df[f'slope_{val}'] = 1

    # Thal (Thalassemia) - Map về 0,1,2,3
    # Lưu ý: Cần khớp với dataset. Giả sử dataset có 0,1,2,3.
    thal_map_val = {"Không rõ": 0, "Bình thường": 1, "Lỗi cố định": 2, "Lỗi có thể đảo ngược": 3}
    val = thal_map_val[thal]
    if val in [1, 2, 3]:
        if f'thal_{val}' in input_df.columns:
            input_df[f'thal_{val}'] = 1
            
    return input_df

# --- THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.title("Trợ lý Tim mạch AI")
    st.info("Hệ thống hỗ trợ chẩn đoán nguy cơ bệnh mạch vành dựa trên Machine Learning.")
    st.write("---")
    st.write("👨‍🎓 **Sinh viên:** Lâm Thanh Đức")
    st.write("🆔 **MSSV:** 23730078")
    st.write("🏫 **Lớp:** CN1.K2023.2")
    st.caption("© 2025 Heart Disease Prediction Project")

# --- TIÊU ĐỀ CHÍNH ---
st.title("💓 Hệ thống Dự đoán Nguy cơ Bệnh Tim Mạch")
st.markdown("**Hướng dẫn:** Nhập đầy đủ các chỉ số lâm sàng bên dưới và nhấn nút **Phân tích**.")

# --- GIAO DIỆN NHẬP LIỆU (TABS) ---
tab1, tab2 = st.tabs(["📝 Nhập liệu Chẩn đoán", "ℹ️ Ý nghĩa chỉ số"])

with tab1:
    with st.form("medical_form"):
        st.subheader("Thông tin bệnh nhân")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 1. Thông tin chung")
            age = st.number_input("Tuổi (Age)", 1, 120, 50)
            sex = st.selectbox("Giới tính (Sex)", ["Nam", "Nữ"])
            cp = st.selectbox("Loại đau ngực (Chest Pain)", 
                              ["Điển hình (Typical Angina)", "Không điển hình (Atypical Angina)", 
                               "Đau không do tim (Non-anginal)", "Không triệu chứng (Asymptomatic)"])
        
        with col2:
            st.markdown("##### 2. Chỉ số sinh tồn")
            # Đã cập nhật label theo yêu cầu
            trestbps = st.number_input("Huyết áp tâm thu (mm Hg)", 50, 300, 120, help="Số trên (Systolic) khi đo huyết áp.")
            thalach = st.number_input("Nhịp tim tối đa (Max HR)", 30, 250, 150)
            exang = st.selectbox("Đau ngực khi tập thể dục?", ["Không", "Có"])

        with col3:
            st.markdown("##### 3. Chỉ số xét nghiệm")
            chol = st.number_input("Cholesterol (mg/dl)", 80, 600, 200)
            fbs = st.selectbox("Đường huyết lúc đói > 120?", ["Sai (False)", "Đúng (True)"])
            restecg = st.selectbox("Điện tâm đồ lúc nghỉ", ["Bình thường", "Sóng ST-T bất thường", "Phì đại thất trái"])

        st.write("---")
        st.markdown("##### 4. Chỉ số chuyên sâu (Kết quả chụp chiếu)")
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        with c_col1:
            oldpeak = st.number_input("Đoạn ST chênh (Oldpeak)", 0.0, 10.0, 0.0, help="Độ chênh xuống của đoạn ST so với đường đẳng điện.")
        with c_col2:
            slope = st.selectbox("Độ dốc đoạn ST (Slope)", ["Lên (Upsloping)", "Bằng (Flat)", "Xuống (Downsloping)"])
        with c_col3:
            ca = st.selectbox("Số mạch máu chính (0-3)", [0, 1, 2, 3])
        with c_col4:
            thal = st.selectbox("Thalassemia", ["Không rõ", "Bình thường", "Lỗi cố định", "Lỗi có thể đảo ngược"], index=1)

        st.write("")
        submit_button = st.form_submit_button("🚀 PHÂN TÍCH NGUY CƠ NGAY", use_container_width=True)

    # --- XỬ LÝ SỰ KIỆN KHI BẤM NÚT ---
    if submit_button:
        
        # 1. VALIDATION LOGIC (Kiểm tra dữ liệu vô lý - Yêu cầu của Thầy)
        warning_msg = []
        
        # Logic Huyết áp & Nhịp tim
        if trestbps > 200 and thalach < 60:
            warning_msg.append("⚠️ **Cảnh báo dữ liệu:** Huyết áp rất cao (>200) nhưng nhịp tim lại thấp (<60). Vui lòng kiểm tra lại.")
        
        # Logic Cholesterol theo độ tuổi
        if chol > 600:
            warning_msg.append("⚠️ **Nghi vấn:** Chỉ số Cholesterol > 600 là cực kỳ hiếm gặp.")
        elif age < 30 and chol > 260:
            warning_msg.append("⚠️ **Cảnh báo Y khoa (Nhóm trẻ):** Tuổi < 30 nhưng Cholesterol cao (>260). Có thể do di truyền.")
        elif 30 <= age <= 50 and chol > 240:
             warning_msg.append("⚠️ **Cảnh báo (Trung niên):** Cholesterol cao (>240). Cần điều chỉnh lối sống.")
        elif age > 50 and chol > 280:
             warning_msg.append("ℹ️ **Lưu ý (Cao tuổi):** Cholesterol cao (>280), tăng nguy cơ xơ vữa.")
             
        # Logic Đau ngực & Nhịp tim
        if exang == "Có" and thalach < 80:
             warning_msg.append("⚠️ **Logic:** Có đau ngực khi gắng sức nhưng nhịp tim tối đa lại thấp (<80).")

        # Hiển thị cảnh báo
        if warning_msg:
            for msg in warning_msg: st.warning(msg)

        # 2. CHẠY MÔ HÌNH AI (PREDICT)
        if model:
            with st.spinner("Đang xử lý dữ liệu..."):
                time.sleep(1) # Hiệu ứng chờ
                try:
                    # A. Tiền xử lý dữ liệu
                    input_df = preprocess_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
                    
                    # B. Dự đoán
                    prediction = model.predict(input_df)[0]      # Kết quả: 0 hoặc 1
                    probability = model.predict_proba(input_df)[0][1] # Xác suất bị bệnh (0.0 - 1.0)
                    risk_score = round(probability * 100, 1)

                    # C. Hiển thị kết quả
                    st.write("---")
                    st.subheader("📋 Kết quả Phân tích từ AI")
                    
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        if prediction == 1:
                            st.error(f"NGUY CƠ CAO: {risk_score}%")
                            st.metric("Kết luận", "Có dấu hiệu bệnh", delta="-Cần đi khám ngay", delta_color="inverse")
                        else:
                            st.success(f"NGUY CƠ THẤP: {risk_score}%")
                            st.metric("Kết luận", "An toàn", delta="+Duy trì sức khỏe", delta_color="normal")
                    
                    with res_col2:
                        st.write("Thanh mức độ rủi ro:")
                        if risk_score > 50:
                            st.progress(int(risk_score), text="Cảnh báo nguy hiểm")
                            st.warning(f"Mô hình dự đoán bạn có **{risk_score}%** nguy cơ mắc bệnh tim mạch dựa trên các chỉ số đầu vào.")
                        else:
                            st.progress(int(risk_score), text="Trong ngưỡng an toàn")
                            st.info(f"Mô hình dự đoán bạn chỉ có **{risk_score}%** nguy cơ. Hãy tiếp tục sống lành mạnh!")
                            
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi khi chạy mô hình: {e}")
                    st.markdown("👉 **Gợi ý:** Có thể do số lượng cột trong file Model không khớp với code xử lý. Hãy kiểm tra lại phần Preprocessing.")
        else:
             st.error("Chưa tải được mô hình. Vui lòng kiểm tra lại file .pkl")

with tab2:
    st.header("Ý nghĩa các chỉ số lâm sàng")
    st.markdown("""
    | Chỉ số | Ý nghĩa y học | Ngưỡng tham khảo |
    | :--- | :--- | :--- |
    | **Age (Tuổi)** | Nguy cơ tăng theo tuổi tác. | - |
    | **Huyết áp (Trestbps)** | Áp lực máu lên động mạch. | > 140 mmHg là cao huyết áp. |
    | **Cholesterol** | Mỡ trong máu. | > 240 mg/dl là cao. |
    | **Fbs (Đường huyết)** | Đường huyết lúc đói. | > 120 mg/dl gợi ý tiểu đường. |
    | **Thalach (Nhịp tim)** | Nhịp tim tối đa khi gắng sức. | Giảm dần theo tuổi (220 - tuổi). |
    | **Exang (Đau ngực tập)** | Đau ngực khi vận động. | Dấu hiệu điển hình của thiếu máu cơ tim. |
    """)
    st.info("Nguồn dữ liệu: UCI Machine Learning Repository - Cleveland Dataset.")