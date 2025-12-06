import streamlit as st
import time

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Dự đoán Bệnh Tim Mạch",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.title("Trợ lý Tim mạch AI")
    st.info("Ứng dụng sử dụng Machine Learning để đánh giá nguy cơ mắc bệnh tim mạch dựa trên 13 chỉ số lâm sàng.")
    st.write("---")
    st.write("👨‍🎓 **Sinh viên:** Lâm Thanh Đức")
    st.write("🆔 **MSSV:** 23730078")
    st.write("🏫 **Lớp:** CN1.K2023.2")
    st.write("---")
    st.caption("© 2025 Heart Disease Prediction Project")

# --- TIÊU ĐỀ CHÍNH ---
st.title("💓 Hệ thống Dự đoán Nguy cơ Bệnh Tim Mạch")
st.markdown("*Nhập các chỉ số sức khỏe của bạn vào bên dưới để nhận kết quả phân tích.*")

# --- GIAO DIỆN NHẬP LIỆU (CHIA TAB) ---
tab1, tab2 = st.tabs(["📝 Nhập liệu Chẩn đoán", "ℹ️ Hướng dẫn & Ý nghĩa chỉ số"])

with tab1:
    # Tạo Form để gom nhóm các input, tránh reload trang liên tục
    with st.form("medical_form"):
        st.subheader("Thông tin bệnh nhân")
        
        # Chia cột cho đẹp (Cột 1: Cá nhân, Cột 2: Sinh tồn, Cột 3: Xét nghiệm)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 1. Thông tin chung")
            age = st.number_input("Tuổi (Age)", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Giới tính (Sex)", options=["Nam", "Nữ"])
            cp = st.selectbox("Loại đau ngực (Chest Pain)", 
                              options=["Điển hình (Typical Angina)", 
                                       "Không điển hình (Atypical Angina)", 
                                       "Đau không do tim (Non-anginal)", 
                                       "Không triệu chứng (Asymptomatic)"])
        
        with col2:
            st.markdown("##### 2. Chỉ số sinh tồn")
            trestbps = st.number_input("Huyết áp lúc nghỉ (mm Hg)", min_value=50, max_value=300, value=120)
            thalach = st.number_input("Nhịp tim tối đa (Max Heart Rate)", min_value=30, max_value=250, value=150)
            exang = st.selectbox("Đau ngực khi tập thể dục?", options=["Không", "Có"])

        with col3:
            st.markdown("##### 3. Chỉ số xét nghiệm")
            chol = st.number_input("Cholesterol (mg/dl)", min_value=80, max_value=600, value=200)
            fbs = st.selectbox("Đường huyết lúc đói > 120 mg/dl?", options=["Sai (False)", "Đúng (True)"])
            restecg = st.selectbox("Điện tâm đồ lúc nghỉ", 
                                   options=["Bình thường", "Sóng ST-T bất thường", "Phì đại thất trái"])

        st.write("---")
        # Các chỉ số chuyên sâu hơn (xếp hàng ngang dưới cùng)
        st.markdown("##### 4. Chỉ số chuyên sâu (Thường có trong kết quả chụp chiếu)")
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        with c_col1:
            oldpeak = st.number_input("Đoạn ST chênh xuống (Oldpeak)", 0.0, 10.0, 0.0)
        with c_col2:
            slope = st.selectbox("Độ dốc đoạn ST (Slope)", options=["Lên (Upsloping)", "Bằng (Flat)", "Xuống (Downsloping)"])
        with c_col3:
            ca = st.selectbox("Số mạch máu chính (0-3)", options=[0, 1, 2, 3])
        with c_col4:
            thal = st.selectbox("Thalassemia", options=["Bình thường", "Lỗi cố định", "Lỗi có thể đảo ngược"])

        st.write("")
        # NÚT DỰ ĐOÁN (Trung tâm của Form)
        submit_button = st.form_submit_button("🚀 PHÂN TÍCH NGUY CƠ NGAY", use_container_width=True)

    # --- XỬ LÝ KẾT QUẢ (BAO GỒM VALIDATION & LOGIC GIẢ LẬP) ---
    if submit_button:
        
        # --- PHẦN VALIDATE DỮ LIỆU (MỚI THÊM VÀO) ---
        warning_msg = []
        
        # Logic 1: Huyết áp quá cao nhưng nhịp tim quá thấp (Vô lý về mặt sinh lý học)
        if trestbps > 200 and thalach < 55:
            warning_msg.append("⚠️ **Cảnh báo dữ liệu:** Huyết áp lúc nghỉ rất cao (>200) nhưng nhịp tim tối đa lại quá thấp (<55). Vui lòng kiểm tra lại thiết bị đo.")
            
        # Logic 2: Tuổi trẻ nhưng Cholesterol quá cao (Cần chú ý đặc biệt)
        if chol > 600:
             warning_msg.append("⚠️ **Nghi vấn nhập liệu:** Chỉ số Cholesterol > 600 mg/dl là cực kỳ hiếm gặp. Bạn có chắc mình nhập đúng không?")
        
        # Trường hợp B: Nhóm TRẺ (<30 tuổi) - Cảnh báo về Gen di truyền
        elif age < 30 and chol > 260:
             warning_msg.append("⚠️ **Cảnh báo Y khoa (Nhóm trẻ):** Bệnh nhân còn trẻ (<30) nhưng Cholesterol rất cao. Đây có thể là dấu hiệu rối loạn mỡ máu di truyền.")
             
        # Trường hợp C: Nhóm TRUNG NIÊN (30-50 tuổi) - Cảnh báo về Lối sống
        elif 30 <= age <= 50 and chol > 240:
             warning_msg.append("⚠️ **Cảnh báo (30-50 tuổi):** Chỉ số Cholesterol vượt ngưỡng an toàn (>240 mg/dl). Ở độ tuổi này, cần xem xét lại chế độ ăn uống và vận động.")

        # Trường hợp D: Nhóm CAO TUỔI (>50 tuổi) - Cảnh báo Bệnh lý
        elif age > 50 and chol > 280:
             warning_msg.append("ℹ️ **Lưu ý (Trên 50 tuổi):** Chỉ số Cholesterol đang ở mức cao (>280 mg/dl), nguy cơ xơ vữa động mạch tăng cao.")

        # Logic 3: Có đau ngực khi tập nhưng nhịp tim tối đa ghi nhận lại thấp
        if exang == "Có" and thalach < 70:
             warning_msg.append("⚠️ **Lưu ý:** Bệnh nhân có đau ngực khi gắng sức nhưng nhịp tim tối đa ghi nhận lại khá thấp (<70).")

        # Hiển thị các cảnh báo (nếu có) trước khi chạy AI
        if warning_msg:
            for msg in warning_msg:
                st.warning(msg)

        # --- PHẦN CHẠY AI (HIỆN TẠI LÀ GIẢ LẬP) ---
        with st.spinner("Đang phân tích dữ liệu với AI..."):
            time.sleep(2) # Giả vờ AI đang suy nghĩ
            
            # --- LOGIC GIẢ (DUMMY) ĐỂ TEST GIAO DIỆN ---
            # Ví dụ: Nếu tuổi > 60 hoặc Cholesterol > 250 thì báo nguy cơ cao
            risk_score = 0
            if age > 55: risk_score += 30
            if chol > 240: risk_score += 30
            if trestbps > 140: risk_score += 20
            # Thêm logic giả theo góp ý của thầy (Đau ngực khi tập = trọng số cao)
            if exang == "Có": risk_score += 15 
            
            # Giới hạn max 100
            if risk_score > 100: risk_score = 99

            # Hiển thị kết quả
            st.write("---")
            st.subheader("📋 Kết quả Phân tích")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if risk_score > 50:
                    st.error(f"NGUY CƠ CAO ({risk_score}%)")
                    st.metric(label="Đánh giá", value="Nguy hiểm", delta="-Cần đi khám ngay", delta_color="inverse")
                else:
                    st.success(f"NGUY CƠ THẤP ({risk_score}%)")
                    st.metric(label="Đánh giá", value="An toàn", delta="+Duy trì lối sống", delta_color="normal")
            
            with res_col2:
                st.progress(risk_score)
                if risk_score > 50:
                    st.warning("⚠️ **Cảnh báo:** Dựa trên các chỉ số (đặc biệt là Tuổi và Cholesterol), hệ thống nhận thấy bạn có dấu hiệu rủi ro tim mạch.")
                    st.markdown("- Hãy kiểm tra lại chế độ ăn uống.\n- Giảm lượng muối và chất béo.\n- **Lời khuyên:** Hãy đến cơ sở y tế gần nhất để bác sĩ kiểm tra kỹ hơn.")
                else:
                    st.info("✅ **Tuyệt vời:** Các chỉ số của bạn hiện tại nằm trong ngưỡng an toàn.")
                    st.markdown("- Hãy tiếp tục duy trì chế độ tập luyện.\n- Khám sức khỏe định kỳ 6 tháng/lần.")

with tab2:
    st.header("Ý nghĩa các chỉ số")
    st.markdown("""
    - **Tuổi (Age):** Yếu tố nguy cơ tăng dần theo tuổi.
    - **Huyết áp (Resting BP):** Huyết áp cao làm tăng gánh nặng cho tim.
    - **Cholesterol:** Mỡ máu cao dễ gây xơ vữa động mạch.
    - **Đau ngực (Chest Pain Type):** Vị trí và tính chất đau ngực là dấu hiệu quan trọng.
    """)
    st.image("https://www.cdc.gov/heartdisease/images/heart_failure_symptoms.jpg", caption="Các triệu chứng phổ biến của bệnh tim (Nguồn: CDC)")