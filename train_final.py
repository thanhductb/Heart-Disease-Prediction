# FILE: train_final.py
# Mục tiêu: Huấn luyện mô hình chuẩn 100% logic Y khoa (ca=0 -> Khỏe)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# 1. CẤU HÌNH
DATA_PATH = 'data/heart.csv'  # Đảm bảo file nằm ở đây
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'heart_disease_model.pkl')

# 2. TẢI VÀ XỬ LÝ DỮ LIỆU
if not os.path.exists(DATA_PATH):
    # Thử tìm ở thư mục cha nếu đang chạy trong folder con
    if os.path.exists('../data/heart.csv'):
        DATA_PATH = '../data/heart.csv'
    else:
        print("❌ LỖI: Không tìm thấy file 'data/heart.csv'")
        exit()

print(f"🔄 Đang đọc dữ liệu từ: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# --- QUAN TRỌNG: ĐẢO NHÃN (FIX LỖI NGƯỢC LOGIC) ---
# Dữ liệu gốc: 1=Khỏe, 0=Bệnh
# Chúng ta đổi thành: 0=Khỏe, 1=Bệnh (để khớp với Web)
df['target'] = 1 - df['target']
print("✅ Đã đảo nhãn thành công (1 - target).")

# Kiểm tra nhanh: Nhóm ca=0 (mạch máu sạch) thì tỷ lệ bệnh phải thấp
sick_rate_ca0 = df[df['ca'] == 0]['target'].mean()
print(f"🧐 Kiểm tra Logic Y khoa: Tỷ lệ bệnh ở nhóm ca=0 là {sick_rate_ca0*100:.1f}% (Nên thấp < 30%)")
# ----------------------------------------------------

# 3. TIỀN XỬ LÝ (ONE-HOT ENCODING)
# Mã hóa giống hệt quy trình chuẩn
df = pd.get_dummies(df, columns=['cp', 'thal', 'slope'], drop_first=True)

X = df.drop('target', axis=1)
y = df['target']

# 4. HUẤN LUYỆN
print("🔄 Đang huấn luyện Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Đánh giá
acc = accuracy_score(y_test, model.predict(X_test))
print(f"🎉 Độ chính xác mô hình: {acc*100:.2f}%")

# 5. LƯU MÔ HÌNH
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"💾 Đã lưu mô hình MỚI NHẤT tại: {MODEL_PATH}")

# 6. KIỂM TRA THỬ NGAY LẬP TỨC (SANITY CHECK)
print("\n--- TEST NHANH MÔ HÌNH VỪA TẠO ---")
# Tạo một người giả định: Rất khỏe (ca=0, thalach cao, không đau ngực)
# Lưu ý: Tạo dataframe rỗng đúng cấu trúc feature
test_person = pd.DataFrame(0, index=[0], columns=X.columns)
test_person['age'] = 30
test_person['ca'] = 0     # Mấu chốt: ca=0
test_person['thalach'] = 160
test_person['oldpeak'] = 0

pred = model.predict(test_person)[0]
prob = model.predict_proba(test_person)[0][1]

print(f"Người thử nghiệm (ca=0, tuổi 30):")
print(f" - Dự đoán: {pred} (0=An toàn, 1=Nguy cơ)")
print(f" - Xác suất bệnh: {prob:.2f}")

if pred == 0:
    print("✅ KẾT QUẢ: CHUẨN! Người khỏe được báo là An toàn.")
else:
    print("❌ KẾT QUẢ: SAI! Vẫn bị ngược. Cần kiểm tra lại code.")