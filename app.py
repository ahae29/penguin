import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from palmerpenguins import load_penguins

# 펭귄 데이터셋 로드
penguins = load_penguins()
penguins = penguins.dropna()  # 결측치 제거

# feature와 target 설정
X = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = penguins['species']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 웹앱 타이틀
st.title("펭귄 종류 분류기")

# 사용자 입력 받기
bill_length = st.number_input("부리 길이 (mm)", min_value=30.0, max_value=70.0, value=40.0)
bill_depth = st.number_input("부리 깊이 (mm)", min_value=10.0, max_value=30.0, value=15.0)
flipper_length = st.number_input("날개 길이 (mm)", min_value=150.0, max_value=250.0, value=200.0)
body_mass = st.number_input("체중 (g)", min_value=2500, max_value=6500, value=4000)

# 예측 버튼
if st.button("예측하기"):
    input_data = [[bill_length, bill_depth, flipper_length, body_mass]]
    prediction = model.predict(input_data)
    st.success(f"예측된 펭귄 종류: {prediction[0]}")

# 앱 실행 안내
st.write("위 입력값을 바탕으로 펭귄의 종류를 예측합니다.")
