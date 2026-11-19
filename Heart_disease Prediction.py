import streamlit as st
import joblib
import pandas as  pd

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("Scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title(" Heart Disease prediction ")
st.markdown("Fillup the Information :")

Age = st.slider("Age",18,100,40)
sex = st.selectbox("Sex",['Male','Female'])
ChestPain = st.selectbox("ChestPain Type",['typical','non-typical','asymptomatic','nonanginal'])
RestingBP =  st.number_input("Resting Blood Pressure(mm  Hg)",80,200,120)
chol = st.number_input("Cholesterol (mm/dl",100,600,200)
fasting_bs =st.selectbox("Fasting blood sugar > 120 mg/dl",[0,1])
rest_ECG = st.selectbox("Resting ECG 0:'normal',1:'having st-t wave abonormality',2:'showing probable left ventricular hypertrophy'",[0,1,2])
maxHR = st.slider("Maximum Heart Rate",60,200,150)
Exang = st.selectbox("Exercise Angina",[0,1])
old_peak = st.slider("Oldpeak (ST depression)",0.0,6.0,1.0)
slope = st.selectbox("ST Slope",[1,2,3])
ca = st.selectbox("Ca 0:'0_vessele',1:'1_vessele',2:'2_vessele',3:'3_vessele' ",[0,1,2,3])
thal = st.selectbox("Thalessemia ",['Normal','fixed','reversable'])


if st.button("Predict"):
   
    raw_input = {
        'Age': Age,
        'Sex': 1 if sex == 'Male' else 0,
        'RestBP': RestingBP,
        'Chol': chol,
        'Fbs': fasting_bs,
        'RestECG': rest_ECG,
        'MaxHR': maxHR,
        'ExAng': Exang,
        'Oldpeak': old_peak,
        'Slope': slope,
        'Ca': ca,
        'ChestPain_asymptomatic': 1 if ChestPain == 'asymptomatic' else 0,
        'ChestPain_nonanginal': 1 if ChestPain == 'nonanginal' else 0,
        'ChestPain_nontypical': 1 if ChestPain == 'non-typical' else 0,
        'ChestPain_typical': 1 if ChestPain == 'typical' else 0,
        'Thal_fixed': 1 if thal == 'fixed' else 0,
        'Thal_normal': 1 if thal == 'Normal' else 0,
        'Thal_reversable': 1 if thal == 'reversable' else 0,
    }

    input_df = pd.DataFrame([raw_input])
    for columns in expected_columns:
        if  columns not in input_df.columns:
            input_df[columns] = 0
    input_df = input_df[expected_columns]
    scale_input = scaler.transform(input_df)
    prediction = model.predict(scale_input)[0]

    if prediction == 1 :
        st.error("High Risk Of Heart Disease ")
    else:
        st.success("Low Risk Of Heart Disease")

    st.write("Check input:", input_df)
