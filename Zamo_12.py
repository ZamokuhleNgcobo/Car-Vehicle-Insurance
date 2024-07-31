import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

@st.cache_data
def load_data():
    data = pd.read_csv('/Users/da-m1-18/Downloads/car_insurance_claim.csv')
    return data

@st.cache_data
def preprocess_data(data):
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data['Vehicle_Age'] = label_encoder.fit_transform(data['Vehicle_Age'])
    data['Vehicle_Damage'] = label_encoder.fit_transform(data['Vehicle_Damage'])
    selected_features = ['Gender', 'Age', 'Annual_Premium', 'Vehicle_Damage', 'Vehicle_Age', 'Previously_Insured']
    X = data[selected_features]
    return X

def user_input_features():
    with st.form(key='user_input_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            gender = 1 if gender == 'Male' else 0
            previously_insured = st.selectbox('Previously Insured', ['Yes', 'No'])
            previously_insured = 1 if previously_insured == 'Yes' else 0

        with col2:
            age = st.slider('Age', 20, 85, 30)
            annual_premium = st.number_input('Annual Premium', min_value=2630, max_value=508073, value=30000, step=1, format="%d")

        with col3:
            vehicle_damage = st.selectbox('Vehicle Damage', ['Yes', 'No'])
            vehicle_damage = 1 if vehicle_damage == 'Yes' else 0
            vehicle_age = st.selectbox('Vehicle Age', ['< 1 Year', '1-2 Year', '> 2 Years'])
            vehicle_age = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}[vehicle_age]

        submit_button = st.form_submit_button(label='Predict')

        data = {
            'Gender': gender,
            'Age': age,
            'Annual_Premium': annual_premium,
            'Vehicle_Damage': vehicle_damage,
            'Vehicle_Age': vehicle_age,
            'Previously_Insured': previously_insured
        }

    return data, submit_button

def main():
    st.set_page_config(page_title="Vehicle Insurance Prediction", page_icon="🛡️")

    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://www.carscoops.com/wp-content/uploads/2020/05/Facelift-BMW-5-Series-1024x555.jpg');
            background-size: cover;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.image('https://cdn.prod.website-files.com/5de2db6d3719a1e2f3e4454c/618fbbfbfb26294d9d56482c_Cover_image.jpg', width=150)
    st.markdown("<h1 style='text-align: center; color: white;'>Vehicle Insurance Prediction</h1>", unsafe_allow_html=True)
    st.write("<h3 style='text-align: center; color: white;'>This App Predicts Whether A Customer will Be Interested In vehicle Insurance Based On Their Information.</h3>", unsafe_allow_html=True)

    data = load_data()
    X = preprocess_data(data)

    user_data, predict_button = user_input_features()

    if predict_button:
        input_df = pd.DataFrame(user_data, index=[0])

        scaler = StandardScaler()
        scaler.fit(X)
        input_scaled = scaler.transform(input_df)

        best_model = tf.keras.models.load_model('/Users/da-m1-18/best_model_final.keras')
        prediction = best_model.predict(input_scaled)

        st.subheader('Prediction:')
        insurance_needed = 'Interested' if prediction[0][0] > 0.5 else 'Not Interested'
        st.write(f"Customer Response:  {insurance_needed}")

if __name__ == '__main__':
    main()