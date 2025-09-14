import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

# Load the trained pipeline
try:
    lr_pipeline = joblib.load('lr_pipeline.pkl')
except FileNotFoundError:
    st.error("Error: 'lr_pipeline.pkl' not found. Please make sure the pipeline is saved.")
    st.stop()

# Load the original data to get column names and the LabelEncoder
try:
    df = pd.read_csv('processed_crop_data22.csv')
    # Fit LabelEncoder to all unique labels in the original data
    le = LabelEncoder()
    le.fit(df['label'])
except FileNotFoundError:
    st.error("Error: 'processed_crop_data22.csv' not found. Please make sure the data file is in the same directory.")
    st.stop()

# Define fertilizer recommendations (same dictionary as in the notebook)
fertilizer_recommendations = {
    "N": {"description": "Nitrogen is essential for plant growth, promoting leafy development and overall vigor.","recommended_fertilizers": ["Urea (46% N)","Ammonium Nitrate (33% N)","Ammonium Sulfate (21% N + Sulfur)","Calcium Ammonium Nitrate (CAN)"]},
    "P": {"description": "Phosphorus promotes root development, flower and fruit production, and early plant growth.","recommended_fertilizers": ["Single Super Phosphate (SSP)","Triple Super Phosphate (TSP)","Monoammonium Phosphate (MAP)","Diammonium Phosphate (DAP)"]},
    "K": {"description": "Potassium improves overall plant health, drought resistance, and disease resistance.","recommended_fertilizers": ["Muriate of Potash (MOP / Potassium Chloride)","Sulfate of Potash (SOP / Potassium Sulfate)","Potassium Nitrate (KNO3)"]},
    "pH Balancer": {"description": "Used to adjust soil pH to optimal range (typically 6.0–7.5) for better nutrient availability.","recommended_fertilizers": ["Lime (Calcium Carbonate) – to raise pH (reduce acidity)","Dolomitic Lime – for calcium and magnesium supply and pH increase", "Elemental Sulfur – to lower pH (increase acidity)","Aluminum Sulfate – fast-acting acidifier"]}
}

st.title("Crop Recommendation System")

st.write("Enter the environmental parameters to get crop recommendations:")

# Input fields in a container
with st.container(border=True): # Use st.container() with border=True
    st.subheader("Input Parameters")
    N = st.slider("Nitrogen (N)", float(df['N'].min()), float(df['N'].max()), float(df['N'].mean()), key='N_slider')
    P = st.slider("Phosphorus (P)", float(df['P'].min()), float(df['P'].max()), float(df['P'].mean()), key='P_slider')
    K = st.slider("Potassium (K)", float(df['K'].min()), float(df['K'].max()), float(df['K'].mean()), key='K_slider')
    temperature = st.slider("Temperature (°C)", float(df['temperature'].min()), float(df['temperature'].max()), float(df['temperature'].mean()), key='temp_slider')
    humidity = st.slider("Humidity (%)", float(df['humidity'].min()), float(df['humidity'].max()), float(df['humidity'].mean()), key='humidity_slider')
    ph = st.slider("pH", float(df['ph'].min()), float(df['ph'].max()), float(df['ph'].mean()), key='ph_slider')
    rainfall = st.slider("Rainfall (mm)", float(df['rainfall'].min()), float(df['rainfall'].max()), float(df['rainfall'].mean()), key='rainfall_slider')
    price = st.slider("Price", float(df['price'].min()), float(df['price'].max()), float(df['price'].mean()), key='price_slider')


if st.button("Get Recommendations"):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall, price]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'price'])

    # Get the predicted probabilities using the pipeline
    probabilities = lr_pipeline.predict_proba(input_data)[0]

    # Get the class labels (decoded from the encoded labels)
    class_labels_decoded = le.inverse_transform(lr_pipeline.classes_)

    # Create a pandas Series of probabilities with decoded class labels as index
    probability_series = pd.Series(probabilities, index=class_labels_decoded)

    # Sort the probabilities in descending order and get the top recommendations
    sorted_probabilities = probability_series.sort_values(ascending=False)
    top_3_recommendations = sorted_probabilities.head(3)

    # Store the recommendations in session state
    st.session_state['top_3_recommendations'] = top_3_recommendations
    st.session_state['input_data'] = input_data


# Display recommendations if they are in session state
if 'top_3_recommendations' in st.session_state:
    top_3_recommendations = st.session_state['top_3_recommendations']
    input_data = st.session_state['input_data']

    with st.container(border=True): # Use st.container() with border=True
        st.subheader("Top 3 Crop Recommendations")
        for crop in top_3_recommendations.index:
            st.write(f"- **{crop}**")


    # Get the second and third recommended crop names
    if len(top_3_recommendations) > 1:
        second_recommended_crop = top_3_recommendations.index[1]
        with st.expander(f"Suggestions for **{second_recommended_crop}**"):
            with st.container(border=True): # Use st.container() with border=True for expander content
                st.subheader(f"Suggestions for {second_recommended_crop}")
                # Filter data for the second recommended crop
                df_second_crop = df[df['label'] == second_recommended_crop].copy()
                average_params_second_crop = df_second_crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'price']].mean()
                insufficient_parameters_second = {}
                for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'price']:
                     if input_data[col].iloc[0] < average_params_second_crop[col]:
                         insufficient_parameters_second[col] = input_data[col].iloc[0]

                if not insufficient_parameters_second:
                     st.write("No insufficient parameters identified for this crop.")
                else:
                    st.write("**Insufficient Parameters Compared to Average:**")
                    for param in insufficient_parameters_second.keys(): # Only show parameter name
                        st.write(f"- **{param}**:")
                        if param == 'ph':
                            st.write("  Consider adjusting soil pH.")
                            if 'pH Balancer' in fertilizer_recommendations:
                                 st.write("  **Recommended pH Balancers:**")
                                 for fertilizer in fertilizer_recommendations['pH Balancer']['recommended_fertilizers']:
                                     st.write(f"    - {fertilizer}")
                        elif param in fertilizer_recommendations:
                            st.write(f"  {fertilizer_recommendations[param]['description']}")
                            st.write("**Recommended Fertilizers:**")
                            for fertilizer in fertilizer_recommendations[param]['recommended_fertilizers']:
                                 st.write(f"    - {fertilizer}")
                        else:
                             st.write(f"  No specific recommendations available for {param}.")


    if len(top_3_recommendations) > 2:
        third_recommended_crop = top_3_recommendations.index[2]
        with st.expander(f"Suggestions for **{third_recommended_crop}**"):
            with st.container(border=True): # Use st.container() with border=True for expander content
                st.subheader(f"Suggestions for {third_recommended_crop}")
                # Filter data for the third recommended crop
                df_third_crop = df[df['label'] == third_recommended_crop].copy()
                average_params_third_crop = df_third_crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'price']].mean()
                insufficient_parameters_third = {}
                for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'price']:
                     if input_data[col].iloc[0] < average_params_third_crop[col]:
                         insufficient_parameters_third[col] = input_data[col].iloc[0]

                if not insufficient_parameters_third:
                     st.write("No insufficient parameters identified for this crop.")
                else:
                    st.write("**Insufficient Parameters Compared to Average:**")
                    for param in insufficient_parameters_third.keys(): # Only show parameter name
                        st.write(f"- **{param}**:")
                        if param == 'ph':
                            st.write("  Consider adjusting soil pH.")
                            if 'pH Balancer' in fertilizer_recommendations:
                                 st.write("**Recommended pH Balancers:**")
                                 for fertilizer in fertilizer_recommendations['pH Balancer']['recommended_fertilizers']:
                                     st.write(f"    - {fertilizer}")
                        elif param in fertilizer_recommendations:
                            st.write(f"  {fertilizer_recommendations[param]['description']}")
                            st.write("**Recommended Fertilizers:**")
                            for fertilizer in fertilizer_recommendations[param]['recommended_fertilizers']:
                                 st.write(f"    - {fertilizer}")
                        else:
                             st.write(f"  No specific recommendations available for {param}.")