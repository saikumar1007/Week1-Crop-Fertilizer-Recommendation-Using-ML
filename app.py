import streamlit as st
import numpy as np
import pickle

# Load your trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Crop label map (adjust based on your model if necessary)
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Recommendation function
def crop_recommend(N, P, K, temperature, humidity, ph, rainfall):
    # Input features
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale the features (important step!)
    transformed = scaler.transform(features)
    
    # Predict crop
    prediction = model.predict(transformed)[0]  
    crop = crop_dict[int(prediction)]
    
    return f"{crop} is the best crop to be cultivated"


# Streamlit UI
st.set_page_config(page_title="Crop Recommendation", page_icon="🌱")
st.title("🌾 Crop Recommendation System")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=140, value=42)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)
temperature = st.number_input("Temperature (°C)", value=20.0)
humidity = st.number_input("Humidity (%)", value=82.0)
ph = st.number_input("Soil pH", value=6.1)
rainfall = st.number_input("Rainfall (mm)", value=202.0)

st.sidebar.header("ℹ️ About the Project")
st.sidebar.write("This Crop Recommendation System uses a **Decision Tree Classifier** trained on soil nutrients and weather data.")

# Predict button
if st.button("🚜 Recommend Crop"):
    result = crop_recommend(N, P, K, temperature, humidity, ph, rainfall)
    st.success(f"✅ {result}")

