import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)


# --------------------------------------------------
# LOAD MODEL AND CATEGORIES
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")


@st.cache_resource
def load_categories():
    return joblib.load("car_categories.pkl")


model = load_model()
categories = load_categories()

# Load Make -> Model mapping
make_model_mapping = joblib.load("make_model_mapping.pkl")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🚗 Car Price Predictor")

st.write(
    "Enter the specifications of the car to estimate "
    "its Manufacturer's Suggested Retail Price (MSRP)."
)

st.divider()


# --------------------------------------------------
# CAR INFORMATION
# --------------------------------------------------

st.subheader("🚘 Car Information")

col1, col2 = st.columns(2)

with col1:
    make = st.selectbox(
        "Make",
        sorted(make_model_mapping.keys()),
        key="make_selectbox"
    )

with col2:
    year = st.number_input(
        "Year",
        min_value=1990,
        max_value=2017,
        value=2015,
        step=1
    )

# Show only models belonging to the selected make
available_models = sorted(make_model_mapping[make])

model_name = st.selectbox(
    "Model",
    available_models,
    key="model_selectbox"
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

available_models = sorted(make_model_mapping[make])

# --------------------------------------------------
# ENGINE INFORMATION
# --------------------------------------------------

st.subheader("⚙️ Engine Information")

col1, col2 = st.columns(2)

with col1:
    engine_fuel_type = st.selectbox(
        "Engine Fuel Type",
        categories["Engine Fuel Type"]
    )

with col2:
    engine_hp = st.number_input(
        "Engine HP",
        min_value=55.0,
        max_value=1001.0,
        value=200.0,
        step=1.0
    )


col1, col2 = st.columns(2)

with col1:
    engine_cylinders = st.number_input(
        "Engine Cylinders",
        min_value=0.0,
        max_value=16.0,
        value=4.0,
        step=1.0
    )

with col2:
    transmission_type = st.selectbox(
        "Transmission Type",
        categories["Transmission Type"]
    )


# --------------------------------------------------
# VEHICLE INFORMATION
# --------------------------------------------------

st.subheader("🚙 Vehicle Information")

col1, col2 = st.columns(2)

with col1:
    driven_wheels = st.selectbox(
        "Driven Wheels",
        categories["Driven_Wheels"]
    )

with col2:
    number_of_doors = st.selectbox(
        "Number of Doors",
        [2.0, 3.0, 4.0]
    )


col1, col2 = st.columns(2)

with col1:
    vehicle_size = st.selectbox(
        "Vehicle Size",
        categories["Vehicle Size"]
    )

with col2:
    vehicle_style = st.selectbox(
        "Vehicle Style",
        categories["Vehicle Style"]
    )


# --------------------------------------------------
# FUEL ECONOMY
# --------------------------------------------------

st.subheader("⛽ Fuel Economy")

col1, col2 = st.columns(2)

with col1:
    highway_mpg = st.number_input(
        "Highway MPG",
        min_value=12,
        max_value=354,
        value=26,
        step=1
    )

with col2:
    city_mpg = st.number_input(
        "City MPG",
        min_value=7,
        max_value=137,
        value=19,
        step=1
    )


# --------------------------------------------------
# POPULARITY
# --------------------------------------------------

st.subheader("📊 Other Information")

popularity = st.number_input(
    "Popularity",
    min_value=2,
    max_value=5657,
    value=1000,
    step=1
)


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔮 Predict Car Price",
    use_container_width=True
):

    # Create dataframe with EXACT same columns
    # used during model training

    input_data = pd.DataFrame([{
        "Make": make,
        "Model": model_name,
        "Year": year,
        "Engine Fuel Type": engine_fuel_type,
        "Engine HP": engine_hp,
        "Engine Cylinders": engine_cylinders,
        "Transmission Type": transmission_type,
        "Driven_Wheels": driven_wheels,
        "Number of Doors": number_of_doors,
        "Vehicle Size": vehicle_size,
        "Vehicle Style": vehicle_style,
        "highway MPG": highway_mpg,
        "city mpg": city_mpg,
        "Popularity": popularity
    }])


    # Make prediction

    prediction = model.predict(input_data)


    # Display result

    st.success("Prediction completed!")

    st.metric(
        label="Estimated MSRP",
        value=f"${prediction[0]:,.2f}"
    )

    st.info(
        "The displayed value is an estimated MSRP "
        "generated by the machine learning model."
    )