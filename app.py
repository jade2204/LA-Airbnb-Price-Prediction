import streamlit as st
import pandas as pd
import joblib


#Load trained model and preprocessor(scaler, encoder)
artifacts= joblib.load("airbnb_model.pkl")

model= artifacts["model"]
preprocessor= artifacts["preprocessor"]

#Page configuration
st.set_page_config(
    page_title= "LA Airbnb Price Prediction",
    page_icon= "🏠",
    layout= "centered"
)

#Title
st.title("LA Airbnb Price Prediction")
st.write("Enter the Airbnb's information below to predict the price per night.")

#Input fields
st.subheader("Property Information")

host_response_time= st.selectbox("Host Response Time", ["within an hour", "within a few hours", "within a day", "a few days or more"])

host_response_rate= st.slider("Host Response Rate (%)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

host_is_superhost= st.selectbox("Host is Superhost", ["t", "f"])

neighbourhood_cleansed= st.text_input("Neighbourhood Cleansed")

neighbourhood_group_cleansed= st.selectbox("Neighbourhood Group Cleansed", ["City of Los Angeles", "Other Cities", "Unincorporated Areas"])

latitude= st.number_input("Latitude", value=34.0522, format="%.6f")

longitude= st.number_input("Longitude", value=-118.2437, format="%.6f")

property_type= st.text_input("Property Type")

room_type=st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])

accommodates= st.number_input("Accommodates", min_value=1, max_value=20, value=1, step=1)

bathrooms= st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5, format="%.1f")

bedrooms= st.number_input("Bedrooms", min_value=0, max_value=10, value=1, step=1)

beds= st.number_input("Beds", min_value=0, max_value=20, value=1, step=1)

minimum_nights= st.number_input("Minimum Nights", min_value=1, max_value=365, value=1, step=1)

availability_365= st.number_input("Availability (Days per Year)", min_value=0, max_value=365, value=365, step=1)

number_of_reviews= st.number_input("Number of Reviews", min_value=0, max_value=1000, value=0, step=1)

review_scores_rating= st.slider("Review Scores Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.01)

instant_bookable= st.selectbox("Instant Bookable", ["t", "f"])

#Prediction
if st.button("Predict Price", type="primary"):

    input_data = pd.DataFrame({
        "host_response_time": [host_response_time],
        "host_response_rate": [host_response_rate],
        "host_is_superhost": [host_is_superhost],
        "neighbourhood_cleansed": [neighbourhood_cleansed],
        "neighbourhood_group_cleansed": [
            neighbourhood_group_cleansed
        ],
        "latitude": [latitude],
        "longitude": [longitude],
        "property_type": [property_type],
        "room_type": [room_type],
        "accommodates": [accommodates],
        "bathrooms": [bathrooms],
        "bedrooms": [bedrooms],
        "beds": [beds],
        "minimum_nights": [minimum_nights],
        "availability_365": [availability_365],
        "number_of_reviews": [number_of_reviews],
        "review_scores_rating": [review_scores_rating],
        "instant_bookable": [instant_bookable]
    })

    try:

        # Apply the same preprocessing used during training
        processed_data = preprocessor.transform(input_data)

        # Predict Airbnb price
        prediction = model.predict(processed_data)

        predicted_price = prediction[0]


        #Display prediction
        st.metric(
            label="Estimated Price per Night",
            value=f"${predicted_price:,.2f}"
        )

        st.info(
            f"The estimated Airbnb price is "
            f"${predicted_price:,.2f} per night."
        )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)
