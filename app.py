import streamlit as st
import numpy as np
import pandas as pd
import joblib

## Load trained model
model = joblib.load('model_lr_hdb.pkl')

## Streamlit app
st.title("HDB Resale Price Prediction")

## Define the input options
towns = ['Tampines', 'Bedok', 'Punggol']
flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM']
storey_ranges = ['04 TO 06', '07 TO 09', '01 TO 03']

# towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
#          'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
#          'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
#          'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
#          'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
#          'TOA PAYOH', 'WOODLANDS', 'YISHUN']

# flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '1 ROOM',
#               'MULTI-GENERATION']

# storey_ranges = ['10 TO 12', '01 TO 03', '04 TO 06', '07 TO 09', '13 TO 15',
#                  '19 TO 21', '22 TO 24', '16 TO 18', '34 TO 36', '28 TO 30',
#                  '37 TO 39', '49 TO 51', '25 TO 27', '40 TO 42', '31 TO 33',
#                  '46 TO 48', '43 TO 45']

## User inputs
town_selected = st.selectbox("Select Town", towns)
flat_type_selected = st.selectbox("Select Flat Type", flat_types)
storey_range_selected = st.selectbox("Select Storey", storey_ranges)
floor_area_selected = st.slider("Select Floor Area (sqm)", min_value=30, max_value=200, value=70)

## Predict button
if st.button("Predict HDB price"):

    ## Create dict for input features
    input_data = {
        'town': town_selected,
        'flat_type': flat_type_selected,
        'storey_range': storey_range_selected,
        'floor_area': floor_area_selected
    }

    ## Convert input data to a DataFrame
    df_input = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range_selected],
        'floor_area': [floor_area_selected]
    })

    ## One-hot encoding
    df_input = pd.get_dummies(df_input, 
                              columns = ['town', 'flat_type', 'storey_range']
                             )
    
    # df_input = df_input.to_numpy()

    df_input = df_input.reindex(columns = model.feature_names_in_,
                                fill_value=0)



    ## Predict
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Resale Price: ${y_unseen_pred:,.2f}")

## Page design
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://www.shutterstock.com/shutterstock/videos/1025418011/thumb/1.jpg");
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)