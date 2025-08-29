import datetime
import os
import pandas as pd
import streamlit as st
import joblib
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.lr import load_model


# ======================= Encoder Loading Step =================

def load_encoder(encoder_path):
    return joblib.load(encoder_path)

def load_encoders():
    item_identifier_encoder = load_encoder("artifacts/encoders/item_identifier_encoder.pkl")
    item_type_encoder = load_encoder("artifacts/encoders/item_type_encoder.pkl")
    return item_identifier_encoder, item_type_encoder


model = load_model("artifacts", "linear_regression")


item_identifier_encoder, item_type_encoder = load_encoders()

# ======================= Encoder Loading Step Ended =================

# ======================= Streamlit UI Code =================
st.title('Big Mart Sales Prediction')

st.write("ML Streamlit Application")

item_weight = st.number_input("Item Weight")
item_visibility =  st.number_input("Item Visibility")
item_mrp =  st.number_input("Item MRP")
established_date = st.date_input("Established Year", datetime.date(2019, 7, 6),)
# Extract year
established_year = established_date.year
item_options = [
     'FDW13',
    'FDG33',
    'NCY18',
    'FDD38',
    'DRE49'
]

# Streamlit selectbox
item_identifier = st.selectbox("Select Item Identifier", item_options)


item_types = [
    "Fruits and Vegetables",
    "Snack Foods",
    "Household",
    "Frozen Foods",
    "Dairy",
    "Canned",
    "Baking Goods",
    "Health and Hygiene",
    "Soft Drinks",
    "Meat",
    "Breads",
    "Hard Drinks",
    "Others",
    "Starchy Foods",
    "Breakfast",
    "Seafood"
]
item_type = st.selectbox("Select Item Type", item_types)

fat_contents = [
    "LF",
    "REG",
]
Item_Fat_Content = st.selectbox("Select Item Fat Content", fat_contents)


output_locations= [
    "Tier 1",
    "Tier 2",
    "Tier 3",
    
]
outlet_size= [
    "Small",
    "Medium",
    "High",
    
]
outlet_type= [
    "Grocery Store",
    "Supermarket Type1",
    "Supermarket Type2",
    "Supermarket Type3"
]


Output_Location_Type = st.selectbox("Select  Output Location Type", output_locations)

Outlet_Size= st.selectbox("Select  Outlet Size", outlet_size)

Outlet_Type= st.selectbox("Select  Outlet Type", outlet_type)

# ======================= Streamlit UI Code Ended =================

# ======================== Feature Encoding =============
# Encode

# Item Identifier and Item Type Target Encoding
identifier_df = pd.DataFrame({"Item_Identifier": [item_identifier]})
type_df = pd.DataFrame({"Item_Type": [item_type]})


Item_Identifier_encoded = int(item_identifier_encoder.transform(identifier_df).iloc[0, 0])

Item_Type_encoded = int(item_type_encoder.transform(type_df).iloc[0, 0])

def build_feature_row(
    item_weight, item_visibility, item_mrp, established_year,
    item_identifier_encoded, item_type_encoded,
    item_fat_content, output_location, outlet_size, outlet_type
):
    
    # Start with base columns
    row = {
        'Item_Weight': item_weight,
        'Item_Visibility': item_visibility,
        'Item_MRP': item_mrp,
        'Outlet_Establishment_Year': established_year,
        'Item_Identifier_encoded': item_identifier_encoded,
        'Item_Type_encoded': item_type_encoded,
    }

    # One-hot encoding manually
    fat_content_options = ["LF", "REG"]
    for fc in fat_content_options:
        row[f'Item_Fat_Content_encoded_{fc}'] = 1 if item_fat_content == fc else 0

    outlet_location_options = ["Tier 1", "Tier 2", "Tier 3"]
    for loc in outlet_location_options:
        row[f'Outlet_Location_Type_{loc}'] = 1 if output_location == loc else 0

    outlet_size_options = ["High", "Medium", "Small"]
    for size in outlet_size_options:
        row[f'Outlet_Size_{size}'] = 1 if outlet_size == size else 0

    outlet_type_options = [
        "Grocery Store",
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3"
    ]
    for typ in outlet_type_options:
        row[f'Outlet_Type_{typ}'] = 1 if outlet_type == typ else 0
    df = pd.DataFrame([row])
    return df

processed_row = build_feature_row(
    item_weight=item_weight,
    item_visibility=item_visibility,
    item_mrp=item_mrp,
    established_year=established_year,
    item_identifier_encoded=Item_Identifier_encoded,
    item_type_encoded=Item_Type_encoded,
    item_fat_content=Item_Fat_Content,
    output_location=Output_Location_Type,
    outlet_size=Outlet_Size,
    outlet_type=Outlet_Type
)

st.write("Processed Input Row:")
st.dataframe(processed_row)

call_model = st.button("Predict")

if call_model:
    prediction = model.predict(processed_row)
    st.markdown(
        f"""
        <div style="background-color:#4CAF50;padding:20px;border-radius:10px">
            <h2 style="color:white;text-align:center;">
                Predicted Sales: {prediction[0]:,.2f}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
# ...existing code...