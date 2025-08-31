import pandas as pd  #reading/manipulating dataframe
import numpy as np #Provides numeric arrays, math functions
import pickle #serialize/deserialize Python objects. You later use it to pickle.load() trained models, encoders, and scalers
import streamlit as st #UI, forms, layout, and rendering

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Industrial Copper Modeling", layout="wide") #sets the page configuration, browser tab title and a wide page layout

# ----------------- Page Header -------------------
st.markdown(
    """
    <div style='text-align:center'>
        <h1 style='color:#007777;'>Industrial Copper Modeling Application</h1>
    </div>
    """, unsafe_allow_html=True
)

# ----------------- Custom CSS --------------------
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #007777;
        color: white;
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #009999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Helper Function ----------------


#float(v) → converts each item to a float.

def validate_inputs(values): #values is expected to be an iterable (list/tuple) of strings or numbers
    try:
        return all(float(v) >= 0 for v in values) #checks every converted value is greater than or equal to 0; if any is negative, the whole expression returns
    except ValueError:
        return False

# ----------------- Dropdown Options ----------------
#Regression: t.pkl Classification: ct.pkl

status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
                  'Wonderful', 'Revised', 'Offered', 'Offerable'] #list of string labels for the Status selectbox. Comment notes these strings must match what the encoder s.pkl expects
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'] #Regression: t.pkl Classification: ct.pkl
country_options = sorted([28., 25., 30., 32., 38., 78., 27., 77., 113., 79.,
                          26., 39., 40., 84., 80., 107., 89.]) #sorted list of numeric country codes (floats). sorted(...) ensures the list appears in ascending order
application_options = sorted([10., 41., 28., 59., 15., 4., 38., 56., 42., 26.,
                              27., 19., 20., 66., 29., 22., 40., 25., 67., 79.,
                              3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]) #Sorted numeric application codes
product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405',
           '640665', '611993', '929423819', '1282007633', '1332077137',
           '164141591', '164336407', '164337175', '1665572032', '1665572374',
           '1665584320', '1665584642', '1665584662', '1668701376', '1668701698',
           '1668701718', '1668701725', '1670798778', '1671863738', '1671876026',
           '1690738206', '1690738219', '1693867550', '1693867563', '1721130331',
           '1722207579'] #list of product reference strings

# ----------------- Tabs ----------------
tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])

# =======================================================================
# TAB 1 - Predict Selling Price
# =======================================================================
with tab1:
    with st.form("predict_price_form"):
        col1, col2, col3 = st.columns([5, 1, 5])

        with col1:
            status = st.selectbox("Status", status_options, key=1)
            item_type = st.selectbox("Item Type", item_type_options, key=2)
            country = st.selectbox("Country", country_options, key=3)
            application = st.selectbox("Application", application_options, key=4)
            product_ref = st.selectbox("Product Reference", product, key=5)

        with col3:
            st.markdown("<h5 style='color:#00777780;'>NOTE: Min & Max given for reference, you can enter any value</h5>", unsafe_allow_html=True)
            quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            thickness = st.text_input("Enter Thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter Width (Min:1, Max:2990)")
            customer = st.text_input("Customer ID (Min:12458, Max:30408185)")

            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")

        # ----------------- Prediction ----------------
        if submit_button:
            if not validate_inputs([quantity_tons, thickness, width, customer]):
                st.error("Please enter valid numeric values. Spaces are not allowed.")
            else:
                with open(r"source/model.pkl", 'rb') as file: #trained regression model used to predict log(price)
                    loaded_model = pickle.load(file) #returns the object saved earlier
                with open(r'source/scaler.pkl', 'rb') as f: #used to scale features before prediction
                    scaler_loaded = pickle.load(f)
                with open(r"source/t.pkl", 'rb') as f: #transformer for item_type presumably an OneHotEncoder or similar
                    t_loaded = pickle.load(f)
                with open(r"source/s.pkl", 'rb') as f: #transformer for status maybe target encoder or one-hot
                    s_loaded = pickle.load(f) 

#np.log(float(quantity_tons)) — natural log of quantity. MUST be > 0, otherwise -inf or error
#application — already numeric (float) from the selectbox
#np.log(float(thickness)) — natural log of thickness. Must be > 0
#float(width) — width as float
#country — float country code from selectbox
#float(customer) — customer ID as float
#int(product_ref) — product ref converted from the selected string to integer
#item_type — string (e.g., 'W'), categorical column
#status — string status categorical column

#the row mixes floats and strings, np.array(...) will likely create an array with dtype object


                new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)),
                                        float(width), country, float(customer), int(product_ref),
                                        item_type, status]])
                

                #new_sample[:, [7]] ---- taking the eighth feature of a new data sample
                #t_loaded ---- applying a pre-trained one-hot encoder
                #toarray() ---- converts it to a dense NumPy array
                new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
                new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray() #pre-trained binary encoder & converts the resulting sparse matrix into a dense NumPy array

                #Rebuilds the full feature vector by concatenating:
                # new_sample[:, :7] → the first 7 numeric columns (indices 0..6)
                # new_sample_ohe → encoded item_type features
                # new_sample_be → encoded status features
                #stacks the arrays horizontally (column-wise)
                new_sample = np.concatenate((new_sample[:, :7], new_sample_ohe, new_sample_be), axis=1)

                #scaler_loaded --- pre-trained scikit-learn scaler, like a StandardScaler or a MinMaxScaler
                #.transform() ---- applies the scaling logic (e.g., mean and standard deviation for StandardScaler, or min and max for MinMaxScaler) that was learned from the training data to the new data
                #Produces the final input matrix for the ML model
                new_sample_scaled = scaler_loaded.transform(new_sample)

                #Calls the regression model’s
                #.predict(...) --- method and extracts the first element [0] from the predicted array. This implies the model predicts log(price)
                new_pred = loaded_model.predict(new_sample_scaled)[0]

                #Converts the predicted log-price back to price scale with np.exp(new_pred)
                #Formats the number with thousands separators and two decimals using Python’s formatted string: :,.2f
                st.success(f"Predicted Selling Price: {np.exp(new_pred):,.2f}")

# =======================================================================
# TAB 2 - Predict Status
# =======================================================================
with tab2:
    with st.form("predict_status_form"):
        col1, col2, col3 = st.columns([5, 1, 5])

        with col1:
            cquantity_tons = st.text_input("Enter Quantity Tons")
            cthickness = st.text_input("Enter Thickness")
            cwidth = st.text_input("Enter Width")
            ccustomer = st.text_input("Customer ID")
            cselling = st.text_input("Selling Price")

        with col3:
            citem_type = st.selectbox("Item Type", item_type_options, key=21)
            ccountry = st.selectbox("Country", country_options, key=31)
            capplication = st.selectbox("Application", application_options, key=41)
            cproduct_ref = st.selectbox("Product Reference", product, key=51)
            csubmit_button = st.form_submit_button(label="PREDICT STATUS")

        # ----------------- Prediction ----------------
        if csubmit_button:
            if not validate_inputs([cquantity_tons, cthickness, cwidth, ccustomer, cselling]): #Validates that each field converts to float and validate is >= 0
                st.error("Please enter valid numeric values. Spaces are not allowed.")
            else:
                with open(r"source/cmodel.pkl", 'rb') as file: #loads a machine learning model cmodel.pkl [rb - read binary, in binary format]
                    cloaded_model = pickle.load(file) #pickle.load() function deserializes the object from the binary file and reconstructs it in memory
                with open(r'source/cscaler.pkl', 'rb') as f: #loads a machine learning model cscaler.pkl [StandardScaler or MinMaxScaler from scikit-learn]
                    cscaler_loaded = pickle.load(f)
                with open(r"source/ct.pkl", 'rb') as f: #loads a machine learning model ct.pkl [ColumnTransformer from scikit-learn, used to apply different data transformations to different columns of a dataset]
                    ct_loaded = pickle.load(f)



                #Builds the raw-feature row for the classifier (shape (1, 9)):
                # np.log(float(cquantity_tons)) — log quantity (must be > 0)
                # np.log(float(cselling)) — log selling price (must be > 0)
                # capplication — numeric from selectbox
                # np.log(float(cthickness)) — log thickness (must be > 0)
                # float(cwidth) — width numeric
                # ccountry — numeric
                # int(ccustomer) — customer ID as int
                # int(cproduct_ref) — product reference as int (converted from string)
                # citem_type — categorical string

                new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)),
                                        capplication, np.log(float(cthickness)), float(cwidth),
                                        ccountry, int(ccustomer), int(cproduct_ref), citem_type]])
                
                #applies a pre-loaded column transformer to the ninth column of the new_sample array [categorical data] 
                # [one-hot encode - categorical labels into a binary vector]
                # .toarray() method converts the sparse matrix output of the transformer into a dense NumPy array

                new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()

                #original data is combined with the newly one-hot encoded features
                #new_sample[:, :8] selects the first eight columns (indices 0 to 7) of the original data, which are the non-categorical features
                # new_sample_ohe is the one-hot encoded representation of the categorical feature
                # np.concatenate(...) joins these two parts horizontally 
                # axis=1 to create a single array that includes both the original and the one-hot encoded features
                new_sample = np.concatenate((new_sample[:, :8], new_sample_ohe), axis=1)


                new_sample_scaled = cscaler_loaded.transform(new_sample) #pre-loaded data scaler (cscaler_loaded) to the combined data
                new_pred = cloaded_model.predict(new_sample_scaled) #pre-trained model (cloaded_model) is used to make a prediction on the fully pre-processed data

                if new_pred == 1:
                    st.success("The Status is: WON")
                else:
                    st.error("The Status is: LOST")
