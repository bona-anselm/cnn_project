import streamlit as st
import streamlit_authenticator as stauth
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# ---- Step 1: Define user credentials ----
names = ['Alice', 'Bob', 'Bona']
usernames = ['anayo', 'bejoy', 'bona']
passwords = ['anayo123', 'bejoy123', 'bona123']

# ---- Step 2: Hash passwords ----
hashed_passwords = stauth.Hasher(passwords).generate()

# ---- Step 3: Prepare credentials dictionary ----
credentials = {
    "usernames": {
        usernames[0]: {"name": names[0], "password": hashed_passwords[0]},
        usernames[1]: {"name": names[1], "password": hashed_passwords[1]},
        usernames[2]: {"name": names[2], "password": hashed_passwords[2]},
    }
}

# ---- Step 4: Create the authenticator ----
authenticator = stauth.Authenticate(
    credentials,
    cookie_name="fraud_detection_cookie",
    key="fraud_detection_key",
    cookie_expiry_days=30
)

# ---- Step 5: Authenticate user ----
name, auth_status, username = authenticator.login("Login", "main")

# ---- Step 6: Check login status ----
if auth_status is False:
    st.error("Username/password is incorrect.")
elif auth_status is None:
    st.warning("Please enter your username and password.")
elif auth_status:
    # Show logout in the sidebar
    authenticator.logout("Logout", "sidebar")

    st.success(f"Welcome, {name}!")
    st.write("You're now authenticated and can access the fraud detection dashboard.")


    # Load the trained model
    model = tf.keras.models.load_model("cnn_fraud_model.h5")

    # Load the scaler used for preprocessing
    time_scaler = joblib.load('time_scaler.pkl')
    amount_scaler = joblib.load('amount_scaler.pkl')


    # Set the title of the app
    st.title("Credit Card Fraud Detection App")
    st.write("Enter transaction features (V1 to V28, Time, and Amount) to predict fraud on the Sidebar.")
    st.write("You can also upload a CSV file with the same features for bulk fraud detection.")


    # Sidebar inputs for all 30 features
    with st.sidebar:
        st.header("Input Transaction Features")
        time = st.number_input("Time", value=100000.0)
        amount = st.number_input("Amount", value=50.0)
        features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]


    st.subheader("Upload CSV for Bulk Fraud Detection")

    uploaded_file = st.file_uploader("Upload CSV file with 30 features", type=['csv'])
    

    def plot_roc(y_true, y_probs):
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        return fig

    # If a file is uploaded, read and process it
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)

        st.write("Uploaded Data Preview:")
        st.dataframe(df_uploaded.head())

        # Make sure these columns exist before transforming
        if 'Time' in df_uploaded.columns and 'Amount' in df_uploaded.columns:
            df_uploaded['scaled_time'] = time_scaler.transform(df_uploaded[['Time']])
            df_uploaded['scaled_amount'] = amount_scaler.transform(df_uploaded[['Amount']])

            # Drop originals
            df_uploaded.drop(['Time', 'Amount'], axis=1, inplace=True)

            # Use only expected features
            input_data = df_uploaded[['scaled_time'] + [f'V{i}' for i in range(1, 29)] + ['scaled_amount']]

            # Reshape and predict
            input_data_reshaped = np.array(input_data).reshape(-1, 30, 1)
            predictions = model.predict(input_data_reshaped)
            df_uploaded['Fraud_Probability'] = predictions
            #df_uploaded['Prediction'] = (predictions > 0.5).astype(int)
            threshold = st.slider("Set classification threshold", 0.0, 1.0, 0.5)
            df_uploaded['Prediction'] = (predictions > threshold).astype(int)

            st.write("Prediction Results:")
            st.dataframe(df_uploaded[['Fraud_Probability', 'Prediction']].head(10))

            # Optional: Download
            csv_out = df_uploaded.to_csv(index=False)
            st.download_button("Download Results", csv_out, "fraud_predictions.csv", "text/csv")
        else:
            st.warning("Uploaded file must include 'Time' and 'Amount' columns.")

        # Histogram of Fraud Probabilities
        st.subheader("Histogram of Fraud Probabilities")
        fig, ax = plt.subplots()
        ax.hist(df_uploaded['Fraud_Probability'], bins=20)
        st.pyplot(fig)

        # ROC Curve for uploaded file (only if true labels available)
        # if 'Class' in df_uploaded.columns:
        #     st.subheader("ROC Curve")
        #     fig = plot_roc(df_uploaded['Class'], df_uploaded['Fraud_Probability'])
        #     st.pyplot(fig)
        # else:
        #     st.info("ROC Curve requires a 'Class' column in your uploaded data for true labels.")
        if 'Class' in df_uploaded.columns:
            if df_uploaded['Class'].nunique() < 2:
                st.warning("ROC Curve requires both classes (0 and 1) to be present in your data.")
            else:
                st.subheader("ROC Curve")
                fig = plot_roc(df_uploaded['Class'], df_uploaded['Fraud_Probability'])
                st.pyplot(fig)


    # Prepossessing function
    # def preprocess_input(v_features, time, amount):
    #     scaler = StandardScaler()
    #     #Simulate standardization (for demo only; ideally fit on training data)
    #     scaled_time = scaler.fit_transform(np.array(time).reshape(-1, 1))[0][0]
    #     scaled_amount = scaler.fit_transform(np.array(amount).reshape(-1, 1))[0][0]

    def preprocess_input(v_features, time, amount):
        scaled_time = time_scaler.transform(np.array([[time]]))[0][0]
        scaled_amount = amount_scaler.transform(np.array([[amount]]))[0][0]


        # Final input shape: (30, 1)
        all_features = [scaled_time] + v_features + [scaled_amount]
        input_array = np.array(all_features).reshape(1, 30, 1)
        return input_array


    # Predict when the button is clicked
    if st.button("Check Fraud"):
        input_data = preprocess_input(features, time, amount)
        prediction = model.predict(input_data)[0][0]
        
        if prediction > 0.5:
            st.error(f"Fraud Detected! (Probability: {prediction:.2%})")
        else:
            st.success(f"Transaction is Normal (Probability: {1 - prediction:.2%})")
   






