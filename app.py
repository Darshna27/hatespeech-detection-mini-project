import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("xgboost_multi_label_model.pkl")  # Load your XGBoost model
vectorizer = joblib.load("vectorizer.pkl")  # Load the vectorizer

# Title of the app
st.title("Hate Speech and Emotion Detection")

# Sidebar Customization
st.sidebar.image("https://sundayguardianlive.com/wp-content/uploads/2018/02/laws%20for%20hate%20speech.jpg", use_container_width=True)  # Add your image URL here
st.sidebar.header("Model Information")

# Update the description to provide more details about the app
st.sidebar.write(
    """
    This app detects hate speech and various emotions from user comments. It uses machine learning models 
    to identify whether a given comment falls under categories such as **Toxic**, **Severe Toxic**, **Obscene**, 
    **Threat**, **Insult**, or **Identity Hate**. The app predicts with a probability for each label and classifies 
    the comment based on custom thresholds set for each category.

    **Features:**
    - Real-time detection of hate speech and offensive language.
    - Categorization of emotions such as **Happy**, **Angry**, **Sad**, etc. (if available).
    - Adjustable thresholds to fine-tune detection accuracy.
    
    Simply enter a comment in the box below and click **Predict** to see the result.
    """
)

# Text input for user
comment = st.text_area("Enter the comment to classify:")

# Define the labels
labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]

# Custom thresholds for prediction confidence
thresholds = {
    "Toxic": 0.3,
    "Severe Toxic": 0.05,
    "Obscene": 0.15,
    "Threat": 0.1,
    "Insult": 0.3,
    "Identity Hate": 0.05
}

# Predict button logic
if st.button("Predict"):
    if comment:  # Check if user input exists
        # Vectorize the input comment
        vectorized_input = vectorizer.transform([comment])

        # Predict probabilities for each label
        prob = model.predict_proba(vectorized_input)  # This returns probabilities
        
        # Verify the shape of the probability output
        st.write("Shape of prob:", np.shape(prob))
        
        # Display Results
        st.subheader("Prediction Results:")
        for i, label in enumerate(labels):
            # Extract positive class probability
            positive_class_prob = prob[i][0][1]  # Model output per label
            
            # Compare against thresholds
            prediction = 'Yes' if positive_class_prob > thresholds[label] else 'No'
            st.write(f"{label}: {prediction} (Probability: {positive_class_prob:.4f})")
        
        st.success("Prediction complete!")
    else:
        st.error("Please enter a comment to classify.")

# Footer
st.markdown(
    """
    
    <div style="text-align:center; color:grey;">
        <p>&copy; 2024 | Hate Speech and Emotion Detection App</p>
    </div>
    """,
    
    unsafe_allow_html=True
)
