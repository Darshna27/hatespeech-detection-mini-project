Hate Speech Detection Using XGBoost

Project Overview

This project aims to detect hate speech in textual data by using machine learning algorithms. The XGBoost algorithm is employed for training the model on a labeled dataset, sourced from Kaggle. The trained model is then integrated with a Streamlit-based web application to allow users to input text and determine whether it contains hate speech. The project showcases the end-to-end process of dataset extraction, preprocessing, model training, and deployment.

Features

Dataset Integration: The dataset is sourced from Kaggle using an API token for seamless access.

Text Classification: The project predicts whether a given input text contains hate speech.

Model Training: XGBoost is used for training, ensuring high accuracy and robust performance.

Interactive Interface: A user-friendly web interface built with Streamlit to analyze input text.

Workflow

1. Dataset Extraction

The dataset is fetched from Kaggle using its API. To enable this:

Download your Kaggle API token from your Kaggle account.

Place the kaggle.json file in the .kaggle folder on your system.

Use the Kaggle API to download the dataset within a Jupyter Notebook.



2. Data Preprocessing

The dataset is loaded into a pandas DataFrame, and preprocessing steps include:

Cleaning text data (removing special characters, converting to lowercase, etc.).

Splitting the data into training and testing subsets.

Encoding labels for classification.



3. Model Training

The XGBoost algorithm is utilized for training the model:

Convert text data into numerical format using a feature extraction method (e.g., TF-IDF or CountVectorizer).

Train the XGBoost classifier.



4. Streamlit Integration

The app.py file is created to provide a user interface for text classification. Users input text, and the application predicts whether it contains hate speech.


5. Running the Application

Install required libraries using pip install -r requirements.txt.

Run the application using Streamlit:

streamlit run app.py

Technologies Used

Python: For preprocessing, training, and application development.

XGBoost: As the machine learning algorithm for classification.

Pandas and NumPy: For data manipulation.

Streamlit: For creating an interactive user interface.

Kaggle API: For dataset extraction.



How to Use

Clone the repository:

git clone <repository-url>
cd <repository-folder>

Place the Kaggle kaggle.json file in the appropriate directory.

Download the dataset using Kaggle API.

Preprocess the data and train the model as per the steps above.

Run the Streamlit application:

streamlit run app.py

Input text in the application to detect hate speech.

Conclusion

This project demonstrates the complete lifecycle of a machine learning application, from dataset extraction to deployment. The XGBoost algorithm ensures accurate classification, and Streamlit provides an easy-to-use interface for real-time predictions.

