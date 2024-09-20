import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data

# Train the model
def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# Sidebar for user input
def user_input_features():
    st.sidebar.header("User Input Features")
    
    def user_inputs():
        mean_radius = st.sidebar.slider('Mean Radius', 6.0, 30.0, 14.0)
        mean_texture = st.sidebar.slider('Mean Texture', 9.0, 40.0, 19.0)
        mean_perimeter = st.sidebar.slider('Mean Perimeter', 40.0, 200.0, 90.0)
        mean_area = st.sidebar.slider('Mean Area', 140.0, 2500.0, 700.0)
        mean_smoothness = st.sidebar.slider('Mean Smoothness', 0.05, 0.2, 0.1)
        mean_compactness = st.sidebar.slider('Mean Compactness', 0.1, 1.0, 0.3)
        mean_concavity = st.sidebar.slider('Mean Concavity', 0.0, 1.0, 0.1)
        mean_concave_points = st.sidebar.slider('Mean Concave Points', 0.0, 1.0, 0.1)
        mean_symmetry = st.sidebar.slider('Mean Symmetry', 0.0, 1.0, 0.5)
        mean_fractal_dimension = st.sidebar.slider('Mean Fractal Dimension', 0.0, 0.5, 0.1)
        radius_error = st.sidebar.slider('Radius Error', 0.0, 0.5, 0.1)
        texture_error = st.sidebar.slider('Texture Error', 0.0, 0.5, 0.1)
        perimeter_error = st.sidebar.slider('Perimeter Error', 0.0, 0.5, 0.1)
        area_error = st.sidebar.slider('Area Error', 0.0, 0.5, 0.1)
        smoothness_error = st.sidebar.slider('Smoothness Error', 0.0, 0.5, 0.1)
        compactness_error = st.sidebar.slider('Compactness Error', 0.0, 0.5, 0.1)
        concavity_error = st.sidebar.slider('Concavity Error', 0.0, 0.5, 0.1)
        concave_points_error = st.sidebar.slider('Concave Points Error', 0.0, 0.5, 0.1)
        symmetry_error = st.sidebar.slider('Symmetry Error', 0.0, 0.5, 0.1)
        fractal_dimension_error = st.sidebar.slider('Fractal Dimension Error', 0.0, 0.5, 0.1)
        worst_radius = st.sidebar.slider('Worst Radius', 6.0, 30.0, 14.0)
        worst_texture = st.sidebar.slider('Worst Texture', 9.0, 40.0, 19.0)
        worst_perimeter = st.sidebar.slider('Worst Perimeter', 40.0, 200.0, 90.0)
        worst_area = st.sidebar.slider('Worst Area', 140.0, 2500.0, 700.0)
        worst_smoothness = st.sidebar.slider('Worst Smoothness', 0.05, 0.2, 0.1)
        worst_compactness = st.sidebar.slider('Worst Compactness', 0.1, 1.0, 0.3)
        worst_concavity = st.sidebar.slider('Worst Concavity', 0.0, 1.0, 0.1)
        worst_concave_points = st.sidebar.slider('Worst Concave Points', 0.0, 1.0, 0.1)
        worst_symmetry = st.sidebar.slider('Worst Symmetry', 0.0, 1.0, 0.5)
        worst_fractal_dimension = st.sidebar.slider('Worst Fractal Dimension', 0.0, 0.5, 0.1)
        return {
            'mean radius': mean_radius,
            'mean texture': mean_texture,
            'mean perimeter': mean_perimeter,
            'mean area': mean_area,
            'mean smoothness': mean_smoothness,
            'mean compactness': mean_compactness,
            'mean concavity': mean_concavity,
            'mean concave points': mean_concave_points,
            'mean symmetry': mean_symmetry,
            'mean fractal dimension': mean_fractal_dimension,
            'radius error': radius_error,
            'texture error': texture_error,
            'perimeter error': perimeter_error,
            'area error': area_error,
            'smoothness error': smoothness_error,
            'compactness error': compactness_error,
            'concavity error': concavity_error,
            'concave points error': concave_points_error,
            'symmetry error': symmetry_error,
            'fractal dimension error': fractal_dimension_error,
            'worst radius': worst_radius,
            'worst texture': worst_texture,
            'worst perimeter': worst_perimeter,
            'worst area': worst_area,
            'worst smoothness': worst_smoothness,
            'worst compactness': worst_compactness,
            'worst concavity': worst_concavity,
            'worst concave points': worst_concave_points,
            'worst symmetry': worst_symmetry,
            'worst fractal dimension': worst_fractal_dimension
        }

    features = pd.DataFrame(user_inputs(), index=[0])
    return features

# Cancer awareness dashboard content
def cancer_awareness():
    st.title("Cancer Awareness and Importance of Early Detection")
    
    st.subheader("What is Breast Cancer?")
    st.write("""
    Breast cancer is a disease in which malignant (cancer) cells form in the tissues of the breast. 
    Early detection of breast cancer greatly increases the chances of successful treatment. 
    It is crucial for people to regularly undergo screenings, such as mammograms, to detect cancer in its early stages.
    """)
    
    st.subheader("Why is Early Detection Important?")
    st.write("""
    - Early detection means finding cancer at an early stage, before it has spread to other parts of the body.
    - Early detection increases the likelihood of more treatment options and a higher survival rate.
    - Screening methods like self-exams, mammograms, and clinical breast exams can help catch cancer early.

    **Awareness programs** and understanding risk factors, symptoms, and preventive measures are critical to combating breast cancer.
    """)
    
    st.subheader("Key Statistics:")
    st.write("""
    - **1 in 8 women** in the U.S. will develop breast cancer in her lifetime.
    - Breast cancer is the second leading cause of cancer death in women.
    - When detected early, **the 5-year survival rate for breast cancer is 99%**.
    """)

    st.subheader("Preventive Measures:")
    st.write("""
    - Regular screenings and self-exams.
    - Maintaining a healthy diet and exercise.
    - Avoiding excessive alcohol consumption and smoking.
    - Being aware of family medical history and discussing it with your doctor.
    """)

    st.subheader("Support and Resources:")
    st.write("""
    Many organizations offer resources for breast cancer education, support, and early detection:
    
    - [National Breast Cancer Foundation](https://www.nationalbreastcancer.org/)
    - [American Cancer Society](https://www.cancer.org/)
    - [Breastcancer.org](https://www.breastcancer.org/)
    """)

# Main App
def main():
    st.sidebar.title("Navigation")
    
    # Dashboard navigation
    options = ["Breast Cancer Prediction", "Cancer Awareness"]
    selection = st.sidebar.selectbox("Choose a section", options)

    if selection == "Breast Cancer Prediction":
        st.title("Breast Cancer Prediction App")
        
        # Load data
        df, data = load_data()

        # Display dataset and info
        if st.checkbox("Show raw data"):
            st.write(df)

        # User input
        input_df = user_input_features()

        # Display user input
        st.subheader("User Input")
        st.write(input_df)

        # Train the model
        model, accuracy = train_model(df)

        # Show accuracy
        st.subheader("Model Accuracy")
        st.write(f"{accuracy * 100:.2f}%")

        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction")
        cancer_labels = data.target_names
        st.write(f"Prediction: {cancer_labels[prediction][0]}")
        st.subheader("Prediction Probability")
        st.write(prediction_proba)
        st.write("""
                 - 0 = Malignant, 1 = Benign
                 - Malignant means the patient is cancerous, and Benign means the patient is non-cancerous.
                 """)

    elif selection == "Cancer Awareness":
        cancer_awareness()

if __name__ == '__main__':
    main()
