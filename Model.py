import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


def train_and_get_components(filepath):
    """
    Loads data, trains the model, and returns all necessary components
    for the web application.
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    # Keep a clean copy to get lists for dropdown menus
    original_df = df.copy()

    # --- Data Preprocessing ---
    df.drop('Patient_ID', axis=1, inplace=True)
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure_mmHg'].str.split('/', expand=True)
    df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
    df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])
    df.drop('Blood_Pressure_mmHg', axis=1, inplace=True)

    # Get unique lists for web form dropdowns from the original data
    symptoms = pd.unique(original_df[['Symptom_1', 'Symptom_2', 'Symptom_3']].values.ravel('K'))
    symptoms = sorted([s for s in symptoms if pd.notna(s)])
    genders = sorted(original_df['Gender'].unique().tolist())

    # Convert categorical columns to numeric using one-hot encoding
    categorical_cols = ['Gender', 'Symptom_1', 'Symptom_2', 'Symptom_3']
    df = pd.get_dummies(df, columns=categorical_cols)

    # Drop columns that are determined after a diagnosis
    df.drop(['Severity', 'Treatment_Plan'], axis=1, inplace=True)

    # Encode the target variable 'Diagnosis'
    le_diagnosis = LabelEncoder()
    df['Diagnosis_Encoded'] = le_diagnosis.fit_transform(df['Diagnosis'])
    df.drop('Diagnosis', axis=1, inplace=True)
    df.fillna(0, inplace=True)

    # --- Model Training ---
    X = df.drop('Diagnosis_Encoded', axis=1)
    y = df['Diagnosis_Encoded']

    # Align columns to ensure prediction works with all possible inputs
    # Create a base dataframe with all possible dummy columns set to 0
    base_cols = pd.get_dummies(original_df[categorical_cols]).columns
    all_possible_cols = X.drop(columns=base_cols, errors='ignore').columns.tolist() + base_cols.tolist()
    X = X.reindex(columns=all_possible_cols, fill_value=0)

    # Use Decision Tree Classifier instead of Random Forest
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    print("Model trained successfully.")

    # Return all the components the app will need
    return model, X.columns.tolist(), le_diagnosis, symptoms, genders

