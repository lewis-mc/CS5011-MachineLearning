# command line input: python train-and-predict.py <train-input-file[0]> <train-labels-file[1]> <test-input-file[2]> <numerical-preprocessing[3]> <categorical-preprocessing[4]> <model-type[5]> <test-prediction-output-file[6]>
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import accuracy_score

# For each column (except 'id'), if it is of type object, try converting to datetime
# If more than 50% of the entries can be parsed as dates, extract features (year, month, day) and drop the original column
def process_datetime_features(df, date_format="%Y-%m-%d"):
   
    df = df.copy()
    for col in df.columns:
        if col.lower() == "id":
            continue
        if df[col].dtype == "object":
            dt_series = pd.to_datetime(df[col], errors="coerce", format=date_format)
            if dt_series.notna().mean() > 0.5:  
                df[col + "_year"] = dt_series.dt.year
                df[col + "_month"] = dt_series.dt.month
                df[col + "_day"] = dt_series.dt.day
                df.drop(columns=[col], inplace=True)
    return df

def main():

    # Read in command line arguements
    if len(sys.argv) != 8:
        print("Incorrect amount of arguements entered.")
        print("Usage: python train-and-predict.py <train-input-file> <train-labels-file> <test-input-file> <numerical-preprocessing> <categorical-preprocessing> <model-type> <test-prediction-output-file>")
        sys.exit(1)
    else:
        _, arg_train_input, arg_train_label, arg_test_input, arg_num_preprocessing, arg_cat_preprocessing, arg_model_type, arg_test_pred_output = sys.argv 

    # # Read in data
    data_dir = "../data"
    train_input_values = pd.read_csv(f"{data_dir}/{arg_train_input}")
    train_input_labels = pd.read_csv(f"{data_dir}/{arg_train_label}")
    test_values = pd.read_csv(f"{data_dir}/{arg_test_input}")


    numeric_features = selector(dtype_include=["int64", "float64"])(train_input_values)
    categorical_features = selector(dtype_include=["object", "bool"])(train_input_values)

    train_input_values = process_datetime_features(train_input_values)

    # Merge training inputs with labels
    if "id" in train_input_values.columns and "id" in train_input_labels.columns:
        merged_train = pd.merge(train_input_values, train_input_labels, on="id", how="inner")
    else:
        merged_train = train_input_values.copy()
        if "status_group" in train_input_labels.columns:
            merged_train["status_group"] = train_input_labels["status_group"]
        else:
            merged_train["status_group"] = train_input_labels.iloc[:, 0]

    # Separate features and target, dropping 'id' and 'status_group' from features.
    if "status_group" not in merged_train.columns:
        print("Error: 'status_group' column not found in merged training data.")
        sys.exit(1)
    y = merged_train["status_group"]
    X = merged_train.drop(columns=["id", "status_group"])

    # Now compute the column selectors on X (which no longer has 'id'):
    numeric_features = selector(dtype_include=["int64", "float64"])(X)
    categorical_features = selector(dtype_include=["object", "bool"])(X)

    result, trained_pipeline = run_for_setup(arg_num_preprocessing, arg_cat_preprocessing, arg_model_type, arg_test_pred_output, numeric_features, categorical_features, X, y )

    print(result)
    
    output_file = "../evaluation/output_modified.csv"
    with open(output_file, "a") as f:
        f.write(result)

    # Process test data in the same way (e.g., datetime features)
    test_values = process_datetime_features(test_values)
    # If there is an 'id' column, preserve it for output and drop it before prediction
    if "id" in test_values.columns:
        test_ids = test_values["id"]
        X_test = test_values.drop(columns=["id"])
    else:
        X_test = test_values
        test_ids = None
    
    # Predict on test data using the trained pipeline
    predictions = trained_pipeline.predict(X_test)
    
    # Save the predictions to the output file.
    # If an 'id' column is available, include it in the output.
    if test_ids is not None:
        pred_df = pd.DataFrame({"id": test_ids, "prediction": predictions})
    else:
        pred_df = pd.DataFrame({"prediction": predictions})
    
    pred_df.to_csv(arg_test_pred_output, index=False)
    print(f"Test predictions saved to {arg_test_pred_output}")



def run_for_setup(num_preprocessing, cat_preprocessing, model_type, test_output, num_features, cat_features, X, y):

    print(f"Starting {model_type} {num_preprocessing} {cat_preprocessing}")

    # Select numerical preprocessing technique
    if num_preprocessing == "StandardScaler":
        num_transformer = StandardScaler()
    elif num_preprocessing == "None":
        num_transformer = "passthrough"
    else:
        print("Error: Invalid <numerical-preprocessing> arguement entered")
        sys.exit(1)
    
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', num_transformer)
    ])

    ordering_dict = {
        "water_quality": ['soft', 'milky', 'coloured', 'fluoride', 'salty', 'fluoride abandoned', 'salty abandoned', 'unknown'],
        "quality_group": ['good', 'milky', 'colored', 'fluoride', 'salty', 'unknown'],
        "quantity": ['enough', 'seasonal', 'insufficient', 'dry', 'unknown'],
        "quantity_group": ['enough', 'seasonal', 'insufficient', 'dry', 'unknown']
    }

    # List of the ordinal columns (in the order you want them encoded)
    ordinal_columns = list(ordering_dict.keys())

    # Build the list of category orders in the same order:
    categories_list = [ordering_dict[col] for col in ordinal_columns]

    sparse_param = True

    # Choose model type
    if model_type == "LogisticRegression":
        classifier = LogisticRegression(max_iter=1000)
    elif model_type == "RandomForestClassifier":
        classifier = RandomForestClassifier()
        sparse_param=False
    elif model_type == "GradientBoostingClassifier":
        classifier = GradientBoostingClassifier()
        sparse_param=False
    elif model_type == "HistGradientBoostingClassifier":
        classifier = HistGradientBoostingClassifier()
        sparse_param=False
    elif model_type == "MLPClassifier":
        classifier = MLPClassifier(max_iter=1000)
        sparse_param = False
    else:
        print("Error: Invalid <model-type> argument. "
                "Choose from LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, "
                "HistGradientBoostingClassifier, or MLPClassifier.")
        sys.exit(1)

    # Choose categorical encoding
    if cat_preprocessing == "OneHotEncoder":
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=sparse_param, min_frequency=0.05)
    elif cat_preprocessing == "OrdinalEncoder":
        cat_transformer = OrdinalEncoder(categories=categories_list)
        cat_features = ordinal_columns
    elif cat_preprocessing == "TargetEncoder":
        cat_transformer = TargetEncoder(smooth=0.1)

    to_str = FunctionTransformer(lambda X: X.astype(str))
    cat_pipeline = Pipeline(steps=[
        ('convert', to_str),
        ('imputer', SimpleImputer(strategy="constant", fill_value="missing")),
        ('encoder', cat_transformer)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ],
        verbose=True)

    # Choose model type
    if model_type == "LogisticRegression":
        classifier = LogisticRegression(max_iter=1000)
    elif model_type == "RandomForestClassifier":
        classifier = RandomForestClassifier()
    elif model_type == "GradientBoostingClassifier":
        classifier = GradientBoostingClassifier()
    elif model_type == "HistGradientBoostingClassifier":
        classifier = HistGradientBoostingClassifier()
    elif model_type == "MLPClassifier":
        classifier = MLPClassifier(max_iter=1000)
    else:
        print("Error: Invalid <model-type> argument. "
                "Choose from LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, "
                "HistGradientBoostingClassifier, or MLPClassifier.")
        sys.exit(1)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    start_time = time.time()
    cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
    end_time = time.time()
    elapsed = end_time - start_time
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    output = model_type + "," + cat_preprocessing + "," + num_preprocessing + "," + str(mean_score) + "," + str(std_score) + "," + str(elapsed) + "\n"
    
    pipeline.fit(X, y)

    return output, pipeline

if __name__ == "__main__":
    main()