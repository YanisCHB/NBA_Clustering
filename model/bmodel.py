import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, delimiter=';')

    # Merge positions
    df['Pos'] = df['Pos'].replace({
        'SG-SF': 'SF',
        'SG-PG': 'SG',
        'PF-SF': 'PF',
        'C-PF': 'C'
    })

    # Drop unnecessary columns
    X = df.drop(columns=['Rk', 'Player', 'Tm', 'Year', 'Pos'])
    y = df['Pos']

    return X, y, X.columns

def svm(donnees, model):
    # Predict using the trained model
    y_pred = model.predict(donnees)

    return y_pred

def train_svm_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    return model, X_test, y_test
