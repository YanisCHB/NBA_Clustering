import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


df = pd.read_csv("/home/aminou/Bureau/data_mm/NBA_Clustering/data/processed/combined_nba_stats_2021_2024.csv", delimiter = ";", header = 0)

df['Pos'] = df['Pos'].replace({
    'SG-SF': 'SF',  
    'SG-PG': 'SG',  # Fusionner SG-PG avec SG
    'PF-SF': 'PF',   # Fusionner PF-SF avec PF
    'C-PF' : 'C'
})



def knn(df):

    # Chargement des données (remplacez par votre propre chemin de fichier ou méthode de chargement)
    # df = pd.read_csv('chemin_du_fichier.csv')

    # Suppression de la colonne des noms de joueurs
    df = df.drop(columns=['Player', 'Tm', 'Year'])  # Remplacez 'player_name' par le nom réel de votre colonne

    # Conversion des variables catégorielles en variables numériques
    df['Pos'] = pd.Categorical(df['Pos']).codes

    # Sélection des colonnes pour le clustering
    features = df.columns.difference(['Pos'])

    # Normalisation des données
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])

    # Application de l'algorithme K-means
    # Détermination du nombre optimal de clusters via la méthode du coude
    inertia = []
    k_values = range(1, 10)  # Testons de 1 à 9 clusters
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    # Tracé de la courbe du coude
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, inertia, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude pour le choix optimal de k')
    plt.show()

    # Choisissez le nombre de clusters basé sur la courbe
    # Par exemple, si le coude est à k=3
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)

    # Ajout des étiquettes de cluster aux données
    df['cluster'] = clusters

    # Affichage des résultats
    print(df.head())

    # Évaluation avec le score de silhouette si nécessaire
    from sklearn.metrics import silhouette_score
    score = silhouette_score(df_scaled, clusters)
    print('Score de silhouette :', score)



def decision_tree_classification(df, features, target):
    """
    Appliquer un arbre de décision pour classifier les données.

    Paramètres:
    df (pd.DataFrame) : DataFrame contenant les données.
    features (list) : Liste des colonnes à utiliser comme variables explicatives.
    target (str) : Nom de la colonne cible à prédire.
    """
    # Préparation des données
    X = df[features]  # Variables explicatives
    y = df[target]    # Variable cible

    # Encoder la variable cible si elle est catégorielle
    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Création et entraînement du modèle d'arbre de décision
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
    y_pred = classifier.predict(X_test)

    # Évaluation du modèle
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

# # Exemple d'utilisation de la fonction # Assurez-vous de remplacer par votre chemin de fichier
# categorical_features = ['Pos']  # Colonnes catégorielles
# numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()  # Colonnes numériques

# # Assurez-vous que 'Pos' est votre variable cible
# all_features = categorical_features + numeric_features
# all_features.remove('Pos')  # Retirez 'Pos' de la liste des features si c'est votre cible
# decision_tree_classification(df, all_features, 'Pos')




def random_forest_classification(df, features, target):
    """
    Appliquer une forêt aléatoire pour classifier les données.

    Paramètres:
    df (pd.DataFrame) : DataFrame contenant les données.
    features (list) : Liste des colonnes à utiliser comme variables explicatives.
    target (str) : Nom de la colonne cible à prédire.
    """
    # Préparation des données
    X = df[features]  # Variables explicatives
    y = df[target]    # Variable cible

    # Encoder la variable cible si elle est catégorielle
    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    # Normalisation des données (optionnel, dépend des données)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Création et entraînement du modèle de forêt aléatoire
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
    y_pred = classifier.predict(X_test)

    # Évaluation du modèle
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

# Exemple d'utilisation de la fonction

categorical_features = ['Pos']  # Colonnes catégorielles
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()  # Colonnes numériques

# Assurez-vous que 'pos' est votre variable cible
all_features = categorical_features + numeric_features
all_features.remove('Pos')  # Retirez 'pos' de la liste des features si c'est votre cible
random_forest_classification(df, all_features, 'Pos')


def svm(donnees):
    # Préparation des données
    df1 = df.drop(columns=['Rk','Player', 'Tm', 'Year']) 
    X = df1.drop('Pos', axis=1)  # Remplacez 'colonne_cible' par le nom de votre variable cible
    y = df1['Pos']

    # Encoder les variables catégorielles si nécessaire
    # X = pd.get_dummies(X)

    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Création et entraînement du modèle SVM
    model = SVC(kernel='linear')  # Vous pouvez changer le kernel ici
    model.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(donnees)

    return y_pred


# Division des données en ensembles d'entraînement et de test



df1 = df.drop(columns=['Rk','Player', 'Pos','Tm', 'Year'])
df1_y = df['Pos']
print(svm(df1.iloc[[1]]))
print(df1_y.iloc[[1]])

print(df1.iloc[[1]])