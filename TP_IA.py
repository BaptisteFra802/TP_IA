import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Charger le fichier CSV
file_path = "sensor_raw.csv"
df = pd.read_csv(file_path)

# Afficher les premières lignes
print(df.head())

# Vérifier les valeurs manquantes
print(df.isnull().sum())

# Remplissage des valeurs manquantes (si nécessaire)
df = df.fillna(method='ffill')

# Encodage de la variable cible
if 'Label' in df.columns:
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])

# Séparation des features et de la cible
X = df.drop(columns=['Label'])
y = df['Label']

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialisation du modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle: {accuracy:.4f}")

# Affichage du rapport de classification
print(classification_report(y_test, y_pred))

# Matrice de confusion
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.title("Matrice de Confusion")
plt.show()
