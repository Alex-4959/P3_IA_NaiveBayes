import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Càrrega del dataset
print("1. Càrrega del dataset")
df = pd.read_csv("star_wars_character_dataset.csv")

# 2. Anàlisi inicial del dataset
print("\n2. Anàlisi inicial del dataset")
print(f" - Quantes instàncies conté? {df.shape[0]}")
print(f" - Quantes variables té? {df.shape[1]}")

# 3. Selecció de variables i eliminació de no rellevants
print("\n3. Selecció de variables")
# Eliminem les variables que no aporten informació segons la teva anàlisi [cite: 70]
df = df.drop(columns=["name", "birth_year", "gender", "films", "vehicles", "starships"])
print(" S'han eliminat les variables no rellevants per a la predicció.")

# --- PREPROCESSAT (CORRECCIÓ CRÍTICA: Fem la neteja ABANS de separar X i Y) ---

# A) Tractament de valors buits (Imputació) [cite: 31]
df['height'] = df['height'].fillna(df['height'].mean())
df['mass'] = df['mass'].fillna(df['mass'].mean())

# Omplim categòrics amb la moda
df['hair_color'] = df['hair_color'].fillna(df['hair_color'].mode()[0])
df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
df['species'] = df['species'].fillna(df['species'].mode()[0])

# B) Eliminem files on la classe (homeworld) és buida [cite: 48]
df = df.dropna(subset=['homeworld'])

# C) Conversió de text a números (Necessari per a GaussianNB)
# Creem un nou dataframe per a les X on tot sigui numèric
X_numeric = pd.DataFrame()
for col in ["hair_color", "skin_color", "eye_color", "sex", "species"]:
    X_numeric[col] = df[col].astype('category').cat.codes

X_numeric['height'] = df['height']
X_numeric['mass'] = df['mass']

# 4. Separació de variables (X i Y) sobre dades ja netes
print("\n4. Separació de variables (X i Y)")
X = X_numeric
y = df['homeworld']
print("X (Atributs numèrics):\n", X.head())

# 5. Train / Test split
print("\n5. Train / Test split")
# Intent per defecte (25% test)
X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(X, y, random_state=42)
print(f" Instàncies test defecte (25%): {len(X_test_def)}")

# Modificació al 30% segons demana la pràctica [cite: 83]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f" Instàncies entrenament (70%): {len(X_train)}")
print(f" Instàncies test (30%): {len(X_test)}")

# 6. Construcció de models
print("\n6. Construcció de models")
# Model Dummy [cite: 87]
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

# Model Naive Bayes 
gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)

# 7. Avaluació dels models
print("\n7. Avaluació dels models")
y_pred_dummy = dummy_clf.predict(X_test)
y_pred_gnb = gnb_clf.predict(X_test)

print("\n--- Matriu de Confusió Dummy ---")
print(confusion_matrix(y_test, y_pred_dummy))

print("\n--- Matriu de Confusió GaussianNB ---")
print(confusion_matrix(y_test, y_pred_gnb))

print("\n--- Informe de Classificació GaussianNB ---")
print(classification_report(y_test, y_pred_gnb, zero_division=0))

# Anàlisi final [cite: 97]
print("\nExplica per què els resultats són baixos en aquest dataset:")
print("Els resultats són baixos perquè el dataset és petit (87 instàncies) i està molt fragmentat.")
print("Hi ha molts planetes d'origen diferents amb només un o dos personatges per planeta,")
print("cosa que fa que el model no tingui prou exemples per aprendre els patrons.")