# Codi de la prac tres

#Importem les llibreries que ens diu el professor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy  import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Creem el data seed
print("1. Càrrega del dataset")
df =  pd.read_csv("star_wars_character_dataset.csv")

print("2. Anàlisi inicial del dataset")
print(" Inspecciona el dataset i respon:")
print("     - Quantes instàncies conté? " f"Instàncies: {df.shape[0]}")
print("     - Quantes variables té? "f"Variables: {df.shape[1]}")

print("3. Selecció de variables")
print("Suposa que volem construir un model que predigui el planeta d’origen (homeworld).")
print(" - Creus que totes les variables són rellevants? No")
print(" - Quines variables consideres importants? Height, mass, hair_color, skin_color, eye_color, species, sex")
print(" - Quines eliminaries? name, birth_year, gender, films, vehicles, starships")
print(" - Elimina les variables que consideris no rellevants i justifica la decisió")
df = df.drop(columns=["name", "birth_year", "gender", "films", "vehicles", "starships"])
print(" Les variables que s'han eliminat no tenen res a veure amb el planeta d'origen, per tant no aporten informació rellevant per a la predicció.")

print("4. Separació de variables (X i Y)")
print("En data mining es distingeix entre variables predictores (X) i variable objectiu (Y)")
print("Separa el dataset en:")
print(" - conjunt de variables d’entrada (X)")
x=df.drop(columns=["homeworld"])
print(x)

print(" - variable objectiu (Y)")
y=df.drop(columns=["height", "mass", "hair_color", "skin_color", "eye_color", "species", "sex"])
print(y)

# Omplim amb la mitjana de la columna
df['height'] = df['height'].fillna(df['height'].mean())
df['mass'] = df['mass'].fillna(df['mass'].mean())

# Omplim amb el valor més freqüent (moda)
df['hair_color'] = df['hair_color'].fillna(df['hair_color'].mode()[0])
df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
df['species'] = df['species'].fillna(df['species'].mode()[0])

# Eliminem les files on la classe és desconeguda
df = df.dropna(subset=['homeworld'])

print("5. Train / Test split")
print("En data mining, el model s’entrena amb un conjunt de dades i s’avalua amb dades noves.")
print(" - Divideix el dataset en conjunt d’entrenament i de test")
X_train, X_test, y_train, y_test = train_test_split(x, y)
print(f"Instàncies d'entrenament: {len(X_train)}")
print(f"Instàncies de test: {len(X_test)}")

print(" - Quin percentatge utilitza per defecte la llibreria?")
print("     Per defecte, la llibreria utilitza un 75% per a l'entrenament i un 25% per al test.")

print(" - Modifica’l perquè el conjunt de test sigui del 30% ")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(f"Instàncies d'entrenament: {len(X_train)}")
print(f"Instàncies de test: {len(X_test)}")

print("6. construcció de models")
print(" Construeix i entrena els següents models (de scikit-learn) utilitzant el conjunt d’entrenament:")
print('Un model DummyClassifier amb estratègia "most_frequent" ')
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
print("Un model Naive Bayes (GaussianNB) ")
gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train.values.ravel())

print("7. Avaluació dels models")
print(" Compara els resultats dels dos models:")
print(" Calcula les mètriques de rendiment")
y_pred_dummy = dummy_clf.predict(X_test)
y_pred_gnb = gnb_clf.predict(X_test)
print("Mostra les matrius de confusió ")
print("DummyClassifier:")
print(confusion_matrix(y_test, y_pred_dummy))
print("Analitza els resultats obtinguts i respon")
print("Explica per què els resultats són baixos en aquest dataset.")
print("Els resultats són baixos perquè el dataset és petit i desequilibrat," \
 " amb moltes classes diferents i poques instàncies per classe. " \
 "Això dificulta que els models puguin aprendre patrons significatius per a la predicció del planeta d'origen.")


