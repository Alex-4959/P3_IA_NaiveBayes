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
print(" - Quantes instàncies conté? " f"Instàncies: {df.shape[0]}")
print(" - Quantes variables té? "f"Variables: {df.shape[1]}")
