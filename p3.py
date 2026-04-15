# Codi de la prac tres

#Importem les llibreries que ens diu el professor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy  import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Creem el data seed
df =  pd.read_csv("star_wars_charecter_dataset.csv")

#Agafem variables immportants

df = df