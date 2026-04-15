import pandas as pd
from sklearn.model_selection import train_test_split

#loading dataset
df = pd.read_csv("Diabetes_and_LifeStyle_Dataset.csv")

#cleaning column names
df.columns = df.columns.str.strip()

#defining the target
targetColumn = "diabetes_stage"

#dropping leakage columns if they exist
columnsToDrop = [targetColumn]
for col in ["diagnosed_diabetes", "diabetes_risk_score"]:
    if col in df.columns:
        columnsToDrop.append(col)

#save original categorical options before encoding
categoricalColumns = [col for col in df.select_dtypes(include=["object"]).columns if col != targetColumn]
categoryMaps = {}

for col in categoricalColumns:
    df[col] = df[col].astype("category")
    categoryMaps[col] = list(df[col].cat.categories)
    df[col] = df[col].cat.codes

#encoding the target
df[targetColumn] = df[targetColumn].astype("category")
targetMap = list(df[targetColumn].cat.categories)
df[targetColumn] = df[targetColumn].cat.codes

#features and target
X = df.drop(columns=columnsToDrop)
y = df[targetColumn]

#separate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#saving data split to csv
pd.concat([X_test,y_test],axis=1).to_csv(path_or_buf="DATA/Test.csv",index = False)
pd.concat([X_train,y_train],axis=1).to_csv(path_or_buf="DATA/Train.csv",index = False)