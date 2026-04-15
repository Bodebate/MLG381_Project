import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#Load Data
train = pd.read_csv(filepath_or_buffer="DATA/Train.csv")
test = pd.read_csv(filepath_or_buffer="DATA/Test.csv")
targetColumn = "diabetes_stage"
X_test = test.drop(columns=targetColumn)
y_test = test[targetColumn]
X_train = train.drop(columns=targetColumn)
y_train = train[targetColumn]

#save original categorical options before encoding
categoricalColumns = [col for col in train.select_dtypes(include=["object"]).columns if col != targetColumn]
categoryMaps = {}

for col in categoricalColumns:
    train[col] = train[col].astype("category")
    categoryMaps[col] = list(train[col].cat.categories)
    train[col] = train[col].cat.codes

#encoding the target
train[targetColumn] = train[targetColumn].astype("category")
targetMap = list(train[targetColumn].cat.categories)
train[targetColumn] = train[targetColumn].cat.codes


#Random Forest
rfModel = RandomForestClassifier(random_state=42)
rfModel.fit(X_train, y_train)
rfPred = rfModel.predict(X_test)
rfAccuracy = accuracy_score(y_test, rfPred)

#Decision Tree
dtModel = DecisionTreeClassifier(random_state=42)
dtModel.fit(X_train, y_train)
dtPred = dtModel.predict(X_test)
dtAccuracy = accuracy_score(y_test, dtPred)

#XGBoost 
xgbModel = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
xgbModel.fit(X_train, y_train)
xgbPred = xgbModel.predict(X_test)
xgbAccuracy = accuracy_score(y_test,xgbPred)

#results of model training
print("Random Forest Accuracy:", rfAccuracy)
print("Decision Tree Accuracy:", dtAccuracy)
print("xgBoost Accuracy:",xgbAccuracy)
print(classification_report(y_test, rfPred))

#saving each model and metadata
rfModelBundle = {
    "model": rfModel,
    "featureColumns": list(X_train.columns),
    "categoricalColumns": categoricalColumns,
    "categoryMaps": categoryMaps,
    "targetMap": targetMap
}
joblib.dump(rfModelBundle, "ARTIFACTS/DiabetesRFModel.pkl")

dtModelBundle = {
    "model": dtModel,
    "featureColumns": list(X_train.columns),
    "categoricalColumns": categoricalColumns,
    "categoryMaps": categoryMaps,
    "targetMap": targetMap
}
joblib.dump(rfModelBundle, "ARTIFACTS/DiabetesDTModel.pkl")


xgbModelBundle = {
    "model": xgbModel,
    "featureColumns": list(X_train.columns),
    "categoricalColumns": categoricalColumns,
    "categoryMaps": categoryMaps,
    "targetMap": targetMap
}
joblib.dump(xgbModelBundle, "ARTIFACTS/DiabetesXGBModel.pkl")

Predictions= pd.DataFrame({"Actual":y_test,"RF_Model_Predictions":rfPred,"DT_Model_Predictions":dtPred,"XGB_Model_Predictions":xgbPred})
Predictions.to_csv(path_or_buf="ARTIFACTS/Predictions.csv",index=False)