#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd

df = pd.read_csv(r"D:\real_estate_project\data\india_housing_prices.csv")
df.head()


# In[63]:


df.shape


# In[64]:


df.info


# In[65]:


df.describe


# In[66]:


print(df.isna().sum().sum())      # total missing values
print(df.duplicated().sum())      # duplicate rows


# In[67]:


bad_floor = df["Floor_No"] > df["Total_Floors"]
print(bad_floor.sum(), bad_floor.mean())


# In[68]:


df["Floor_Invalid"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)   # remember which rows were wrong
df["Floor_No"] = df[["Floor_No", "Total_Floors"]].min(axis=1)              # cap floor at the building height


# In[69]:


col = "Price_per_SqFt"
q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(low, high)
print("Outliers:", ((df[col] < low) | (df[col] > high)).sum())


# In[70]:


df["PPSF_Capped"] = df[col].clip(lower=low, upper=high)


# In[71]:


print((df["Age_of_Property"] + df["Year_Built"]).unique())


# In[72]:


df = df.drop(columns=["Age_of_Property"])


# In[73]:


df.to_csv("../data/cleaned_housing.csv", index=False)


# In[74]:


amenity_list = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]

for a in amenity_list:
    df["Has_" + a] = df["Amenities"].str.contains(a).astype(int)

df["Amenity_Count"] = df[["Has_" + a for a in amenity_list]].sum(axis=1)
df[["Amenities", "Has_Gym", "Amenity_Count"]].head()


# In[75]:


transport_map = {"Low": 1, "Medium": 2, "High": 3}
df["Transport_Score"] = df["Public_Transport_Accessibility"].map(transport_map)

df["Infrastructure_Score"] = (
    df["Nearby_Schools"] / 10 + df["Nearby_Hospitals"] / 10 + df["Transport_Score"] / 3
) / 3


# In[76]:


df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"]


# In[77]:


df["Parking_Space"] = df["Parking_Space"].map({"No": 0, "Yes": 1})
df["Security"] = df["Security"].map({"No": 0, "Yes": 1})


# In[78]:


df[["Amenity_Count", "Infrastructure_Score", "Floor_Ratio", "Parking_Space"]].describe()


# In[79]:


df.to_csv("../data/cleaned_housing.csv", index=False)


# In[80]:


cols = ["Amenity_Count", "Infrastructure_Score", "Floor_Ratio", "Parking_Space"]
df[cols].agg(["min", "max", "mean"]).T


# In[81]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


# In[82]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("../data/cleaned_housing.csv")
df.shape


# In[83]:


# Q1. Distribution of property prices

plt.figure(figsize=(8, 4))
sns.histplot(df["Price_in_Lakhs"], bins=50)
plt.title("Distribution of Property Prices")
plt.xlabel("Price (Lakhs)")
plt.savefig("../charts/q1_price_distribution.png", bbox_inches="tight")
plt.show()


# In[84]:


# Q2. Distribution of property sizes

plt.figure(figsize=(8, 4))
sns.histplot(df["Size_in_SqFt"], bins=50)
plt.title("Distribution of Property Sizes")
plt.xlabel("Size (sq ft)")
plt.savefig("../charts/q2_size_distribution.png", bbox_inches="tight")
plt.show()


# In[85]:


# Q3. Price per sq ft by property type

plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt")
plt.title("Price per Sq Ft by Property Type")
plt.savefig("../charts/q3_ppsf_by_type.png", bbox_inches="tight")
plt.show()


# In[86]:


# Q4. Relationship between size and price

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df.sample(5000, random_state=42), x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.3)
plt.title("Size vs Price")
plt.savefig("../charts/q4_size_vs_price.png", bbox_inches="tight")
plt.show()

print(df["Size_in_SqFt"].corr(df["Price_in_Lakhs"]))


# In[87]:


# Q5. Outliers in price per sq ft and size

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df["Price_per_SqFt"], ax=axes[0])
axes[0].set_title("Price per Sq Ft")
sns.boxplot(y=df["Size_in_SqFt"], ax=axes[1])
axes[1].set_title("Size")
plt.savefig("../charts/q5_outliers.png", bbox_inches="tight")
plt.show()


# In[88]:


# Q6. Average price per sq ft by state

state_ppsf = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
state_ppsf.plot(kind="barh")
plt.title("Average Price per Sq Ft by State")
plt.xlabel("Price per Sq Ft (Lakhs)")
plt.gca().invert_yaxis()
plt.savefig("../charts/q6_ppsf_by_state.png", bbox_inches="tight")
plt.show()


# In[89]:


# Q7. Average property price by city

city_price = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 10))
city_price.plot(kind="barh")
plt.title("Average Property Price by City")
plt.xlabel("Price (Lakhs)")
plt.gca().invert_yaxis()
plt.savefig("../charts/q7_price_by_city.png", bbox_inches="tight")
plt.show()


# In[90]:


# Q8. Median age of properties by locality

df["Age"] = 2025 - df["Year_Built"]

locality_age = df.groupby("Locality")["Age"].median()
print(locality_age.describe())

plt.figure(figsize=(8, 4))
sns.histplot(locality_age, bins=20)
plt.title("Median Property Age Across 500 Localities")
plt.xlabel("Median Age (years)")
plt.savefig("../charts/q8_median_age_by_locality.png", bbox_inches="tight")
plt.show()

print(locality_age.nlargest(5))    # 5 localities with the oldest properties
print(locality_age.nsmallest(5))   # 5 with the newest


# In[91]:


# Q9. BHK distribution across cities

bhk_city = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100

plt.figure(figsize=(8, 12))
sns.heatmap(bhk_city, annot=True, fmt=".0f", cmap="Blues")
plt.title("BHK Share by City (%)")
plt.savefig("../charts/q9_bhk_by_city.png", bbox_inches="tight")
plt.show()


# In[92]:


# Q10. Price trends for the top 5 most expensive localities

top5 = df.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5).index
print(list(top5))

trend = (df[df["Locality"].isin(top5)]
         .groupby(["Year_Built", "Locality"])["Price_in_Lakhs"].mean()
         .reset_index())

plt.figure(figsize=(10, 5))
sns.lineplot(data=trend, x="Year_Built", y="Price_in_Lakhs", hue="Locality")
plt.title("Average Price by Year Built: Top 5 Localities")
plt.savefig("../charts/q10_top5_trend.png", bbox_inches="tight")
plt.show()


# In[93]:


# Q11. Correlation between all numeric feature

num_cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
            "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
            "Amenity_Count", "Infrastructure_Score", "Floor_Ratio"]

plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Between Numeric Features")
plt.savefig("../charts/q11_correlation.png", bbox_inches="tight")
plt.show()


# In[94]:


# Q12 and Q13. Schools and hospitals vs price per sq ft

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.barplot(data=df, x="Nearby_Schools", y="Price_per_SqFt", ax=axes[0])
axes[0].set_title("Price per Sq Ft by Number of Nearby Schools")

sns.barplot(data=df, x="Nearby_Hospitals", y="Price_per_SqFt", ax=axes[1])
axes[1].set_title("Price per Sq Ft by Number of Nearby Hospitals")

plt.savefig("../charts/q12_q13_schools_hospitals.png", bbox_inches="tight")
plt.show()

print(df["Nearby_Schools"].corr(df["Price_per_SqFt"]))
print(df["Nearby_Hospitals"].corr(df["Price_per_SqFt"]))


# In[95]:


# Q14. Price by furnished status

plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs")
plt.title("Price by Furnished Status")
plt.savefig("../charts/q14_price_by_furnishing.png", bbox_inches="tight")
plt.show()

print(df.groupby("Furnished_Status")["Price_in_Lakhs"].agg(["mean", "median"]))


# In[96]:


# Q15. Price per sq ft by facing direction

plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="Facing", y="Price_per_SqFt")
plt.title("Price per Sq Ft by Facing Direction")
plt.savefig("../charts/q15_ppsf_by_facing.png", bbox_inches="tight")
plt.show()

print(df.groupby("Facing")["Price_per_SqFt"].mean())


# In[97]:


# Q16 and Q17. How many properties per owner type and availability status?

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

sns.countplot(data=df, x="Owner_Type", ax=axes[0])
axes[0].set_title("Properties by Owner Type")

sns.countplot(data=df, x="Availability_Status", ax=axes[1])
axes[1].set_title("Properties by Availability Status")

plt.savefig("../charts/q16_q17_owner_availability.png", bbox_inches="tight")
plt.show()

print(df["Owner_Type"].value_counts())
print(df["Availability_Status"].value_counts())


# In[98]:


# Q18. Does parking affect price?

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="Parking_Space", y="Price_in_Lakhs")
plt.xticks([0, 1], ["No parking", "Parking"])
plt.title("Average Price by Parking Availability")
plt.savefig("../charts/q18_parking_price.png", bbox_inches="tight")
plt.show()

print(df.groupby("Parking_Space")[["Price_in_Lakhs", "Price_per_SqFt"]].mean())


# In[99]:


# Q19. How do amenities affect price per sq ft?

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.barplot(data=df, x="Amenity_Count", y="Price_per_SqFt", ax=axes[0])
axes[0].set_title("Price per Sq Ft by Number of Amenities")

amenity_effect = {}
for a in ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]:
    amenity_effect[a] = df.groupby("Has_" + a)["Price_per_SqFt"].mean()[1]
pd.Series(amenity_effect).plot(kind="bar", ax=axes[1])
axes[1].set_title("Avg Price per Sq Ft: Properties WITH Each Amenity")

plt.savefig("../charts/q19_amenities.png", bbox_inches="tight")
plt.show()


# In[100]:


# Q20. Public transport vs price per sq ft

order = ["Low", "Medium", "High"]

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Price_per_SqFt", order=order)
plt.title("Price per Sq Ft by Public Transport Access")
plt.savefig("../charts/q20_transport.png", bbox_inches="tight")
plt.show()

print(df.groupby("Public_Transport_Accessibility")["Price_per_SqFt"].mean().reindex(order))


# In[101]:


import numpy as np

# 1) Is this property cheaper per sq ft than its city's median?
city_median_ppsf = df.groupby("City")["Price_per_SqFt"].transform("median")
df["Undervalued"] = (df["Price_per_SqFt"] <= city_median_ppsf).astype(int)

# 2) Annual growth rate
rng = np.random.default_rng(42)
df["Growth_Rate"] = (
    0.05
    + 0.04 * df["Infrastructure_Score"]
    + 0.01 * df["Amenity_Count"] / 5
    + 0.015 * df["Undervalued"]
    + rng.normal(0, 0.005, len(df))
)

# 3) Price after 5 years
df["Price_5Y_Lakhs"] = df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** 5


# In[102]:


df["Good_Investment"] = (df["Growth_Rate"] >= 0.09).astype(int)


# In[103]:


df["Good_Investment"] = (df["Growth_Rate"] >= 0.09).astype(int)


# In[104]:


print(df["Good_Investment"].value_counts(normalize=True))
print(df["Growth_Rate"].describe())
print((df["Price_5Y_Lakhs"] / df["Price_in_Lakhs"]).describe())
print(df.groupby("Undervalued")["Good_Investment"].mean())


# In[105]:


df.to_csv("../data/cleaned_housing.csv", index=False)


# In[106]:


amenity_list = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]

numeric_cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No",
                "Total_Floors", "Floor_Ratio", "Nearby_Schools", "Nearby_Hospitals",
                "Transport_Score"]

binary_cols = ["Parking_Space", "Security", "Floor_Invalid"] + ["Has_" + a for a in amenity_list]

cat_cols = ["City", "Property_Type", "Furnished_Status", "Facing",
            "Owner_Type", "Availability_Status"]

feature_cols = numeric_cols + binary_cols + cat_cols

X = df[feature_cols]
y_class = df["Good_Investment"]
y_reg = df["Price_5Y_Lakhs"]


# In[107]:


from sklearn.model_selection import train_test_split

X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_class, y_reg,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

print(X_train.shape, X_test.shape)
print(yc_train.mean(), yc_test.mean())


# In[108]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("bin", "passthrough", binary_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print(X_train_prep.shape, X_test_prep.shape)


# In[109]:


import numpy as np
print(np.round(X_train_prep[:, :len(numeric_cols)].mean(axis=0), 2))
print(np.round(X_train_prep[:, :len(numeric_cols)].std(axis=0), 2))


# In[110]:


print(X.isna().sum().sum())


# In[111]:


import time
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)

class_results = {}
reg_results = {}

def evaluate_classifier(name, model):
    start = time.time()
    model.fit(X_train_prep, yc_train)
    pred = model.predict(X_test_prep)
    prob = model.predict_proba(X_test_prep)[:, 1]
    class_results[name] = {
        "accuracy": accuracy_score(yc_test, pred),
        "precision": precision_score(yc_test, pred, zero_division=0),
        "recall": recall_score(yc_test, pred),
        "f1": f1_score(yc_test, pred),
        "roc_auc": roc_auc_score(yc_test, prob),
    }
    print(name, {k: round(v, 3) for k, v in class_results[name].items()},
          "| time:", round(time.time() - start, 1), "s")
    return model

def evaluate_regressor(name, model):
    start = time.time()
    model.fit(X_train_prep, yr_train)
    pred = model.predict(X_test_prep)
    reg_results[name] = {
        "rmse": mean_squared_error(yr_test, pred) ** 0.5,
        "mae": mean_absolute_error(yr_test, pred),
        "r2": r2_score(yr_test, pred),
    }
    print(name, {k: round(v, 4) for k, v in reg_results[name].items()},
          "| time:", round(time.time() - start, 1), "s")
    return model


# In[112]:


baseline_clf = evaluate_classifier("Baseline (always majority)", DummyClassifier(strategy="most_frequent"))
log_reg = evaluate_classifier("Logistic Regression", LogisticRegression(max_iter=1000))
rf_clf = evaluate_classifier("Random Forest", RandomForestClassifier(
    n_estimators=100, max_depth=12, n_jobs=-1, random_state=42))


# In[113]:


mean_baseline = evaluate_regressor("Baseline (predict average)", DummyRegressor())
lin_reg = evaluate_regressor("Linear Regression", LinearRegression())
rf_reg = evaluate_regressor("Random Forest", RandomForestRegressor(
    n_estimators=100, max_depth=12, n_jobs=-1, random_state=42))


# In[114]:


mult = (yr_train / X_train["Price_in_Lakhs"]).mean()
simple_pred = X_test["Price_in_Lakhs"] * mult
print("multiplier:", round(mult, 3))
print("RMSE:", mean_squared_error(yr_test, simple_pred) ** 0.5)
print("R2:", r2_score(yr_test, simple_pred))


# In[115]:


import pandas as pd
names = preprocessor.get_feature_names_out()
imp = pd.Series(rf_clf.feature_importances_, index=names).sort_values(ascending=False).head(10)
print(imp)


# In[116]:


import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBClassifier, XGBRegressor

mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("Real_Estate_Investment_Advisor")


# In[117]:


def run_classifier(name, model, params):
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("prep", clone(preprocessor)), ("model", model)])
        pipe.fit(X_train, yc_train)

        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(yc_test, pred),
            "precision": precision_score(yc_test, pred),
            "recall": recall_score(yc_test, pred),
            "f1": f1_score(yc_test, pred),
            "roc_auc": roc_auc_score(yc_test, prob),
        }

        mlflow.log_param("task", "classification")
        mlflow.log_param("model_name", name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
        print(name, {k: round(v, 3) for k, v in metrics.items()})
    return pipe


def run_regressor(name, model, params):
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("prep", clone(preprocessor)), ("model", model)])
        pipe.fit(X_train, yr_train)

        pred = pipe.predict(X_test)
        metrics = {
            "rmse": mean_squared_error(yr_test, pred) ** 0.5,
            "mae": mean_absolute_error(yr_test, pred),
            "r2": r2_score(yr_test, pred),
        }

        mlflow.log_param("task", "regression")
        mlflow.log_param("model_name", name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
        print(name, {k: round(v, 4) for k, v in metrics.items()})
    return pipe


# In[118]:


# --- Classification ---
run_classifier("LogReg_clf", LogisticRegression(max_iter=1000), {"max_iter": 1000})

run_classifier("RandomForest_clf",
               RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
               {"n_estimators": 100, "max_depth": 12})

run_classifier("XGB_clf_depth4",
               XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=-1, random_state=42),
               {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1})

best_clf = run_classifier("XGB_clf_depth6",
               XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42),
               {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1})

# --- Regression ---
run_regressor("LinearReg", LinearRegression(), {})

run_regressor("RandomForest_reg",
              RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
              {"n_estimators": 100, "max_depth": 12})

best_reg = run_regressor("XGB_reg",
              XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42),
              {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1})


# In[ ]:


# --- Classification ---
run_classifier("LogReg_clf", LogisticRegression(max_iter=1000), {"max_iter": 1000})

run_classifier("RandomForest_clf",
               RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
               {"n_estimators": 100, "max_depth": 12})

run_classifier("XGB_clf_depth4",
               XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=-1, random_state=42),
               {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1})

best_clf = run_classifier("XGB_clf_depth6",
               XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42),
               {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1})

# --- Regression ---
run_regressor("LinearReg", LinearRegression(), {})

run_regressor("RandomForest_reg",
              RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
              {"n_estimators": 100, "max_depth": 12})

best_reg = run_regressor("XGB_reg",
              XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42),
              {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1})


# In[ ]:


import joblib, os
os.makedirs("../models", exist_ok=True)

joblib.dump(best_clf, "../models/best_classifier.joblib")
joblib.dump(best_reg, "../models/best_regressor.joblib")


# In[ ]:


import json
with open("../models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)


# In[ ]:


jupyter nbconvert --to script notebooks/01_explore.ipynb --output-dir src


# In[ ]:




