import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

#Load dataset
df= pd.read_csv("LA_Airbnb.csv")

# Remove rows with missing target values
df = df.dropna(subset=["price"])

#Split the dataset into the input features and target
features = [
    "host_response_time",
    "host_response_rate",
    "host_is_superhost",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "instant_bookable"
]

X= df[features]
y=df["price"]

#Split the dataset into training set and test set
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

numerical_features= X.select_dtypes(include=["int64", "float64"]).columns
categorical_features= X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ]),
            categorical_features
        )
    ]
)

X_train= preprocessor.fit_transform(X_train)
X_test= preprocessor.transform(X_test)

#Train the Linear regression model
model= LinearRegression()
model.fit(X_train, y_train)
y_pred= model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)


# Save model and preprocessing
joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor
    },
    "airbnb_model.pkl"
)