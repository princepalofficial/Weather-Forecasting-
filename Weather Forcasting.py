import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Create Dataset
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "day": np.arange(1, n + 1),
    "temperature": np.random.uniform(15,40,n),
    "humidity": np.random.uniform(30,90,n),
    "wind_speed": np.random.uniform(2,30,n),
    "pressure": np.random.uniform(990, 1030, n)
}) 

# 2. create next day temperature
noise = np.random.normal(0,1.5,n)
df["next_day_temperature"] = (df["temperature"] * 0.75 - df["humidity"] * 0.03
                              + df["wind_speed"] * 0.02 + df["pressure"] * 0.01 + noise)

# 3. Dataset
print("\n ****** Dataset ******")
print(df.head())
print("\nShape:", df.shape)

# 4. Feature and Target
x = df[
    [
        "temperature",
        "humidity",
        "wind_speed",
        "pressure"]
]
y = df["next_day_temperature"]

# 5. Train-test Split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42)
print("\n Training data:", x_train.shape)
print("\n Testing data:", x_test.shape)

# 6. Linear Regression Model
model = LinearRegression()

# 7. Model Traing
model.fit(x_train, y_train)
print("\n ***** model Trained *****")

# 8. prediction
y_pred = model.predict(x_test)

# 9. Actual vs predicted
result = pd.DataFrame({
    "Actual": y_test.values,
    "predicted": y_pred
})

print("\n***** Actul vs Predicted ******")
print(result.head(10))

# 10. model Evaluation
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\n ******* model Performance *******")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2_score:", round(r2, 2)
      )
# 11. new weather prediction
new_weather = pd.DataFrame({
    "temperature": [30],
    "humidity": [60],
    "wind_speed": [10],
    "pressure": [1010]
})

prediction = model.predict(new_weather)
print("\n****** New weather ******")
print(new_weather)
print("\nNext day Temperature:",round(prediction[0], 2),"℃")
