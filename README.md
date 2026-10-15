Weather Forecasting Using Machine Learning

About the Project

This is a beginner-friendly Machine Learning project that predicts the next day's temperature using weather-related features. It uses Python, NumPy, Pandas, and Scikit-learn.

Note: This project uses synthetic (artificially generated) data for learning and demonstration. It does not use real historical weather data or provide reliable real-world weather forecasts.

Objective

- Create a synthetic weather dataset.
- Explore and prepare weather data.
- Train a Linear Regression model.
- Predict the next day's temperature.
- Evaluate model performance using common regression metrics.

Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

Dataset Features

The program generates 1,000 synthetic records with these columns:

Feature| Description
"day"| Day number
"temperature"| Current temperature
"humidity"| Humidity level
"wind_speed"| Wind speed
"pressure"| Atmospheric pressure
"next_day_temperature"| Generated target temperature for the next day

The target temperature is generated using a mathematical formula with random noise.

Machine Learning Algorithm

Linear Regression

The dataset is divided into training and testing sets using an 80:20 split. The model learns from the training data and predicts temperatures for the test data.

Model Evaluation

The program calculates these evaluation metrics:

- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- R² Score: Coefficient of determination

These metrics evaluate predictions against the generated dataset; they do not establish real-world weather forecasting accuracy.

Installation

Install the required libraries:

pip install numpy pandas scikit-learn

How to Run

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Run the Python file:

python "Weather Forcasting.py"

Output

When you run the program, it displays:

- The first few rows of the dataset
- Training and testing data shapes
- Actual versus predicted temperatures
- Model evaluation metrics
- A prediction for a sample weather input

Future Improvements

- Use real historical weather data.
- Add data visualization.
- Compare multiple Machine Learning algorithms.
- Test the model on unseen real-world weather data.

Author

Prince Pal

GitHub: https://github.com/princepalofficial
