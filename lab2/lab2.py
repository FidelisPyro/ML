# Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load nbaallelo_log.csv into a dataframe
NBA = pd.read_csv('nbaallelo_log.csv')

# Create binary feature for game_result with 0 for L and 1 for W
NBA['win'] = np.where(NBA['game_result'] == 'W', 1, 0)

# Store relevant columns as variables
X = NBA[['elo_i']]
y = NBA[['win']]

# Initialize and fit the logistic model using the LogisticRegression() function
model = LogisticRegression()
model.fit(X, y.values.ravel())

# Print the weights for the fitted model
print('w1:', model.coef_)

# Print the intercept of the fitted model
print('w0:', model.intercept_)

# Find the proportion of instances correctly classified
score = model.score(X, y)
print(round(score, 3))
