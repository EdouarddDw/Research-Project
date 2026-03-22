import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
import shapiq


# Initialize the shapiq Explainer for the MLP
# We use a subset of the data as background for speed
explainer = shapiq.Explainer(
    model=model.predict,
    data=X_df,
    index="k-SII",     # Shapley Interaction Index
    max_order=2        # Detect up to 2nd-order (pairwise) interactions
)

# Explain a single instance (the first row)
# 'budget' determines the number of model evaluations for the approximation
interaction_values = explainer.explain(X_df.iloc[0], budget=2000)

print(interaction_values)