import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

days = np.arange(1, 31)
rainfall = 0.8 * days + np.random.normal(0, 3, size=len(days)) + 10

df = pd.DataFrame({"day": days, "rainfall": rainfall})

X = df[["day"]]
y = df["rainfall"]

model = LinearRegression().fit(X, y)
df["predicted"] = model.predict(X)

r2 = r2_score(y, df["predicted"])
rmse = np.sqrt(mean_squared_error(y, df["predicted"]))   # <-- changed line

print(f"Linear model: rainfall = {model.intercept_:.2f} + {model.coef_[0]:.2f} × day")
print(f"R² = {r2:.3f}, RMSE = {rmse:.3f}")

plt.figure(figsize=(8,5))
plt.scatter(df["day"], df["rainfall"], color="royalblue", label="Generated data")
plt.plot(df["day"], df["predicted"], color="crimson", linewidth=2, label="Linear fit")
plt.title("Simple Linear Regression Model (Synthetic Data)")
plt.xlabel("Day")
plt.ylabel("Rainfall (mm/day)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
