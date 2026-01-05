from sklearn.linear_model import LinearRegression
import numpy as np

# Dữ liệu học
X = np.array([1, 2, 3, 4]).reshape(-1, 1)
y = np.array([2, 4, 6, 8])

# Tạo model AI
model = LinearRegression()
model.fit(X, y)

# Dự đoán
print("Dự đoán cho 5:", model.predict([[5]])[0])
