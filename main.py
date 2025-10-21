# 1 import library
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 2 menyiapkan data
X = [[1], [2], [3], [4]] # two dimensional list / array
y = [101, 102, 103, 104] # one dimensional list / aray

# 3 instansiasi objek linear regression
model = LinearRegression()

# 4 melatih model linear regression
model.fit(X, y)

# 5 print hasil prediksi
print(model.predict([[20]]))

# 5 Visualisasi dengan library Matplotlib
plt.scatter(X, y)
plt.plot(X, y)
plt.scatter(20, model.predict([[20]]))
plt.show()
