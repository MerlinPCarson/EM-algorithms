import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])

data = np.genfromtxt('GMM_dataset.txt')
print data[1499]
print data.shape
plt.scatter(data[:,0], data[:,1], s=10)
plt.show()

