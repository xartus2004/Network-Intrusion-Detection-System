import matplotlib.pyplot as plt
import numpy as np

# Example data for the table
data = [
    [0.987099, 0.93991, 0.998511, 0.998263, 0.999752, 0.999802, 0.998065, 0.896045, 0.970129, 0.942192],
    [0.978964, 0.947807, 0.996626, 0.996626, 0.997222, 0.997222, 0.996229, 0.903354, 0.972614, 0.950982],
]

# Create a figure and axis
fig, ax = plt.subplots()

# Hide axes
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=data, colLabels=["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6", "Model 7", "Train Score", "Test Score"], loc='center')

# Adjust the table size
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Scale to make it more readable

plt.title("Model Performance")
plt.show()
