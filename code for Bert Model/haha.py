import seaborn as sns
import matplotlib.pyplot as plt

data = [[498, 27], [27, 498]]
labels = ["Negative", "Positive"]

sns.heatmap(data, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.title("Estimated Confusion Matrix")
plt.xlabel("Predicted Label"); plt.ylabel("Actual Label")
plt.show()