from sklearn.datasets import make_classification
import numpy as np
x, y = make_classification(
        n_samples=2000,
        n_features=649,
        n_informative=30,
        n_redundant=30,
        n_repeated=0,
        n_classes=10,
        n_clusters_per_class=1,
        random_state=1234,
        shuffle=False,
        class_sep=5.0
        )
ma = np.zeros((2000, 650))
print(y[200])
for i in range(650):
    if i == 0:
        for j in range(2000):
            ma[j][i] = y[j]
    else:
        for j in range(2000):
            ma[j][i] = x[j][i - 1]
print(ma[115][0])
import csv
with open('ma.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in ma:
        csvwriter.writerow(row)

