import os,sys
import numpy as np
import matplotlib.pyplot as plt

actual_coords = dict()
distances = list()

with open("predictions_downloaded_dataset.txt","r") as fp:
	for line in fp:
		ls = line.strip().split()
		actual_lat, actual_lng = map(float, ls[0].split("_")[:2])
		distances.append(pow(pow(actual_lat - round(float(ls[1]),3), 2) + pow(actual_lng - round(float(ls[2]),3), 2), 0.5))

distances = np.array(distances)

print "Mean - ", distances.mean()
print "Std - ", distances.std()
print "Min - ", distances.min()
print "Max - ", distances.max()

low_lim = distances.mean() - distances.std()
high_lim = distances.mean() + distances.std()

count = 0
for ele in distances:
	if low_lim <= ele and ele <= high_lim:
		count += 1

print "% within 1 Std - ", (float(count)/float(len(distances)))*100.0

plt.hist(distances, normed=False, bins = 5)
plt.ylabel('Number of instances')
plt.show()
