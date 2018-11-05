import os, cv2, imutils, sys, heapq, glob
from sklearn import svm
from sklearn.cluster import MeanShift
from skimage.measure import compare_ssim
from sklearn import svm
import random
import numpy as np

def get_nearest_neighbors_func(candidate_images_path, query_image_path, query_image_name, num_nn):
	# cv_query_image = cv2.imread(query_image_path)
	# cv_query_image_resized = cv2.resize(cv_query_image, (100,100), interpolation=cv2.INTER_AREA)
	cv_query_image_resized = cv2.imread(query_image_path)

	image_diff = list()
	heapq.heapify(image_diff)

	for filename in glob.glob(os.path.join(candidate_images_path, '*.jpg')):

		curr_file = filename.strip().split("/")[-1]	
		if curr_file == query_image_name:
			continue

		curr_image = cv2.imread(filename)

		(score, diff) = compare_ssim(cv_query_image_resized, curr_image, full=True, multichannel=True)
		heapq.heappush(image_diff, (score, filename.strip().split("/")[-1], curr_image))
		print("COMPARED TO - " + curr_file)

	return (heapq.nlargest(num_nn, image_diff), cv_query_image_resized)

def get_clusters(nearest_neighbors, candidate_images_path, query_image_path, query_image_name, gps_data_file):
	
	image_to_gps_dict = dict()

	with open(gps_data_file, "r") as fp:
		for line in fp:
			data = line.strip().split()
			if image_to_gps_dict.get(data[0], None)==None:
				image_to_gps_dict[data[0]] = [float(data[1]), float(data[2])]

	training_set = list()
	centers = list()

	for tup in nearest_neighbors:
		training_set.append(image_to_gps_dict[tup[1]])

	bandwidth = 0.05
	training_set = np.array(training_set)

	ms = MeanShift(bandwidth=bandwidth,bin_seeding=False)
	ms.fit(training_set)
	centers = ms.cluster_centers_
	labels = ms.labels_
	
	return (nearest_neighbors, centers, labels)

def get_histogram_of_neighbors(nearest_neighbors, query_image_cv, candidate_images_path):

	histograms_of_neighbors = list()

	for i in range(len(nearest_neighbors)):
		image = nearest_neighbors[i][2]
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
		hist = cv2.normalize(hist).flatten()
		histograms_of_neighbors.append(hist.tolist())

	query_image = cv2.cvtColor(query_image_cv, cv2.COLOR_BGR2RGB)
	hist_query = cv2.calcHist([query_image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
	hist_query = cv2.normalize(hist_query).flatten()

	return (histograms_of_neighbors, hist_query)

def train_svm_and_predict(histograms_of_neighbors, hist_query, centers, labels):

	if len(centers)==1:
		return centers[0]

	clf = svm.SVC(decision_function_shape='ovr')
	clf.fit(histograms_of_neighbors, labels)
	dec = clf.decision_function([hist_query])
	print "DEC: ", dec, dec.argmax()
	return centers[dec.argmax()]
