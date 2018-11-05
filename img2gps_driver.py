import os,sys,glob,random,cv2
from img2gps_helper_funcs import get_nearest_neighbors_func, get_clusters, get_histogram_of_neighbors, train_svm_and_predict

spatial_data_path = "./../GSV_Downloaded_spatialdata.txt"

candidate_images_path = "./../GSV_Downloaded_Resized"
query_image_path = "./../GSV_Downloaded_Resized_Test_Images"
num_nn = 5

file_list = list()

for file in glob.glob(query_image_path + "/*.jpg"):
	file_list.append(file)

# print len(file_list)
# file_list = random.sample(file_list, 1000)

for filename in file_list:
	query_image_path = filename
	query_image_name = query_image_path.strip().split("/")[-1]

	# Get neighbors
	neighbors, query_image_cv = get_nearest_neighbors_func(candidate_images_path, query_image_path, query_image_name, num_nn)

	# Get clusters
	neighbors, centers, labels = get_clusters(neighbors, candidate_images_path, query_image_path, query_image_name, spatial_data_path)
	centers = centers.tolist()
	labels = labels.tolist()

	print centers

	# Get histogram of every neighboring image in a numpy list
	hist_neighbors, hist_query = get_histogram_of_neighbors(neighbors, query_image_cv, candidate_images_path)

	# Return final GPS coordinates after training SVM
	prediction = train_svm_and_predict(hist_neighbors, hist_query, centers, labels)
	print ""
	print "Prediction: ", prediction

	with open("predictions_downloaded_dataset.txt","a+") as fp:
		fp.write(query_image_name + "\t" + str(prediction[0]) + "\t" + str(prediction[1]) + "\n")
