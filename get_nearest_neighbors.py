import os, cv2, imutils, sys, heapq, glob
from sklearn import svm
from skimage.measure import compare_ssim

candidate_images_path = sys.argv[1]
query_image_path = sys.argv[2]
query_image_name = query_image_path.strip().split("/")[-1]
num_nn = int(sys.argv[3])

cv_query_image = cv2.imread(query_image_path)
cv_query_image_resized = cv2.resize(cv_query_image, (100,100), interpolation=cv2.INTER_AREA)
gray_query_image = cv2.cvtColor(cv_query_image_resized, cv2.COLOR_BGR2GRAY)

image_diff = list()
heapq.heapify(image_diff)
counter = 0

for filename in glob.glob(os.path.join(candidate_images_path, '*.jpg')):

	curr_file = filename.strip().split("/")[-1]	
	if curr_file == query_image_name:
		continue

	curr_image = cv2.imread(filename)
	gray_curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

	(score, diff) = compare_ssim(gray_query_image, gray_curr_image, full=True)
	heapq.heappush(image_diff, (score, curr_file))
	
	counter += 1
	print(str(counter) + ". COMPARED TO - " + curr_file)

nearest_neighbors = heapq.nlargest(num_nn, image_diff)
print(nearest_neighbors)
