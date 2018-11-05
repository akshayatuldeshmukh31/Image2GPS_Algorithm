import imutils
import cv2
import glob, os, sys

# Path where images are located
src_path = sys.argv[1]

# Path where resized images will be placed
dst_path = sys.argv[2]

# Resized width
width = int(sys.argv[3])

# Resized height
height = int(sys.argv[4])

# Extension of image file
extension = sys.argv[5]

count = 0

for filename in glob.glob(os.path.join(src_path, '*.' + extension)):
	tmp_image = cv2.imread(filename)
	resized_tmp_image = cv2.resize(tmp_image, (width, height), interpolation=cv2.INTER_AREA)
	cv2.imwrite(dst_path + "/" + filename.strip().split("/")[-1], resized_tmp_image)
	count += 1
	print("RESIZED " + str(count)  + " - " + filename.strip().split("/")[-1])
