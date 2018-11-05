import glob, os, sys

fp = open("../GSV_Downloaded_spatialdata.txt", "w")

for filename in glob.glob(os.path.join(sys.argv[1], '*.jpg')):
	lat,lng = filename.split("/")[-1].split("_")[:2]
	fp.write(filename.split("/")[-1] + "\t" + lat + "\t" + lng + "\n")

fp.close()
