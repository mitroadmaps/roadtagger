import math
import numpy as np
from scipy import interpolate




def distance(p1, p2):
	a = p1[0] - p2[0]
	b = p1[1] - p2[1]
	return np.sqrt(a*a + b*b)


def rawSimilarity(path1, path2):
	vx,vy = 0, 0
	s = 0 

	for i in range(len(path1)):
		s = s + distance(path1[i], path2[i])

		vx = vx + path1[i][0] - path2[i][0]
		vy = vy + path1[i][1] - path2[i][1]

	ErrVector = (vx, vy)

	return s, ErrVector


def PathSimilarity(path1, path2, interpolation=40, subWindowSize = 32, kind = 'quadratic', threshold = 0.00020): #, kind = 'cubic'):
	
	XX = np.arange(1.0/interpolation,1.0,1.0/interpolation)
	#XX = list(np.arange(0,1,0.1))

	path1Lat = map(lambda x:x[0], path1)
	path1Lon = map(lambda x:x[1], path1)

	path2Lat = map(lambda x:x[0], path2)
	path2Lon = map(lambda x:x[1], path2)


	for i in range(len(path1Lat)):
		d = distance([path1Lat[i], path1Lon[i]], [path2Lat[i], path2Lon[i]])
		if d > threshold:
			return 1000, (0,0)

	duplicate_loc = {}
	flag = False
	for i in range(len(path1Lat)-1):
		if (path1Lat[i], path1Lon[i]) in duplicate_loc.keys():
			flag = True
		else:
			duplicate_loc[(path1Lat[i], path1Lon[i])] = 1

	if flag == True:
		return 1000, (0,0)

	duplicate_loc = {}
	flag = False
	for i in range(len(path2Lat)-1):
		if (path2Lat[i], path2Lon[i]) in duplicate_loc.keys():
			flag = True
		else:
			duplicate_loc[(path2Lat[i], path2Lon[i])] = 1


	if flag == True:
		return 1000, (0,0)
		


	tck1, _ = interpolate.splprep([path1Lat, path1Lon], s=0, k = 3)
	tck2, _ = interpolate.splprep([path2Lat, path2Lon], s=0, k = 3)

	path1Int_ = interpolate.splev(XX, tck1)
	path2Int_ = interpolate.splev(XX, tck2)

	path1Int = map(lambda x: [path1Int_[0][x], path1Int_[1][x]], range(len(path1Int_[0])))
	path2Int = map(lambda x: [path2Int_[0][x], path2Int_[1][x]], range(len(path2Int_[0])))


	n = len(path1Int)



	min_dist = 10000
	err_vec = (0,0)


	for s1 in range(0, n - subWindowSize):
		for s2 in range(0, n - subWindowSize):
			dist, vec = rawSimilarity(path1Int[s1:s1+subWindowSize], path2Int[s2:s2+subWindowSize])

			if dist < min_dist:
				min_dist = dist
				err_vec = vec


	return min_dist, err_vec




if __name__ == "__main__":
	path1 = [[1,1],[2,2],[3,2.5],[4,3]]
	path2 = [[0.5,0.5], [1.5,1.5], [2.5,2.25], [3.5,2.75]]
	#path2 = [[0.5,2.5], [1.5,3.5], [2.5,4.25], [3.5,4.75]]
	#path2 = [[0,0],[1,0.5],[2,1.1],[3,2],[4,3],[5,3]]

	dist,err_vec = PathSimilarity(path1, path2, interpolation=20)

	print(dist)
	print(err_vec)