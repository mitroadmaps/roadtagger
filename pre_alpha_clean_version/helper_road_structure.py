import mapdriver as md 
import numpy as np
import math
import sys
import scipy.ndimage
import scipy.misc
import cv2
from PIL import Image
import pickle
import bintrees 
from PathSimilarity import PathSimilarity
from rtree import index
from scipy import interpolate
import socket

def Coord2Pixels(lat, lon, min_lat, min_lon, max_lat, max_lon, sizex, sizey):
	#print(max_lat, min_lat, sizex)
	ilat = sizex - int((lat-min_lat) / ((max_lat - min_lat)/sizex))
	#ilat = int((lat-min_lat) / ((max_lat - min_lat)/sizex))
	ilon = int((lon-min_lon) / ((max_lon - min_lon)/sizey))

	return ilat, ilon



def TraceQueryBatch(data, host="localhost", port=8006):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    data_string = ""
    for i in range(len(data)/5):
    	data_string = data_string + str(data[i*5+0])+","+str(data[i*5+1])+","+str(data[i*5+2])+","+str(data[i*5+3])+","+str(data[i*5+4])+","


    #print("DataSentLen", len(data_string))

    s.send(data_string)
    result = s.recv(16384)
    items = result.split()
    result = [int(item) for item in items]
    #print(result)

    return result

def TraceQueryBatch3P(data, host="localhost", port=8002):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    data_string = ""
    for i in range(len(data)/9):
    	data_string = data_string + str(data[i*9+0])+","+str(data[i*9+1])+","+str(data[i*9+2])+","+str(data[i*9+3])+","+str(data[i*9+4])+","+str(data[i*9+5])+","+str(data[i*9+6])+","+str(data[i*9+7])+","+str(data[i*9+8])+","


    #print("DataSentLen", len(data_string))

    s.send(data_string)
    result = s.recv(16384)
    items = result.split()
    result = [int(item) for item in items]
    #print(result)

    return result




def distance(p1, p2):
	a = p1[0] - p2[0]
	b = (p1[1] - p2[1])*math.cos(math.radians(p1[0]))
	return np.sqrt(a*a + b*b)


class RoadGraph:
	def __init__(self, filename=None, region = None):
		self.nodeHash = {} # [tree_idx*10000000 + local_id] ->  id 
		self.nodeHashReverse = {} 
		self.nodes = {}	# id -> [lat,lon]
		self.edges = {} # id -> [n1, n2]
		self.nodeLink = {}   # id -> list of next node
		self.nodeID = 0 
		self.edgeID = 0
		self.edgeHash = {} # [nid1 * 10000000 + nid2] -> edge id 
		self.edgeScore = {}
		self.nodeTerminate = {}
		self.nodeScore = {}
		self.nodeLocations = {}

		if filename is not None:

			dumpDat = pickle.load(open(filename, "rb"))

			forest = dumpDat[1]

			self.forest = forest
			tid = 0
			for t in forest:
				for n in t:
					idthis = tid*10000000 + n['id']

					thislat = n['lat']
					thislon = n['lon']

					if region is not None:
						if thislat < region[0] or thislon < region[1] or thislat > region[2] or thislon > region[3]:
							continue

					#if n['edgeScore'] < 7.0 : # skip those low confidential edges
					#
					#	continue

					if n['similarWith'][0] != -1:
						idthis = n['similarWith'][0]*10000000 + n['similarWith'][1]

						thislat = forest[n['similarWith'][0]][n['similarWith'][1]]['lat']
						thislon = forest[n['similarWith'][0]][n['similarWith'][1]]['lon']

						

					if n['OutRegion'] == 1:
						self.nodeTerminate[tid*10000000+n['parent']] = 1


					idparent = tid*10000000 + n['parent']
					parentlat = t[n['parent']]['lat']
					parentlon = t[n['parent']]['lon']

					if n['parent'] == 0:
						print(tid, n['id'])


					self.addEdge(idparent, parentlat, parentlon, idthis, thislat, thislon)



				tid += 1

		



	def addEdge(self, nid1,lat1,lon1,nid2,lat2,lon2, reverse=False, nodeScore1 = 0, nodeScore2 = 0, edgeScore = 0):  #n1d1->n1d2
		

		if nid1 not in self.nodeHash.keys():
			self.nodeHash[nid1] = self.nodeID
			self.nodeHashReverse[self.nodeID] = nid1
			self.nodes[self.nodeID] = [lat1, lon1]
			self.nodeLink[self.nodeID] = []
			#self.nodeLinkReverse[self.nodeID] = []
			self.nodeScore[self.nodeID] = nodeScore1
			self.nodeID += 1

		if nid2 not in self.nodeHash.keys():
			self.nodeHash[nid2] = self.nodeID
			self.nodeHashReverse[self.nodeID] = nid2
			self.nodes[self.nodeID] = [lat2, lon2]
			self.nodeLink[self.nodeID] = []
			#self.nodeLinkReverse[self.nodeID] = []
			self.nodeScore[self.nodeID] = nodeScore2
			self.nodeID += 1

		localid1 = self.nodeHash[nid1]
		localid2 = self.nodeHash[nid2]

		if localid1 * 10000000 + localid2 in self.edgeHash.keys():
			print("Duplicated Edge !!!", nid1, nid2)

			return 

		self.edges[self.edgeID] = [localid1, localid2]
		self.edgeHash[localid1 * 10000000 + localid2] = self.edgeID
		self.edgeScore[self.edgeID] = edgeScore
		self.edgeID += 1

		if localid2 not in self.nodeLink[localid1]:
			self.nodeLink[localid1].append(localid2)

		if reverse == True:
			if localid2 not in self.nodeLinkReverse.keys():
				self.nodeLinkReverse[localid2] = []

			if localid1 not in self.nodeLinkReverse[localid2]:
				self.nodeLinkReverse[localid2].append(localid1)


	def addEdgeToOneExistedNode(self, nid1,lat1,lon1,nid2, reverse=False, nodeScore1 = 0, edgeScore = 0):  #n1d1->n1d2
		

		if nid1 not in self.nodeHash.keys():
			self.nodeHash[nid1] = self.nodeID
			self.nodeHashReverse[self.nodeID] = nid1
			self.nodes[self.nodeID] = [lat1, lon1]
			self.nodeLink[self.nodeID] = []
			self.nodeLinkReverse[self.nodeID] = []
			self.nodeScore[self.nodeID] = nodeScore1
			self.nodeID += 1

		localid1 = self.nodeHash[nid1]
		localid2 = nid2

		self.edges[self.edgeID] = [localid1, localid2]
		self.edgeHash[localid1 * 10000000 + localid2] = self.edgeID
		self.edgeScore[self.edgeID] = edgeScore
		self.edgeID += 1

		if localid2 not in self.nodeLink[localid1]:
			self.nodeLink[localid1].append(localid2)

		if localid1 not in self.nodeLinkReverse[localid2]:
			self.nodeLinkReverse[localid2].append(localid1)






	def simplfyWithShortestPath(self, sourceID, state=1):
		self.nodeDistance = {}
		self.nodeFrom = {}
		self.nodeSettle = {}



		tree = bintrees.BinaryTree()

		for nid in self.nodes.keys():
			self.nodeDistance[nid] = 10000000000
			self.nodeFrom[nid] = -1
			self.nodeSettle[nid] = 0

		tree[sourceID] = 0
		self.nodeDistance[sourceID] = 0

		#for i in range(len(self.nodes.keys())):
		while True:
			dmin = 10000000
			dminid = -1

			# for j in self.nodes.keys():
			# 	if self.nodeDistance[j] < dmin and self.nodeSettle[j] == 0:
			# 		dmin = self.nodeDistance[j]
			# 		dminid = j

			# if dminid == -1 :
			# 	break
			if len(tree) == 0:
				break

			dminid = tree.min_key()
			dmin = tree[dminid]

			tree.remove_items([dminid])



			#print(dminid, dmin)


			#self.nodeSettle[dminid] = 1


			for nextNode in self.nodeLink[dminid]:
				lat1 = self.nodes[dminid][0]
				lon1 = self.nodes[dminid][1]

				lat2 = self.nodes[nextNode][0]
				lon2 = self.nodes[nextNode][1]


				a = lat1 - lat2
				b = (lon1 - lon2) /  math.cos(math.radians(lat1))

				dist = np.sqrt(a*a + b*b)


				if self.nodeDistance[nextNode] > dmin + dist:
					self.nodeDistance[nextNode] = dmin + dist
					self.nodeFrom[nextNode] = dminid
					tree[nextNode] = dmin + dist

		print("Tree Size", len(tree))

		if state == 1:
			#for i in range(len(self.nodes.keys())):
			for idthis in self.nodeTerminate.keys():
				i = self.nodeHash[idthis]
				if self.nodeFrom[i] != -1:
					cur = i
					path = []
					while True:
						path.append(cur)
						p = self.nodeFrom[cur]

						self.edgeScore[self.edgeHash[p*10000000+cur]] += 1

						self.nodeScore[p] = max(self.nodeScore[p], self.edgeScore[self.edgeHash[p*10000000+cur]])
						self.nodeScore[cur] = max(self.nodeScore[cur], self.edgeScore[self.edgeHash[p*10000000+cur]])


						cur = p

						if cur == sourceID:
							break

					path.reverse() # From root...

					# for j in range(2, len(path)-1, 20):
					# 	self.nodeLocations[path[j]] = 1

					for j in range(len(path)):
						if path[j] % 20 == 0:
							self.nodeLocations[path[j]] = 1

					self.nodeLocations[self.nodeHash[idthis]] = 1

			self.nodeLocations[sourceID] = 1



		if state == 2:
			for idthis in self.nodeLocations.keys():
				i = idthis
				if self.nodeFrom[i] != -1:
					cur = i
					path = []
					while True:
						path.append(cur)
						p = self.nodeFrom[cur]

						self.edgeScore[self.edgeHash[p*10000000+cur]] += 1

						self.nodeScore[p] = max(self.nodeScore[p], self.edgeScore[self.edgeHash[p*10000000+cur]])
						self.nodeScore[cur] = max(self.nodeScore[cur], self.edgeScore[self.edgeHash[p*10000000+cur]])


						cur = p

						if cur == sourceID:
							break



	def mergeTwoNodesBiDirection(self, id1, id2): #remove id2
		if id1 == id2:
			return 

		print("id1, id2", id1, id2)

		print(self.nodeLink[id2])
		for next_node in self.nodeLink[id2]:

			# remove links
			print(next_node, id2, self.nodeLink[next_node], self.nodeLink[id1])
			self.nodeLink[next_node].remove(id2)

			edgeId = self.edgeHash[next_node * 10000000 + id2]
			del self.edges[edgeId]

			edgeId = self.edgeHash[id2 * 10000000 + next_node]
			del self.edges[edgeId]

			#print("Deleted ", self.deletedNodes.keys())
			# Add to id1's nodeLink list
			if (next_node not in self.nodeLink[id1]) and (next_node not in self.deletedNodes.keys()) and (next_node != id1):
				self.nodeLink[id1].append(next_node)
				print(id1,"+",next_node)

				localid1 = id1
				localid2 = next_node

				self.edges[self.edgeID] = [localid1, localid2]
				self.edgeHash[localid1 * 10000000 + localid2] = self.edgeID
				self.edgeScore[self.edgeID] = 10
				self.edgeID += 1

			# Add id1 to next node's nodeLink
			if id1 not in self.nodeLink[next_node] and next_node != id1:
				self.nodeLink[next_node].append(id1)

				localid1 = next_node
				localid2 = id1

				self.edges[self.edgeID] = [localid1, localid2]
				self.edgeHash[localid1 * 10000000 + localid2] = self.edgeID
				self.edgeScore[self.edgeID] = 10
				self.edgeID += 1




		pass

	def mergeTwoNodes(self, id1, id2): #remove id2
		if id1 == id2:
			return 

		self.nodes[id1][0] = (self.nodes[id1][0] + self.nodes[id2][0]) / 2
		self.nodes[id1][1] = (self.nodes[id1][1] + self.nodes[id2][1]) / 2


		print("id1, id2", id1, id2)

		print(self.nodeLink[id2])


		for next_node in self.nodeLink[id2]:
			self.nodeLinkReverse[next_node].remove(id2)

			edgeId = self.edgeHash[id2 * 10000000 + next_node]
			del self.edges[edgeId]

			if (next_node not in self.nodeLink[id1]) and (next_node not in self.deletedNodes.keys()) and (next_node != id1):
				self.nodeLink[id1].append(next_node)

				localid1 = id1
				localid2 = next_node

				self.edges[self.edgeID] = [localid1, localid2] 
				self.edgeHash[localid1 * 10000000 + localid2] = self.edgeID
				self.edgeScore[self.edgeID] = 10
				self.edgeID += 1


			if id1 not in self.nodeLinkReverse[next_node] and next_node != id1:
				self.nodeLinkReverse[next_node].append(id1)


		for next_node in self.nodeLinkReverse[id2]:
			self.nodeLink[next_node].remove(id2)

			edgeId = self.edgeHash[next_node * 10000000 + id2]
			del self.edges[edgeId]

			if (next_node not in self.nodeLinkReverse[id1]) and (next_node not in self.deletedNodes.keys()) and (next_node != id1):
				self.nodeLinkReverse[id1].append(next_node)

				localid1 = next_node
				localid2 = id1

				self.edges[self.edgeID] = [localid1, localid2]
				self.edgeHash[localid1 * 10000000 + localid2] = self.edgeID
				self.edgeScore[self.edgeID] = 10
				self.edgeID += 1

			if id1 not in self.nodeLink[next_node] and next_node != id1:
				self.nodeLink[next_node].append(id1)

		pass

	def getPath(self, id1, length = 6, limit = 40):

		paths = []
		path_tmp = list(range(length))

		def searchPath(idthis, depth, pid = -1):
			#global paths 
			#global path_tmp 
			#global length

			if len(paths) > limit:
				return 

			path_tmp[length - depth] = idthis

			if depth == 1:
				p = map(lambda x:[self.nodes[x][0], self.nodes[x][1], x], path_tmp)
				paths.append(p)
				return 

			for next_node in self.nodeLink[idthis] :
				ok = True


				if next_node == pid :
					continue

				if next_node not in self.nodeScore.keys():
					continue

				if self.nodeScore[next_node] < 1 :
					continue

				if self.edgeScore[self.edgeHash[idthis * 10000000 + next_node]] < 1:
					continue

				searchPath(next_node, depth - 1, pid = idthis)

			pass



		searchPath(id1, length)

		return paths



	def BiDirection(self):
		edgeList = list(self.edges.values())

		for edge in edgeList:
			localid1 = edge[1]
			localid2 = edge[0]

			self.edges[self.edgeID] = [localid1, localid2]
			self.edgeHash[localid1 * 10000000 + localid2] = self.edgeID
			self.edgeScore[self.edgeID] = self.edgeScore[self.edgeHash[localid2 * 10000000 + localid1]]
			self.edgeID += 1

			if localid2 not in self.nodeLink[localid1]:
				self.nodeLink[localid1].append(localid2)

	def ReverseDirectionLink(self):
		edgeList = list(self.edges.values())

		self.nodeLinkReverse = {}

		for edge in edgeList:
			localid1 = edge[1]
			localid2 = edge[0]

			if localid1 not in self.nodeLinkReverse :
				self.nodeLinkReverse[localid1] = [localid2]
			else:
				if localid2 not in self.nodeLinkReverse[localid1]:
					self.nodeLinkReverse[localid1].append(localid2)

		for nodeId in self.nodes.keys():
			if nodeId not in self.nodeLinkReverse.keys():
				self.nodeLinkReverse[nodeId] = []



	def CombineSimilarSnippets(self, region = None, oneround = False):
		idx = index.Index()
		for idthis in self.nodes.keys():
			if idthis in self.nodeScore.keys():
				if self.nodeScore[idthis] > 0:
					if region == None:
						idx.insert(idthis, (self.nodes[idthis][0], self.nodes[idthis][1],self.nodes[idthis][0]+0.000001, self.nodes[idthis][1]+0.000001))
					elif self.nodes[idthis][0] > region[0] and self.nodes[idthis][1] > region[1] and self.nodes[idthis][0] < region[2] and self.nodes[idthis][1] < region[3]:
						idx.insert(idthis, (self.nodes[idthis][0], self.nodes[idthis][1],self.nodes[idthis][0]+0.000001, self.nodes[idthis][1]+0.000001))

		#self.deletedNodes = {}

		while True:

			update_counter = 0

			for idthis in self.nodes.keys():
				if idthis in self.deletedNodes.keys():
					continue

				if idthis in self.nodeScore.keys():
					if self.nodeScore[idthis] > 0:
						

						lat = self.nodes[idthis][0]
						lon = self.nodes[idthis][1]

						if region is not None:
							if lat < region[0] or lon < region[1] or lat > region[2] or lon > region[3]:
								continue



						r = 0.00015
						possible_nodes = list(idx.intersection((lat-r,lon-r, lat+r, lon+r)))

						paths_this = self.getPath(idthis)

						best_p1 = []
						best_p2 = []
						best_d = 10000
						best_err = []

						threshold = 0.00048


						print(idthis, self.nodeScore[idthis], " PossibleNodes ", len(possible_nodes))

						def getDistance(item):
							lat_ = self.nodes[item][0]
							lon_ = self.nodes[item][1]

							a = lat_ - lat
							b = lon_ - lon 

							return np.sqrt(a*a + b*b)

						possible_nodes = sorted(possible_nodes, key = getDistance)

						c = 0
						for possible_node in possible_nodes:
							if possible_node == idthis:
								continue 

							if possible_node in self.deletedNodes.keys():
								continue

							c = c + 1

							if c > 3:
								break

							paths_target = self.getPath(possible_node)

							print("Check "+str(len(paths_this)*len(paths_target)) + " path pairs")

							cc = 0
							for p1 in paths_this:

								if cc > 200:
									break
								for p2 in paths_target:
									flag = True

									cc = cc + 1

									d,err = PathSimilarity(p1,p2)

									if d < best_d and d < threshold:
										best_d = d
										best_err = err
										best_p1 = p1
										best_p2 = p2


						
						if best_d < threshold:
							print(idthis, best_d, best_err,update_counter)
							print(best_p1)
							print(best_p2)
							print(" ")
							update_counter = update_counter + 1
							for i in range(len(p1)):

								# if distance(self.nodes[best_p1[i]], self.nodes[best_p2[i]]) > 0.00020:
								# 	continue

								if best_p1[i][2] in self.deletedNodes.keys() and best_p2[i][2] in self.deletedNodes.keys():
									continue

								if best_p1[i][2] in self.deletedNodes.keys() and best_p2[i][2] not in self.deletedNodes.keys():
									self.mergeTwoNodes(self.deletedNodes[best_p1[i][2]], best_p2[i][2])
									if self.deletedNodes[best_p1[i][2]] != best_p2[i][2]:
										self.deletedNodes[best_p2[i][2]] = self.deletedNodes[best_p1[i][2]]
								else:
									self.mergeTwoNodes(best_p1[i][2], best_p2[i][2])
									if best_p2[i][2] != best_p1[i][2]:
										self.deletedNodes[best_p2[i][2]] = best_p1[i][2]
								
								#print("Deleted ", self.deletedNodes)
								

			print(update_counter," Update(s)")
			#break

			if update_counter == 0 or oneround == True:
				break

		return update_counter
			

	def Smoothen(self, n = 3):

		self.edgeInt = {}

		 

		for edgeId, edge in self.edges.iteritems():
			n1 = edge[0]
			n2 = edge[1]

			#self.edgeInt[edgeId] = None
			#continue

			if n1 not in self.nodeLinkReverse:
				self.edgeInt[edgeId] = None
				continue


			n0 = -1

			for nn in self.nodeLinkReverse[n1]:
				if nn not in self.deletedNodes.keys() and self.nodeScore[nn] > 0 and self.edgeScore[self.edgeHash[nn * 10000000 + n1]] > 0:
					n0 = nn 


			if n0 == -1 :
				self.edgeInt[edgeId] = None
				continue



			paths = self.getPath(n0, length = 4)


			if len(paths) == 0:
				self.edgeInt[edgeId] = None
				continue

			path = None

			for p in paths :
				if p[1][2] == n1 and p[2][2] == n2:
					path = p


			if path is None:
				self.edgeInt[edgeId] = None
				continue




			X = map(lambda x: x[0], path)
			Y = map(lambda x: x[1], path)

			tck1, u = interpolate.splprep([X, Y], s=0, k = 3)


			interval = (u[2] - u[1])/(n+1)
			TT = np.arange(u[1]+interval,u[2]-0.000000001,interval)

			pathInt_ = interpolate.splev(TT, tck1)

			pathInt = map(lambda x: [pathInt_[0][x], pathInt_[1][x]], range(len(pathInt_[0])))

			self.edgeInt[edgeId] = pathInt


		pass

	# DFS
	def TOPOWalkDFS(self, nodeid, step = 0.00005, r = 0.00300, direction = False):

		localNodeList = {}
		localNodeDistance = {}

		mables = []

		localEdges = {}


		#localNodeList[nodeid] = 1
		#localNodeDistance[nodeid] = 0

		def explore(node_cur, node_prev, dist):
			old_node_dist = 1
			if node_cur in localNodeList.keys():
				old_node_dist = localNodeDistance[node_cur]
				if localNodeDistance[node_cur] <= dist:
					return

			if dist > r :
				return

				  

			lat1 = self.nodes[node_cur][0]
			lon1 = self.nodes[node_cur][1]

			localNodeList[node_cur] = 1
			localNodeDistance[node_cur] = dist
			
			#mables.append((lat1, lon1))

			if node_cur not in self.nodeLinkReverse.keys():
				self.nodeLinkReverse[node_cur] = []

			reverseList = []

			if direction == False:
				reverseList = self.nodeLinkReverse[node_cur]

			for next_node in self.nodeLink[node_cur] + reverseList:

				edgeS = 0

				if node_cur * 10000000 + next_node in self.edgeHash.keys():
					edgeS = self.edgeScore[self.edgeHash[node_cur * 10000000 + next_node]]
				
				if next_node * 10000000 + node_cur in self.edgeHash.keys():
					edgeS = max(edgeS, self.edgeScore[self.edgeHash[next_node * 10000000 + node_cur]])


				if self.nodeScore[next_node] > 0 and edgeS > 0:
					pass
				else:
					continue

				if next_node == node_prev :
					continue

				lat0 = 0
				lon0 = 0

				lat1 = self.nodes[node_cur][0]
				lon1 = self.nodes[node_cur][1]

				lat2 = self.nodes[next_node][0]
				lon2 = self.nodes[next_node][1]

				#TODO check angle of next_node


				localEdgeId = node_cur * 10000000 + next_node

				# if localEdgeId not in localEdges.keys():
				# 	localEdges[localEdgeId] = 1

				l = distance((lat2,lon2), (lat1,lon1))
				num = int(math.ceil(l / step))


				bias = step * math.ceil(dist / step) - dist
				cur = bias



				if old_node_dist + l < r :
					explore(next_node, node_cur, dist + l)
				else:

					while cur < l:
						alpha = cur / l 
				#for a in range(1,num):
				#	alpha = float(a)/num 
						if dist + l * alpha > r :
							break

						latI = lat2 * alpha + lat1 * (1-alpha)
						lonI = lon2 * alpha + lon1 * (1-alpha)

						if (latI, lonI) not in mables:
							mables.append((latI, lonI))

						cur += step

					l = distance((lat2,lon2), (lat1,lon1))

					explore(next_node, node_cur, dist + l)



		explore(nodeid, -1, 0)


		return mables


	def distanceBetweenTwoLocation(self, loc1, loc2, max_distance):
		localNodeList = {}
		localNodeDistance = {}

		#mables = []

		localEdges = {}


		edge_covered = {}  # (s,e) --> distance from s and distance from e 


		if loc1[0] == loc2[0] and loc1[1] == loc2[1] :
			return abs(loc1[2] - loc2[2])

		elif loc1[0] == loc2[1] and loc1[1] == loc2[0]:
			return abs(loc1[2] - loc2[3])

		ans_dist = 100000

		Queue = [(loc1[0], -1, loc1[2]), (loc1[1], -1, loc1[2])]

		while True:

			if len(Queue) == 0:
				break

			args = Queue.pop(0)

			node_cur, node_prev, dist = args[0], args[1], args[2]

			old_node_dist = 1
			if node_cur in localNodeList.keys():
				old_node_dist = localNodeDistance[node_cur]
				if localNodeDistance[node_cur] <= dist:
					continue

			if dist > max_distance :
				continue

			lat1 = self.nodes[node_cur][0]
			lon1 = self.nodes[node_cur][1]

			localNodeList[node_cur] = 1
			localNodeDistance[node_cur] = dist
			
			#mables.append((lat1, lon1))

			if node_cur not in self.nodeLinkReverse.keys():
				self.nodeLinkReverse[node_cur] = []

			reverseList = []
			reverseList = self.nodeLinkReverse[node_cur]

			visited_next_node = []
			for next_node in self.nodeLink[node_cur] + reverseList:
				if next_node == node_prev:
					continue

				if next_node == node_cur :
					continue

				if next_node == loc1[0] or next_node == loc1[1] :
					continue

				if next_node in visited_next_node:
					continue 

				visited_next_node.append(next_node)



				edgeS = 0

				

				lat0 = 0
				lon0 = 0

				lat1 = self.nodes[node_cur][0]
				lon1 = self.nodes[node_cur][1]

				lat2 = self.nodes[next_node][0]
				lon2 = self.nodes[next_node][1]

				localEdgeId = node_cur * 10000000 + next_node

				# if localEdgeId not in localEdges.keys():
				# 	localEdges[localEdgeId] = 1


				if node_cur == loc2[0] and next_node == loc2[1]:
					new_ans = dist + loc2[2]
					if new_ans < ans_dist :
						ans_dist = new_ans 
				elif node_cur == loc2[1] and next_node == loc2[0]:
					new_ans = dist + loc2[3]
					if new_ans < ans_dist :
						ans_dist = new_ans




				l = distance((lat2,lon2), (lat1,lon1))
				Queue.append((next_node, node_cur, dist + l))
				
				


		


		return ans_dist


	# BFS (much faster)
	def TOPOWalk(self, nodeid, step = 0.00005, r = 0.00300, direction = False, newstyle = False, nid1=0, nid2=0, dist1=0, dist2= 0, bidirection = False, CheckGPS = None, metaData = None):

		localNodeList = {}
		localNodeDistance = {}

		mables = []

		localEdges = {}


		edge_covered = {}  # (s,e) --> distance from s and distance from e 


		#localNodeList[nodeid] = 1
		#localNodeDistance[nodeid] = 0

		if newstyle == False:
			Queue = [(nodeid, -1, 0)]

		else:
			Queue = [(nid1, -1, dist1), (nid2, -1, dist2)]


		# Add holes between nid1 and nid2 


		lat1 = self.nodes[nid1][0]
		lon1 = self.nodes[nid1][1]

		lat2 = self.nodes[nid2][0]
		lon2 = self.nodes[nid2][1]

		l = distance((lat2,lon2), (lat1,lon1))
		num = int(math.ceil(l / step))

		alpha = 0 

		while True:
			latI = lat1*alpha + lat2*(1-alpha)
			lonI = lon1*alpha + lon2*(1-alpha)

			d1 = distance((latI,lonI),(lat1,lon1))
			d2 = distance((latI,lonI),(lat2,lon2))

			if dist1 - d1 < r or dist2 -d2 < r:
				if (latI, lonI, lat2 - lat1, lon2 - lon1) not in mables:
					mables.append((latI, lonI, lat2 - lat1, lon2 - lon1)) # add direction

					if bidirection == True:
						if nid1 in self.nodeLink[nid2] and nid2 in self.nodeLink[nid1]:
							mables.append((latI+0.00001, lonI+0.00001, lat2 - lat1, lon2 - lon1))  #Add another mables

			alpha += step/l

			if alpha > 1.0:
				break




		while True:

			if len(Queue) == 0:
				break

			args = Queue.pop(0)

			node_cur, node_prev, dist = args[0], args[1], args[2]

			old_node_dist = 1
			if node_cur in localNodeList.keys():
				old_node_dist = localNodeDistance[node_cur]
				if localNodeDistance[node_cur] <= dist:
					continue

			if dist > r :
				continue

				  

			lat1 = self.nodes[node_cur][0]
			lon1 = self.nodes[node_cur][1]

			localNodeList[node_cur] = 1
			localNodeDistance[node_cur] = dist
			
			#mables.append((lat1, lon1))

			if node_cur not in self.nodeLinkReverse.keys():
				self.nodeLinkReverse[node_cur] = []

			reverseList = []

			if direction == False:
				reverseList = self.nodeLinkReverse[node_cur]

			visited_next_node = []
			for next_node in self.nodeLink[node_cur] + reverseList:
				if next_node == node_prev:
					continue

				if next_node == node_cur :
					continue

				if next_node == nid1 or next_node == nid2 :
					continue

				if next_node in visited_next_node:
					continue 




				visited_next_node.append(next_node)



				edgeS = 0

				# if node_cur * 10000000 + next_node in self.edgeHash.keys():
				# 	edgeS = self.edgeScore[self.edgeHash[node_cur * 10000000 + next_node]]
				
				# if next_node * 10000000 + node_cur in self.edgeHash.keys():
				# 	edgeS = max(edgeS, self.edgeScore[self.edgeHash[next_node * 10000000 + node_cur]])


				# if self.nodeScore[next_node] > 0 and edgeS > 0:
				# 	pass
				# else:
				# 	continue

				# if next_node == node_prev :
				# 	continue

				lat0 = 0
				lon0 = 0

				lat1 = self.nodes[node_cur][0]
				lon1 = self.nodes[node_cur][1]

				lat2 = self.nodes[next_node][0]
				lon2 = self.nodes[next_node][1]

				#TODO check angle of next_node


				localEdgeId = node_cur * 10000000 + next_node

				# if localEdgeId not in localEdges.keys():
				# 	localEdges[localEdgeId] = 1

				l = distance((lat2,lon2), (lat1,lon1))
				num = int(math.ceil(l / step))


				bias = step * math.ceil(dist / step) - dist
				cur = bias



				if old_node_dist + l < r :
					Queue.append((next_node, node_cur, dist + l))
					#explore(next_node, node_cur, dist + l)
				else:

					start_limitation = 0
					end_limitation = l 
					if (node_cur, next_node) in edge_covered.keys():
						start_limitation = edge_covered[(node_cur, next_node)]

					#if next_node == node_cur :
						#print("BUG")

					if (next_node, node_cur) in edge_covered.keys():
						end_limitation = l-edge_covered[(next_node, node_cur)]

					#end_limitation = l

					#if next_node not in localNodeDistance.keys(): # Should we remove this ?


					turnnel_edge = False
					if metaData is not None:
						nnn1 = self.nodeHashReverse[next_node]
						nnn2 = self.nodeHashReverse[node_cur]

						if metaData.edgeProperty[metaData.edge2edgeid[(nnn1,nnn2)]]['layer'] < 0:
							turnnel_edge  = True
							



					while cur < l:
						alpha = cur / l 
			
						if dist + l * alpha > r :
							break

						if l * alpha < start_limitation:
							cur += step
							continue 

						if l * alpha > end_limitation:
							break

						latI = lat2 * alpha + lat1 * (1-alpha)
						lonI = lon2 * alpha + lon1 * (1-alpha)


						
						if (latI, lonI, lat2 - lat1, lon2 - lon1) not in mables and turnnel_edge is False:
							mables.append((latI, lonI, lat2 - lat1, lon2 - lon1)) # add direction


							if bidirection == True:
								if next_node in self.nodeLink[node_cur] and node_cur in self.nodeLink[next_node] and turnnel_edge is False:
									mables.append((latI+0.00001, lonI+0.00001, lat2 - lat1, lon2 - lon1))  #Add another mables


						cur += step


					if (node_cur, next_node) in edge_covered.keys():
						#if cur-step < edge_covered[(node_cur, next_node)]:
						#	print(node_cur, edge_covered[(node_cur, next_node)], cur-step)

						edge_covered[(node_cur, next_node)] = cur - step #max(cur, edge_covered[(node_cur, next_node)])
						#edge_covered[(node_cur, next_node)] = cur
					else:
						edge_covered[(node_cur, next_node)] = cur - step
						#edge_covered[(node_cur, next_node)] = cur




					l = distance((lat2,lon2), (lat1,lon1))
					Queue.append((next_node, node_cur, dist + l))
					#explore(next_node, node_cur, dist + l)


		result_marbles = []

		if CheckGPS is None:
			result_marbles = mables
		else:
			for mable in mables:
				if CheckGPS(mable[0], mable[1]) == True:
					result_marbles.append(mable)



		#explore(nodeid, -1, 0)


		return result_marbles


	def removeNode(self, nodeid):
		for next_node in self.nodeLink[nodeid]:
			edgeid = self.edgeHash[nodeid * 10000000 + next_node]

			del self.edges[edgeid]
			del self.edgeScore[edgeid]
			del self.edgeHash[nodeid * 10000000 + next_node]

			if nodeid in self.nodeLinkReverse[next_node]:
				self.nodeLinkReverse[next_node].remove(nodeid)


		for prev_node in self.nodeLinkReverse[nodeid]:
			edgeid = self.edgeHash[prev_node * 10000000 + nodeid]

			del self.edges[edgeid]
			del self.edgeScore[edgeid]
			del self.edgeHash[prev_node * 10000000 + nodeid]

			if nodeid in self.nodeLink[prev_node]:
				self.nodeLink[prev_node].remove(nodeid)


		del self.nodes[nodeid]
		del self.nodeScore[nodeid]
		del self.nodeLink[nodeid]
		del self.nodeLinkReverse[nodeid]



	def removeDeadEnds(self, oneround = False):
		deleted = 0
		for nodeid in self.nodes.keys():
			if self.nodeHashReverse[nodeid] in self.nodeTerminate.keys():
				continue

			if self.nodeHashReverse[nodeid] % 10000000 == 0:
				continue

			d = self.NumOfNeighbors(nodeid)
			if d == 1 or len(self.nodeLink[nodeid]) == 0 or len(self.nodeLinkReverse[nodeid]) == 0:
				self.removeNode(nodeid)
				deleted += 1

		return deleted 




	def setScoreThreshold(self, threshold = 1):
		self.scoreThreshold = threshold


	def NumOfNeighbors(self, nodeid):
		neighbor = {}

		for next_node in self.nodeLink[nodeid] + self.nodeLinkReverse[nodeid]:
			neighbor[next_node] = 1

		return len(neighbor.keys())

	def getNeighbors(self,nodeid):
		neighbor = {}

		for next_node in self.nodeLink[nodeid] + self.nodeLinkReverse[nodeid]:
			if next_node != nodeid:
				neighbor[next_node] = 1

		return neighbor.keys()



def edgeIntersection(baseX, baseY, dX, dY, n1X, n1Y, n2X, n2Y):
	t = dX * n1Y + dY * n2X - dX * n2Y - dY * n1X

	c = n2X * n1Y - n1X * n2Y + baseX * (n2Y - n1Y) + baseY * (n1X -n2X)

	if t == 0 :
		return 0,0,0,0

	alpha = c / t

	if alpha < 0 : 
		return 0,0,0,0

	iX = baseX + alpha * dX
	iY = baseY + alpha * dY

	d = (iX - n1X)*(n2X - iX) + (iY - n1Y) * (n2Y - iY)

	if d < 0 :
		return 0,0,0,0

	extend_length = np.sqrt(alpha * dX * alpha * dX + alpha * dY * alpha * dY)

	return iX, iY, extend_length, 1




class RoadGraphCombiner:
	def __init__(self, RoadGraphA, RoadGraphB, ConnectivityChecker = None, CNNOutput = None, region = None):

		if CNNOutput is not None:
			cnn_dat = scipy.ndimage.imread("skeleton_all.png") * 255
			


		# Add to index
		idx = index.Index()
		for edgeId, edge in RoadGraphA.edges.iteritems():
			n1 = edge[0]
			n2 = edge[1]

		
			if n1 in RoadGraphA.deletedNodes.keys() or n2 in RoadGraphA.deletedNodes.keys():
				continue

			if RoadGraphA.nodeScore[n1] < 1 or RoadGraphA.nodeScore[n2] < 1 :
				continue

			if n1 in RoadGraphA.nodeTerminate.keys() or n2 in RoadGraphA.nodeTerminate.keys():
				continue

			score = RoadGraphA.edgeScore[RoadGraphA.edgeHash[n1*10000000 + n2]]
			if score <1:
				continue


			lat1 = RoadGraphA.nodes[n1][0]
			lon1 = RoadGraphA.nodes[n1][1]
			lat2 = RoadGraphA.nodes[n2][0]
			lon2 = RoadGraphA.nodes[n2][1]


			idx.insert(edgeId, (min(lat1, lat2), min(lon1, lon2), max(lat1, lat2), max(lon1, lon2)))


		# Find Candidate Edge
		DeadEnds = {}

		for edgeId, edge in RoadGraphB.edges.iteritems():
			n1 = edge[0]
			n2 = edge[1]

			nn1 = RoadGraphB.NumOfNeighbors(n1)
			nn2 = RoadGraphB.NumOfNeighbors(n2)

			if nn1 == 1 or nn2 == 1 :
				DeadEnds[edgeId] = 1

		# Extend Edge 

		newNodeId = 0

		print("DeadEnds ", len(DeadEnds.keys()))

		for edgeId in DeadEnds.keys():
			edge = RoadGraphB.edges[edgeId]
			n1 = edge[0]
			n2 = edge[1]

			nn1 = RoadGraphB.NumOfNeighbors(n1)
			nn2 = RoadGraphB.NumOfNeighbors(n2)

			lat1 = RoadGraphB.nodes[n1][0]
			lon1 = RoadGraphB.nodes[n1][1]

			lat2 = RoadGraphB.nodes[n2][0]
			lon2 = RoadGraphB.nodes[n2][1]


			# if lat1 < 42.3519 or lat1 > 42.3631 or lon1 < -71.1277 or lon1 > -71.1127:
			# 	continue




			avglat = (lat1+lat2)/2
			avglon = (lon1+lon2)/2




			if nn1 == 1 :
				dlat = lat1 - lat2
				dlon = lon1 - lon2

				lat = lat1
				lon = lon1
				r = 0.00100 # 100 meters

				possibleEdges = list(idx.intersection((lat-r,lon-r, lat+r, lon+r)))

				# if len(possibleEdges) > 0:
				# 	print(len(possibleEdges))

				min_dist = 0.00050
				newlat = 0
				newlon = 0
				old_n1 = 0
				old_n2 = 0


				for possibleEdge in possibleEdges:
					pn1 = RoadGraphA.edges[possibleEdge][0]
					pn2 = RoadGraphA.edges[possibleEdge][1]

					pn1lat = RoadGraphA.nodes[pn1][0]
					pn1lon = RoadGraphA.nodes[pn1][1]

					pn2lat = RoadGraphA.nodes[pn2][0]
					pn2lon = RoadGraphA.nodes[pn2][1]


					pn3 = -1 
					pn0 = -1

					for next_node in RoadGraphA.nodeLink[pn2]:
						if next_node != pn1 :
							pn3 = next_node

					for prev_node in RoadGraphA.nodeLinkReverse[pn1]:
						if prev_node != pn2 :
							pn0 = prev_node

					if pn0 == -1 or pn3 == -1 :
						continue

					pn0lat = RoadGraphA.nodes[pn0][0]
					pn0lon = RoadGraphA.nodes[pn0][1]

					pn3lat = RoadGraphA.nodes[pn3][0]
					pn3lon = RoadGraphA.nodes[pn3][1]




					newlat_, newlon_, dist, ok = edgeIntersection(lat, lon, dlat, dlon, pn1lat, pn1lon, pn2lat, pn2lon)

					if dist < min_dist and ok == 1:
						min_dist = dist

						newlat = newlat_
						newlon = newlon_

						old_n1 = pn1
						old_n2 = pn2

						if self.CNNConnectivityChecker(lat, lon, newlat, newlon, cnn_dat, region) == True:
							if self.ConnectivityChecker(avglat, avglon, pn1lat, pn1lon, pn0lat, pn0lon)==True or self.ConnectivityChecker(avglat, avglon, pn2lat, pn2lon, pn3lat, pn3lon)==True:

								RoadGraphA.addEdgeToOneExistedNode("Tmp"+str(newNodeId), newlat, newlon, old_n1, reverse=True, nodeScore1 = 100, edgeScore = 100)
								RoadGraphA.addEdgeToOneExistedNode("Tmp"+str(newNodeId), newlat, newlon, old_n2, reverse=True, nodeScore1 = 100, edgeScore = 100)

								RoadGraphA.addEdge("Tmp"+str(newNodeId), newlat, newlon, "CNN"+str(n1), lat, lon, reverse=True, nodeScore1 = 100, nodeScore2 = 100, edgeScore = 100)

								newNodeId += 1
								print(newNodeId)



				# TODO
				# print(min_dist)

				# if min_dist < 0.00050: # 50 meters
				# 	print(newlat, newlon)
				# 	RoadGraphA.addEdgeToOneExistedNode("Tmp"+str(newNodeId), newlat, newlon, old_n1, reverse=True, nodeScore1 = 100, edgeScore = 100)
				# 	RoadGraphA.addEdgeToOneExistedNode("Tmp"+str(newNodeId), newlat, newlon, old_n2, reverse=True, nodeScore1 = 100, edgeScore = 100)

				# 	RoadGraphA.addEdge("Tmp"+str(newNodeId), newlat, newlon, "CNN"+str(n1), lat, lon, reverse=True, nodeScore1 = 100, nodeScore2 = 100, edgeScore = 100)

				# 	newNodeId += 1


			if nn2 == 1 :
				dlat = lat2 - lat1
				dlon = lon2 - lon1

				lat = lat2
				lon = lon2
				r = 0.00100 # 100 meters

				possibleEdges = list(idx.intersection((lat-r,lon-r, lat+r, lon+r)))

				# if len(possibleEdges) > 0:
				# 	print(len(possibleEdges))


				min_dist = 0.00050
				newlat = 0
				newlon = 0
				old_n1 = 0
				old_n2 = 0


				for possibleEdge in possibleEdges:
					pn1 = RoadGraphA.edges[possibleEdge][0]
					pn2 = RoadGraphA.edges[possibleEdge][1]

					pn1lat = RoadGraphA.nodes[pn1][0]
					pn1lon = RoadGraphA.nodes[pn1][1]

					pn2lat = RoadGraphA.nodes[pn2][0]
					pn2lon = RoadGraphA.nodes[pn2][1]

					pn3 = -1 
					pn0 = -1

					for next_node in RoadGraphA.nodeLink[pn2]:
						if next_node != pn1 :
							pn3 = next_node

					for prev_node in RoadGraphA.nodeLinkReverse[pn1]:
						if prev_node != pn2 :
							pn0 = prev_node

					if pn0 == -1 or pn3 == -1 :
						continue

					pn0lat = RoadGraphA.nodes[pn0][0]
					pn0lon = RoadGraphA.nodes[pn0][1]

					pn3lat = RoadGraphA.nodes[pn3][0]
					pn3lon = RoadGraphA.nodes[pn3][1]




					newlat_, newlon_, dist, ok = edgeIntersection(lat, lon, dlat, dlon, pn1lat, pn1lon, pn2lat, pn2lon)

					if dist < min_dist and ok == 1:
						min_dist = dist

						newlat = newlat_
						newlon = newlon_

						old_n1 = pn1
						old_n2 = pn2

						if self.CNNConnectivityChecker(lat, lon, newlat, newlon, cnn_dat, region) == True:
							if self.ConnectivityChecker(avglat, avglon, pn1lat, pn1lon, pn0lat, pn0lon)==True or self.ConnectivityChecker(avglat, avglon, pn2lat, pn2lon, pn3lat, pn3lon)==True:
								RoadGraphA.addEdgeToOneExistedNode("Tmp"+str(newNodeId), newlat, newlon, old_n1, reverse=True, nodeScore1 = 100, edgeScore = 100)
								RoadGraphA.addEdgeToOneExistedNode("Tmp"+str(newNodeId), newlat, newlon, old_n2, reverse=True, nodeScore1 = 100, edgeScore = 100)

								RoadGraphA.addEdge("Tmp"+str(newNodeId), newlat, newlon, "CNN"+str(n2), lat, lon, reverse=True, nodeScore1 = 100, nodeScore2 = 100, edgeScore = 100)

								newNodeId += 1
								print("NodeID",newNodeId)


				# TODO

				#print(min_dist)

				# if min_dist < 0.00050: # 50 meters
				# 	print(newlat, newlon)
				# 	RoadGraphA.addEdgeToOneExistedNode("Tmp"+str(newNodeId), newlat, newlon, old_n1, reverse=True, nodeScore1 = 100, edgeScore = 100)
				# 	RoadGraphA.addEdgeToOneExistedNode("Tmp"+str(newNodeId), newlat, newlon, old_n2, reverse=True, nodeScore1 = 100, edgeScore = 100)

				# 	RoadGraphA.addEdge("Tmp"+str(newNodeId), newlat, newlon, "CNN"+str(n2), lat, lon, reverse=True, nodeScore1 = 100, nodeScore2 = 100, edgeScore = 100)

				# 	newNodeId += 1

		# Add GraphB to GraphA 
		for edgeId, edge in RoadGraphB.edges.iteritems():
			n1 = edge[0]
			n2 = edge[1]

			lat1 = RoadGraphB.nodes[n1][0]
			lon1 = RoadGraphB.nodes[n1][1]

			lat2 = RoadGraphB.nodes[n2][0]
			lon2 = RoadGraphB.nodes[n2][1]

			RoadGraphA.addEdge("CNN"+str(n1), lat1, lon1, "CNN"+str(n2), lat2, lon2, reverse=True,  nodeScore1 = 100, nodeScore2 = 100, edgeScore = 100)

		pass

	def ConnectivityChecker(self, lat1, lon1, lat2, lon2, lat3, lon3,  radius = 0.00005):

		data = [lat1, lon1, lat2, lon2, radius]

		result = TraceQueryBatch(data)

		gpsAt1 = result[0]
		gpsAt2 = result[1]

		validateTrips = result[3] + result[4]

		print(result)

		#validateTrips = 0

		if gpsAt1 < 5 :
			return True

		if validateTrips < 3:
			return False

		data = [lat1, lon1, lat2, lon2, lat3, lon3, radius, radius, radius]
		validateTrips = max(validateTrips, TraceQueryBatch3P(data))


		data = [lat3, lon3, lat2, lon2, lat1, lon1, radius, radius, radius]
		validateTrips = max(validateTrips, TraceQueryBatch3P(data))


		print("validateTrips ", validateTrips)

		if validateTrips > 2 :
			return True

		return False

	def CNNConnectivityChecker(self, lat1, lon1, lat2, lon2, img, region, l = 80, check_range = 12):
		min_lat = region[0]
		min_lon = region[1]
		max_lat = region[2]
		max_lon = region[3]

		sizex = np.shape(img)[0]
		sizey = np.shape(img)[1]


		ilat, ilon = Coord2Pixels(lat1, lon1, min_lat, min_lon, max_lat, max_lon, sizex, sizey)
		ilatT, ilonT = Coord2Pixels(lat2, lon2, min_lat, min_lon, max_lat, max_lon, sizex, sizey)

		n = []

		n.append((ilat, ilon))
		n.append((ilat+1, ilon))
		n.append((ilat-1, ilon))
		n.append((ilat, ilon+1))
		n.append((ilat, ilon-1))
		n.append((ilat+1, ilon+1))
		n.append((ilat+1, ilon-1))
		n.append((ilat-1, ilon+1))
		n.append((ilat-1, ilon-1))


		visited = []
		mark = []

		flag = False

		while True:
			if len(n) == 0:
				break

			loc = n.pop(0)

			if loc in visited:
				continue

			if (loc[0] - ilat) * (loc[0] - ilat) + (loc[1] - ilon) * (loc[1] - ilon) > l * l :
				continue 

			if img[loc[0], loc[1]] == 0:
				continue

			visited.append(loc)

			n.append((loc[0]+1, loc[1]))
			n.append((loc[0]-1, loc[1]))
			n.append((loc[0], loc[1]+1))
			n.append((loc[0], loc[1]-1))
			n.append((loc[0]+1, loc[1]+1))
			n.append((loc[0]-1, loc[1]+1))
			n.append((loc[0]+1, loc[1]-1))
			n.append((loc[0]-1, loc[1]-1))


			if (loc[0] - ilatT) * (loc[0] - ilatT) + (loc[1] - ilonT) * (loc[1] - ilonT) <= check_range * check_range :
				flag = True
				break


		return flag



if __name__ == "__main__":

	dumpDat = pickle.load(open(sys.argv[1], "rb"))
