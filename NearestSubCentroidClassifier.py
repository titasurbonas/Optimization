import numpy as np

class CentroidClassifier(object):
	def __init__(self, data, labels, subclass,startclass):
		self.data = data
		self.labels = labels
		self.subclass = subclass
		self.startclass = startclass
		self.splitdata= []
		self.Centroids = []
		#split data for classes
		for i in range(int(np.amax(self.labels))):
			self.splitdata.append(self.data[self.labels[:,0] ==i+self.startclass])
	def centroids(self, subclasses):
		from sklearn.cluster import KMeans
		self.subclasses = subclasses
		if not self.subclass:
			for i in range(len(self.splitdata)):
				self.Centroids.append(np.mean(self.splitdata[i], axis=0))
		elif subclasses != 0 :
			for i in range(len(self.splitdata)):
				kmeans = KMeans(n_clusters=subclasses, random_state=0).fit(self.splitdata[i])
				self.Centroids.append(kmeans.cluster_centers_)
	def predict(self,data):
		import math
		Class =0
		best = math.inf
		if not self.subclass:
			for i in range(len(self.Centroids)):
				distance = math.sqrt( ((data[0]-self.Centroids[i][0])**2)+((data[1]-self.Centroids[i][1])**2) )
				if distance < best:
					best = distance
					Class = i + self.startclass
			return Class
		elif self.subclass:
			for Centroids in range(len(self.Centroids)):
			
				for Centroid in range(len(self.Centroids[Centroids])):
					distance = math.sqrt( ((data[0]-self.Centroids[Centroids][Centroid][0])**2)+
											((data[1]-self.Centroids[Centroids][Centroid][1])**2) )
					if distance < best:
						best = distance
						Class = Centroids + self.startclass
			return Class

	def score(self,data, labels):
		score =0
		for row in range(np.size(data,0)):
			if self.predict(data[row]) == int(labels[row]):	
				score +=1
		return score/row