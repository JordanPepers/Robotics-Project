import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentreTracker:

	def __init__(self, maxDisappeared=50, maxDistance=50):

		self.nextFishID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		self.maxDisappeared = maxDisappeared
		self.maxDistance = maxDistance

	def deregister(self, fishID):

		del self.disappeared[fishID]
		del self.objects[fishID]

	def register(self, centroid):

		self.objects[self.nextFishID] = centroid
		self.disappeared[self.nextFishID] = 0
		self.nextFishID += 1

	def update(self, rects):

		if len(rects) == 0:
			for fishID in list(self.disappeared.keys()):
				self.disappeared[fishID] += 1

				if self.disappeared[fishID] > self.maxDisappeared:
					self.deregister(fishID)

			return self.objects


		inputCentres = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cY = int((startY + endY) / 2.0)
			cX = int((startX + endX) / 2.0)
			inputCentres[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentres)):
				self.register(inputCentres[i])

		else:
			objectCentres = list(self.objects.values())
			fishIDs = list(self.objects.keys())


			D = dist.cdist(np.array(objectCentres), inputCentres)
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]

			usedCols = set()
			usedRows = set()


			for (row, col) in zip(rows, cols):

				if row in usedRows or col in usedCols:
					continue

				if D[row, col] > self.maxDistance:
					continue

				fishID = fishIDs[row]
				self.objects[fishID] = inputCentres[col]
				self.disappeared[fishID] = 0

				usedCols.add(col)
				usedRows.add(row)


			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			if D.shape[0] >= D.shape[1]:

				for row in unusedRows:

					fishID = fishIDs[row]
					self.disappeared[fishID] += 1

					if self.disappeared[fishID] > self.maxDisappeared:
						self.deregister(fishID)

			else:
				for col in unusedCols:
					self.register(inputCentres[col])

		return self.objects