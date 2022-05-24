import numpy as np
# 
# X = [[[0], [1], [2], [3]]]
# y = [0, 0, 1, 1]
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X,y)
# neigh_dist, _ = neigh.kneighbors([[[0.5]]])
# 
# 
# print(neigh.predict([[[0.5]]]))
# from sklearn.cluster import KMeans
import numpy as np
#
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [10, 2], [10, 4], [10, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# print(kmeans.cluster_centers_)
# print(kmeans.labels_)
# print(kmeans.n_features_in_)
# print(kmeans.predict([[0, 0], [12, 3]]))

a = np.array([[0, 0], [12, 3]])

repeat_embedding = np.repeat([a[0]],repeats=5,axis=0)
repeat_embedding1 = np.repeat([a[1]],repeats=5,axis=0)
dist_to_centers = np.linalg.norm(repeat_embedding - repeat_embedding1, axis=1)
print(repeat_embedding)
print(dist_to_centers)



