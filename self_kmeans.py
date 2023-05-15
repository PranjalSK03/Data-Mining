import random
import math
import matplotlib.pyplot as plt
import pandas as pd

def k_means(points, k):
    # randomly select k points as initial centroids
    centroids = random.sample(points, k)
    clusters = [[] for _ in range(k)]
    # repeat until convergence
    while True:
        # assign each point to the nearest centroid
        for point in points:
            distances = [math.dist(point, centroid) for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(point)
        # recompute the centroids
        new_centroids = []
        for i in range(k):
            cluster = clusters[i]
            if len(cluster) > 0:
                centroid = [sum(p[i] for p in cluster) / len(cluster) for i in range(len(point))]
                new_centroids.append(centroid)
        # check for convergence
        if new_centroids == centroids:
            break
        centroids = new_centroids
        clusters = [[] for _ in range(k)]
    return clusters


number_of_clusters=int(input('Enter the number of clusters.'))
points = []

filename = (pd.read_csv("Dataset/a1.txt", sep = " ", names=['c1', 'c2']))

number_of_points=int(len(filename))

for i in range (0,len(filename)):
    point=(filename.loc[i,"c1"], filename.loc[i,"c2"])
    points.append(point)
    

# print(points)

clusters = k_means(points, number_of_clusters)

for i in range(number_of_clusters):
	print(i," th cluster: ", clusters[i])
 
colors = []

for i in range(200):
    colors.append('#%06X' % random.randint(0, 0xFFFFFF))

for i in range(number_of_clusters):
    cluster_points = clusters[i]
    x= [point[0] for point in cluster_points]
    y= [point[1] for point in cluster_points]
    plt.scatter(x,y, color=colors[i % len(colors)])
plt.show()