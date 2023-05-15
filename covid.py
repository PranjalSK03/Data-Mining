import csv
import math
import matplotlib.pyplot as plt
import random
import numpy as np


filename = "D:\Source_Codes\DM_lab\country_wise_latest.csv"

death = []
confirmed = []
recovered = []
country = []

with open (filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        death.append(int(row["Deaths"]) + int(row["New deaths"]))
        confirmed.append(int(row["Confirmed"]) + int(row["New cases"]))
        recovered.append(int(row["Recovered"]) + int(row["New recovered"]))
        country.append(row["Country/Region"])
        
ratio_dth_conf = []
ratio_rec_conf = []

# finding the ratios for the required field      
for i in range(0,len(death)):
    if death[i]!=0:
        ratio_dth_conf.append(float(confirmed[i]/death[i]))
    else:
        ratio_dth_conf.append(float(1))
    if recovered[i]!=0:
        ratio_rec_conf.append(float(confirmed[i]/recovered[i]))
    else:
        ratio_rec_conf.append(float(1))

#normalizing the data using the min-max normalization
dth_min = min(ratio_dth_conf)
dth_max = max(ratio_dth_conf)
        
rec_min = min(ratio_rec_conf)
rec_max = max(ratio_rec_conf)
 
for i in range(0, len(ratio_dth_conf)):
    ratio_dth_conf[i] = ((ratio_dth_conf[i] - dth_min)/(dth_max - dth_min))*100 + 1
           
for i in range(0, len(ratio_rec_conf)):
    ratio_rec_conf[i] = ((ratio_rec_conf[i] - rec_min)/(rec_max - rec_min))*100 + 1
    
print(ratio_dth_conf)
print(ratio_rec_conf)

# dist_matrix = np.zeros((len(ratio_dth_conf), len(ratio_rec_conf)))
# for i in range(len(ratio_dth_conf)):
#     for j in range(len(ratio_rec_conf)):
#         dist_matrix[i][j] = np.sqrt((ratio_dth_conf[i] - ratio_dth_conf[j])**2 + (ratio_rec_conf[i] - ratio_rec_conf[j])**2)

# print("\n")
# print(dist_matrix)

#making the point tuple
points = []

for i in range(len(ratio_dth_conf)):
    array = []
    array.append(ratio_dth_conf[i])
    array.append(ratio_rec_conf[i])
    array.append(country[i])
    points.append(array)
    
print("\n")

###################################################################################
#DBscan algo
def euclidean_distance(point1, point2):
    return math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(len(point1)-1)))

def get_neighbors(points, point, eps):
    neighbors = []
    for other_point in points:
        if euclidean_distance(point, other_point) <= eps:
            neighbors.append(other_point)
    return neighbors

def dbscan(points, eps, min_samples):
    clusters = []
    visited = set()
    noise = set()

    for point in points:
        if tuple(point) in visited:
            continue
        visited.add(tuple(point))
        neighbors = get_neighbors(points, point, eps)

        if len(neighbors) < min_samples:
            noise.add(tuple(point))
        else:
            cluster = set()
            clusters.append(cluster)
            expand_cluster(points, point, neighbors, cluster, eps, min_samples, visited, noise, clusters)

    return clusters

def expand_cluster(points, point, neighbors, cluster, eps, min_samples, visited, noise, clusters):
    cluster.add(tuple(point))
    i = 0
    while i < len(neighbors):
        neighbor = neighbors[i]
        if tuple(neighbor) not in visited:
            visited.add(tuple(neighbor))
            new_neighbors = get_neighbors(points, neighbor, eps)
            if len(new_neighbors) >= min_samples:
                neighbors.extend(new_neighbors)
        if tuple(neighbor) not in (point for cluster in clusters for point in cluster):
            cluster.add(tuple(neighbor))
            if tuple(neighbor) in noise:
                noise.remove(tuple(neighbor))
        i += 1

#########################################################################################
#main program
eps = float(input("Enter the value of epsilon : "))
min_samples = int(input("Enter the number of the minimum no of points to be core : "))
clusters = dbscan(points, eps, min_samples)

print("\n")
for i in range (0,len(clusters)):
    print("Cluster ",i+1,":\n {", end=" ")
    for j in clusters[i]:
        print(j[2], ", " ,end=" ")
    print("} \n")
 
colors = []

for i in range(200):
    colors.append('#%06X' % random.randint(0, 0xFFFFFF))
for i in range(len(clusters)):
    x = []
    y = []
    for a in clusters[i]:
        x.append(a[0])
        y.append(a[1])
    plt.scatter(x,y, color=colors[i % len(colors)])
plt.show();