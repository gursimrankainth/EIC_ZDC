import ROOT  # type: ignore
import numpy as np  # type: ignore
from dbscan_mod import DBSCAN_S1, DBSCAN_S2

#Load the data
data = np.load('singlephotondata.npz', allow_pickle=True)
event_data_hcal = data['event_data_hcal']
event_data_ecal = data['event_data_ecal']

#Rotate the hit data to the particle's reference frame 
def rotate(event, angle=0.025): 
    rot_lst = []  
    for hit in event:  
        x = hit[0] * np.cos(angle) + hit[2] * np.sin(angle)
        z = -hit[0] * np.sin(angle) + hit[2] * np.cos(angle)
        if len(hit) == 5:
            rot_lst.append((x, hit[1], z, hit[3], hit[4]))
        elif len(hit) == 4:  
            rot_lst.append((x, hit[1], z, hit[3]))
        else: 
            rot_lst.append((x, hit[1], z))
    return rot_lst

events = []
for i, event in enumerate(event_data_hcal[-10:]): 
    hith = [(hit[0], hit[1], hit[2], hit[3], "h") for hit in event_data_hcal[i]]
    hite = [(hit[0], hit[1], hit[2], hit[3], "e") for hit in event_data_ecal[i]]
    temp_list = hith + hite
    events.append(temp_list)

#No noise cut, unrotated
#Scale up HCal energy (sampling fraction is different in ECal and HCal)
scaled_event_filt = []
for event in events: 
    scaled_event = []
    for hit in event:
        hit = list(hit)  
        if hit[4] == 'h':  
            hit[3] = hit[3] * 200  
        scaled_event.append(tuple(hit))  
    scaled_event_filt.append(scaled_event)  

        
#Clustering
def merge_clusters(clusters, distance_threshold):
    def compute_centroid(points):
        return np.mean([p[:3] for p in points], axis=0)

    def is_within_threshold(centroid1, centroid2):
        return np.linalg.norm(centroid1 - centroid2) <= distance_threshold

    def find_closest_cluster(centroid, cluster_centroids):
        distances = {k: np.linalg.norm(centroid - c) for k, c in cluster_centroids.items()}
        closest_cluster = min(distances, key=distances.get)
        return closest_cluster

    #Step 1: Filter out clusters with keys starting with "-1"
    valid_clusters = {k: v for k, v in clusters.items() if not str(k).startswith('-1')}
    ignored_clusters = {k: v for k, v in clusters.items() if str(k).startswith('-1')}

    #Step 2: Separate large and small clusters
    large_clusters = {k: v for k, v in valid_clusters.items() if len(v) > 4}
    small_clusters = {k: v for k, v in valid_clusters.items() if len(v) <= 4}

    #Step 3: Merge large clusters based on centroid distance
    merged_clusters = []
    used_keys = set()

    for key1, points1 in large_clusters.items():
        if key1 in used_keys:
            continue

        centroid1 = compute_centroid(points1)
        new_cluster = points1

        for key2, points2 in large_clusters.items():
            if key1 != key2 and key2 not in used_keys:
                centroid2 = compute_centroid(points2)
                if is_within_threshold(centroid1, centroid2):
                    new_cluster.extend(points2)
                    used_keys.add(key2)

        used_keys.add(key1)
        merged_clusters.append(new_cluster)

    #Compute centroids for the merged large clusters
    merged_cluster_centroids = {i: compute_centroid(cluster) for i, cluster in enumerate(merged_clusters)}

    #Step 4: Merge remaining small clusters with the closest large cluster
    for small_points in small_clusters.values():
        small_centroid = compute_centroid(small_points)
        if merged_cluster_centroids:
            closest_cluster_idx = find_closest_cluster(small_centroid, merged_cluster_centroids)
            merged_clusters[closest_cluster_idx].extend(small_points)
        else:
            #If there are no large clusters, treat the small clusters as merged clusters
            merged_clusters.append(small_points)

    #Step 5: Combine merged clusters with ignored clusters
    final_clusters = {i: cluster for i, cluster in enumerate(merged_clusters)}
    for key, points in ignored_clusters.items():
        final_clusters[key] = points

    return final_clusters

def step_1_check(clusters_1):
    #Find the largest cluster
    largest_cluster_key = max(clusters_1, key=lambda k: len(clusters_1[k]) if k != -1 else 0)
    largest_cluster = clusters_1[largest_cluster_key]

    #Compute the core for all clusters aside from the largest one and the one with key -1
    cores = {}
    keys_to_remove = []  # List to keep track of clusters to be removed

    for key, cluster in clusters_1.items():
        if key != largest_cluster_key and key != -1:
            core = np.mean([p[:3] for p in cluster], axis=0)
            cores[key] = core

    #Merge clusters with cores where the z position is greater than 36000 into the largest cluster
    for key, core in cores.items():
        if core[2] > 36000:  # z position
            largest_cluster.extend(clusters_1[key])
            keys_to_remove.append(key)

    #Remove the clusters identified for removal
    for key in keys_to_remove:
        del clusters_1[key]

    #Add the largest cluster back to the dictionary
    clusters_1[largest_cluster_key] = largest_cluster
    return clusters_1

def merge_dicts_unique_keys(dict1, dict2):
    merged_dict = dict1.copy()  # Start with a copy of the first dictionary
    for key, value in dict2.items():
        new_key = key
        # Ensure the key is unique
        while new_key in merged_dict:
            new_key = str(new_key) + '_1'  # Append '_1' to the key until it's unique
        merged_dict[new_key] = value
    return merged_dict

def euclidean_dist_3D(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)

#Parameters for Steps 1 and 2 
epsilon_s1 = 60
min_pts_s1 = 3
epsilon_s2 = 50
min_pts_s2 = 4

clustered_events = [] 
for event_filt in scaled_event_filt: 
    #Step 1: 
    dbscan_1 = DBSCAN_S1(euclidean_dist_3D, epsilon_s1, min_pts_s1, energy_threshold=0.01)
    clusters_1 = dbscan_1.cluster(event_filt)
    clusters_1 = step_1_check(clusters_1)

    #Step 2: 
    if len(clusters_1) < 5:
        key_largest_clust = max(clusters_1, key=lambda k: len(clusters_1[k]))
        clust_test = clusters_1.pop(key_largest_clust)
        dbscan_2 = DBSCAN_S2(euclidean_dist_3D, epsilon_s2, min_pts_s2, distance_threshold_mm=100)
        clusters_2 = dbscan_2.cluster(clust_test)
        #Merge the clusters if there are too many produced after the second step 
        if len(clusters_2) > 5: 
            clusters_2_merged = merge_clusters(clusters_2, distance_threshold=25)
        #Merge the results from steps 1 and 2
        clusters_mod = merge_dicts_unique_keys(clusters_1, clusters_2_merged)
    else: 
        clusters_mod = clusters_1
    clustered_events.append(clusters_mod)

#print(len(events), len(clustered_events))
#print(clustered_events[0])
test = clustered_events[1]
for key, cluster in test.items():
    print("\nKey:", key)
    print(cluster)