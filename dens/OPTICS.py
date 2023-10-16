import numpy as np
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import time
import csv
import sys

# Create a sample dataset (replace this with your own data)
#Load and read data frame:
colnames=['1', '2', '3','CPU','MEM'] 
data = pd.read_csv('FLASKTEMP+PYFLOODERv1.csv',names=colnames, header=None)
data = data [['CPU','MEM']]
grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.3)
plt1 = plt.subplot(grid[0, 0])
# plt2 = plt.subplot(grid[1, 0])
X = np.array(data)

# Create an OPTICS clustering object
st = time.process_time()
clustering = OPTICS(min_samples=5, xi=0.05)

# Fit the OPTICS model to your data
clustering.fit(X)

# Plot the reachability plot
reachability = clustering.reachability_
ordering = clustering.ordering_
et = time.process_time()
res = et-st
sob =  sys.getsizeof(X)
print('CPU Execution time:', res, 'seconds')
print('Size of object:', sys.getsizeof(X) ,'Bytes')
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(ordering)), reachability, marker='.')
plt.title('Reachability Plot')
plt.xlabel('Data Points')
plt.ylabel('Reachability Distance')
plt.show()

# Extract clusters using a reachability distance threshold
# You can adjust this threshold to obtain different clusters
threshold = 0.5
labels = clustering.labels_

# Print the cluster labels
print("Cluster labels:", labels)

# Visualize the clusters
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('OPTICS Clustering')
# plt2.scatter(res,sob)
plt.title('Xρόνος εκτέλεσης και κατανάλωση μνήμης',fontsize=18)
plt.suptitle('OPTICS - Eκτιμώμενες ομαδοποιημένες μετρήσεις της συσκευής - Mη Ομαλή Συμπεριφορά',fontsize=22)
# plt2.set_xlabel('Χρόνος εκτέλεσης της κεντρικής μονάδας επεξεργασίας (seconds)')
# plt2.set_ylabel('Mέγεθος αντικειμένου (bytes)')
plt1.set_xlabel('Στιγμιαία χρήση της κεντρικής μονάδας επεξεργασίας')
plt1.set_ylabel('Στιγμιαία χρήση μνήμης')
# plt2.set_xlim(-100,100)
# plt2.set_ylim(-100,20000)
plt.show()