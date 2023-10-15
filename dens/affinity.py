#Import libraries:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import time
import csv
import sys


#Load and read data frame:
colnames=['1', '2', '3','CPU','MEM'] 
data = pd.read_csv('FLASKTEMP+PYFLOODERv1.csv',names=colnames, header=None)
data = data [['CPU','MEM']]
grid = plt.GridSpec(2, 1, wspace=0.4, hspace=0.3)
plt1 = plt.subplot(grid[0, 0])
plt2 = plt.subplot(grid[1, 0])


#Define X as numpy array:
X = np.array(data)


#print(X)

#Fit the model:
st = time.process_time()
# gia pyflooder2min --> preference = -16000
af = AffinityPropagation(preference=30000000000, random_state=0).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)
et = time.process_time()
#Print results:
print(labels)

#Print number of clusters:
print(n_clusters_)
res = et-st
sob =  sys.getsizeof(X)
print('CPU Execution time:', res, 'seconds')
print('Size of object:', sys.getsizeof(X) ,'Bytes')


import matplotlib.pyplot as plt
from itertools import cycle

#plt1.close()
#plt1.figure(1)
plt1.clear()
n_clusters_ =2
colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt1.plot(X[class_members, 0], X[class_members, 1], col + ".")
    plt1.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )
    l_size_of_object = []
    l_size_of_object.append(sob)
    cputime = []
    cputime.append(res)
    for x in X[class_members]:
        plt1.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)


plt2.scatter(cputime,l_size_of_object)
plt.title('Xρόνος εκτέλεσης και κατανάλωση μνήμης',fontsize=18)
plt.suptitle('Eκτιμώμενες ομαδοποιημένες μετρήσεις της συσκευής - Mη Ομαλή Συμπεριφορά',fontsize=22)
plt2.set_xlabel('Χρόνος εκτέλεσης της κεντρικής μονάδας επεξεργασίας (seconds)')
plt2.set_ylabel('Mέγεθος αντικειμένου (bytes)')
plt1.set_xlabel('Στιγμιαία χρήση της κεντρικής μονάδας επεξεργασίας')
plt1.set_ylabel('Στιγμιαία χρήση μνήμης')
plt2.set_xlim(-100,100)
plt2.set_ylim(-100,20000)
plt.show()