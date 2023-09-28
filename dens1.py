import numpy as np
from DenStream import DenStream
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import math
import csv
import sys
import time

with open('file.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))    
    
plt.style.use('classic')

data=np.array(data)
data=np.delete(data,[0,1,2,3,6],axis=1)
data= data.astype(float)
data = np.true_divide(data,100)

clusterer = DenStream(lambd=0.0005, eps=0.02 , beta=0.5, mu=10)
points = []
colors = np.array([x for x in 'rcmykbgrcmykbgrcmykbgrcmyk'])

grid = plt.GridSpec(1, 1, wspace=4, hspace=5)
plt1 = plt.subplot(grid[0, 0])
for i in range(len(data)):
    
    plt1.clear()
    plt1.set_xlim(0,2)
    plt1.set_ylim(0,2)


    d = data[i:i+1,:]
    plt1.plot(d[0,0],d[0,1],'x',color='blue')
    points.append(d)

    #Add point to clusterer
    clusterer.partial_fit(d,1)

    for p in clusterer.p_micro_clusters:
        plt1.scatter(p.center()[0], p.center()[1], color='green',s=p.radius()*50000, alpha=0.9)

    

    for o in clusterer.o_micro_clusters:
        plt1.scatter(o.center()[0], o.center()[1], color='red',s=o.radius()*50000, alpha=0.3)
    

    #Do clustering
    pmc_clusters = clusterer.cluster_p_mcs()

    for i in range(len(clusterer.p_micro_clusters)):
        print(i,pmc_clusters[i], clusterer.p_micro_clusters[i].weight())
    c_id = 0
    while c_id in pmc_clusters:
        cx = 0
        cy = 0
        cn = 0
        for i,p in enumerate(pmc_clusters):
            if p == c_id:
                cx = cx + clusterer.p_micro_clusters[i].center()[0] 
                #* weight tou microcluster

                cy = cy + clusterer.p_micro_clusters[i].center()[1]
                #cn = cn + weight tou i microcluster
                cn = cn + 1
        cx = cx/cn


        cy = cy/cn
        r = 0
        for i,p in enumerate(pmc_clusters):
            if p == c_id:
                rc = math.sqrt((clusterer.p_micro_clusters[i].center()[0] - cx)**2 + (clusterer.p_micro_clusters[i].center()[1]-cy)**2)
                #(for 5 ((clusterer.p_micro_clusters[i].center()[0] - cx)**2 + (clusterer.p_micro_clusters[i].center()[1]-cy)**2)
                       # metablhth = gia x gia y gia z klp kai olo auto sto tetragwno
                if (rc > r) : r = rc
        plt1.scatter(cx, cy, color=colors[c_id],s=100000*r, alpha=0.3)
        c_id = c_id + 1

    plt.suptitle('Συντήρηση και συγχώνευση micro-cluster',fontsize=22)
    plt.xlabel('Χρήση CPU')
    plt.ylabel('Xρήση Memory')
    plt.pause(0.00001)
    plt.show(block=False)

