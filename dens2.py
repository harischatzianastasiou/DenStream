import numpy as np
from DenStream import DenStream
import matplotlib.pyplot as plt


import chart_studio.plotly as py

import math
import csv
import sys
import time

with open('aploflask1hour.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))    

#st = time.process_time()
# plt.ylim(0,1)
plt.style.use('classic')

data=np.array(data)
data=np.delete(data,[0,1,2,3,6],axis=1)
data= data.astype(float)
data = np.true_divide(data,100)
print(data)

clusterer = DenStream(lambd=0.005, eps=0.0005 , beta=0.5, mu=3)
points = []
colors = np.array([x for x in 'rcmykbgrcmykbgrcmykbgrcmyk'])

grid = plt.GridSpec(1, 1, wspace=4, hspace=5)
plt1 = plt.subplot(grid[0, 0])
# plt2 = plt.subplot(grid[1, 0])
res = 0
for i in range(len(data)):
    #
   # plt.ylim(0,1)
    #plt.xlim(0,1.5)
    plt1.clear()
    #plt2.clear()
    plt1.set_xlim(0,2)
    plt1.set_ylim(0,1)

    #plt1.title("Point {}".format(str(i)))

    d = data[i:i+1,:]
    #for ip in points:
    #    plt.plot(ip[0,0],ip[0,1],'+',color='blue')
    #print(row.shape)
    plt1.plot(d[0,0],d[0,1],'x',color='blue')
    points.append(d)

    #Add point to clusterer
    et1 = time.process_time()
    clusterer.partial_fit(d,1)

    print(d)
    print(f"Number of p_micro_clusters is {len(clusterer.p_micro_clusters)}")
    # cpu time
    # clusterer cpu time ana iteration, ram iterations 
    # k means loopa
    for p in clusterer.p_micro_clusters:
        print(p.center())
        print(f"p.radius = {p.radius()}")   
        plt1.scatter(p.center()[0], p.center()[1], color='green',s=p.radius()*50000, alpha=0.9)

    
    
    print(f"Number of o_micro_clusters is {len(clusterer.o_micro_clusters)}")
    for o in clusterer.o_micro_clusters:
        # print("ocluster") --
        print(o.center()) 
        print(f"o.radius = {o.radius()}") 
        plt1.scatter(o.center()[0], o.center()[1], color='red',s=o.radius()*50000, alpha=0.3)
    

    #Do clustering
    pmc_clusters = clusterer.cluster_p_mcs()
    # print('Size of object:', sys.getsizeof(clusterer.p_micro_clusters) ,'Bytes') --
    # print('Size of object:', sys.getsizeof(clusterer.o_micro_clusters) ,'Bytes') --

    sob =  sys.getsizeof(clusterer.p_micro_clusters) +  sys.getsizeof(clusterer.o_micro_clusters);

    print(pmc_clusters)

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


    et2 = time.process_time() 

    et = et2- et1
    # get execution time
    res = res + (et2- et1)
    # print('CPU Execution time:', res, 'seconds') --

    cputime = []
    cputime.append(res);

    size_of_object = []
    size_of_object.append(sob);

    # plt2.scatter(size_of_object,cputime)
    #ticks = range(0,len(thislist)+1, 1)
    #plt1.set_xticks(ticks, minor=False)
    # plt2.set_xlim(-100,10000)
    # plt2.set_ylim(-100,100)
   
    # plt.pause(0.00000000001)

    plt.title('Συντήρηση και συγχώνευση micro-cluster')
    plt.xlabel('time')
    plt.ylabel('Mean Hours Users Spends on the Website')
    # plt.pause(3)
    # plt.show(block=False)

