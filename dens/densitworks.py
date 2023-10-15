import numpy as np
from DenStream import DenStream
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import math
import os
import csv
import sys
import time

with open('FLASKTEMP.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))    
    
plt.style.use('dark_background')

data=np.array(data)
data=np.delete(data,[0,1,2],axis=1)
data= data.astype(float)
data = np.true_divide(data,100)

clusterer = DenStream(lambd=0.0005, eps=0.004 , beta=0.5, mu=10)
points = []
colors = np.array([x for x in 'rcmykbgrcmykbgrcmykbgrcmyk'])

grid = plt.GridSpec(2, 1, wspace=0.4, hspace=0.7)
plt1 = plt.subplot(grid[0, 0])
plt2 = plt.subplot(grid[1, 0])
res = 0
for z in range(len(data)):
    
    plt1.clear()
    plt1.set_xlim(0,1.2)
    plt1.set_ylim(0,1)


    d = data[z:z+1,:]
    plt1.plot(d[0,0],d[0,1],'x',color='blue')
    points.append(d)

    et1 = time.process_time()

    #Add point to clusterer
    clusterer.partial_fit(d,1)
    print(f"Aριθμός p_micro_clusters is {len(clusterer.p_micro_clusters)}")

    for p in clusterer.p_micro_clusters:
        print(f"Κέντρo p_micro_cluster = {p.center()}")
        print(f"Radius of p_micro_cluster= {p.radius()}")
        plt1.scatter(p.center()[0], p.center()[1], color='green',s=p.radius()*5000, alpha=0.9)

    
    print(f"Αριθμός o_micro_clusters is {len(clusterer.o_micro_clusters)}")
    for o in clusterer.o_micro_clusters:
        print(f"Κέντρo o_micro_cluster = {o.center()}")
        plt1.scatter(o.center()[0], o.center()[1], color='red',s=o.radius()*5000, alpha=0.9)
    

    #Do clustering
    pmc_clusters = clusterer.cluster_p_mcs()

    v_size_of_object = sys.getsizeof(clusterer.p_micro_clusters) + sys.getsizeof(clusterer.o_micro_clusters);
                                

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
                if (rc > r) : r = rc
        plt1.scatter(cx, cy, color=colors[c_id],s=1000000*r, alpha=0.9)
        c_id = c_id + 1

    et2 = time.process_time()
    et = et2 - et1
    res = res + (et2 - et1)

    print('Χρόνος εκτέλεσης της κεντρικής μονάδας επεξεργασίας: ', res, 'seconds')

    cputime = []
    cputime.append(res)
    l_size_of_object = []
    l_size_of_object.append(v_size_of_object)

    print('Mέγεθος αντικειμένου: ', sys.getsizeof( l_size_of_object) , 'bytes')
    plt2.clear()
    plt2.scatter(cputime,l_size_of_object)
    plt2.set_xlabel('Χρόνος εκτέλεσης της κεντρικής μονάδας επεξεργασίας (seconds)')
    plt2.set_ylabel('Mέγεθος αντικειμένου (bytes)')
    plt2.set_xlim( 0, 100)
    plt2.set_ylim( 0, 1000)

    plt.title('Xρόνος εκτέλεσης και κατανάλωση μνήμης',fontsize=18)
    plt.suptitle('Oμαδοποιημένες μετρήσεις της συσκευής - Ομαλή Συμπεριφορά',fontsize=22)
    
    plt1.set_xlabel('Στιγμιαία χρήση της κεντρικής μονάδας επεξεργασίας')
    plt1.set_ylabel('Στιγμιαία χρήση μνήμης')
    
    plt.pause(0.0000000001)

    filename = os.path.join ('plots', f"plot_{z}.png")
    plt.savefig(filename,bbox_inches='tight')
    plt.show(block=False)
filename = os.path.join ('plots', f"plot_{z}.png")
plt.savefig(filename)
