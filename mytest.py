import numpy as np
from DenStream import DenStream
import matplotlib.pyplot as plt
import math
import csv

with open('.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

plt.style.use('ggplot')

plt.ylim(0,1)

data = np.array(data)
data = np.delete(data,[0,1,2,3,6],axis=1)
data = data.astype(float)
data = np.true_divide(data,100)
print(data)

clusterer = DenStream(lambd=0.005, eps=30, beta=0.5, mu=3)
for i in range(len(data)):
    d = data[i:i+1,:]
    print(d)
    #print(row.shape):
    clusterer.partial_fit(d, 1)
    #plt.plot(d[0,0],d[0,1],'x')
    print(f"Number of p_micro_clusters is {len(clusterer.p_micro_clusters)}")
    for p in clusterer.p_micro_clusters:
        #print(p.center())
        #print(p.radius())   
        plt.scatter(p.center()[0], p.center()[1], color='green',s=p.radius(), alpha=0.9,
            cmap='viridis')
    print(f"Number of o_micro_clusters is {len(clusterer.o_micro_clusters)}")
    for o in clusterer.o_micro_clusters:
        #print(o.center())
        #print(o.radius())   
        plt.scatter(o.center()[0], o.center()[1], color='red',s=30, alpha=0.3,
            cmap='viridis')
    
    #Do clustering    
    pmc_clusters = clusterer.cluster_p_mcs()
    print(pmc_clusters)
    plt.pause(0.00000000000001)
    plt.show(block=False)


