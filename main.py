# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
import json
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from kneed import KneeLocator

def report(files, iterations):
    #creates the report
    times = []
    main_times = []
    sparse = {}
    regions = {}
    ports = {}
    navcodes = {}
    navdesc = {}
    nav_objs1 = []
    nav_objs2 = []
    count = 0
    #goes through the each file and parses line by line for json objects
    for file in files:
        for line in open(file,'r'):
            data = json.loads(line)
            df = pd.json_normalize(data)
            if df['mmsi'].values[0] == 205792000:
                nav_objs1.append(nav_obj(df['mmsi'].values[0],df['epochMillis'].values[0],df['navigation.navCode'].values[0],df['navigation.navDesc'].values[0]))
            if df['mmsi'].values[0] == 413970021:
                nav_objs2.append(nav_obj(df['mmsi'].values[0],df['epochMillis'].values[0],df['navigation.navCode'].values[0],df['navigation.navDesc'].values[0]))
            
            navcodes = update_navcodes(navcodes,df)
            navdesc = update_navdesc(navdesc,df)
            regions = update_regions(regions,df)
            ports = update_ports(ports, df)
            sparse = update_sparse(sparse, df)
            times.append(df['epochMillis'].values[0])
            if count > iterations:
                break
            count += 1
        if count > iterations: #added this late
            break
    
    #prints the main time period timestamps
    print('3. Main Time Periods:')
    main_times.append(get_times(times))
    for period in main_times:
        for interval in period:
            print('Period: ' + str(datetime.datetime.fromtimestamp(interval[0]/1000.0)) + ' - ' + str(datetime.datetime.fromtimestamp(interval[1]/1000.0)))
    print('\n')
    
    #prints most sparse variables
    top_sparse = most_sparse(sparse, count)
    print('4. Most Sparse Variables: 1.{} 2.{} 3.{}'.format(top_sparse[0],top_sparse[1],top_sparse[2]))
    print('\n')
    
    #prints regions and ports frequency count
    print('5. Region:',regions)
    print('Port',ports)
    print('\n')
    
    #creates frequency table
    columns = ['Navigation Codes','Navigation Descriptions','Regions','Ports']
    length = max(max(len(navcodes),len(navdesc)),max(len(regions),len(ports)))
    cell_text = [['{0} : {1}'.format(key, value) for key, value in navcodes.items()],
                 ['{0} : {1}'.format(key, value) for key, value in navdesc.items()],
                 ['{0} : {1}'.format(key, value) for key, value in regions.items()],
                 ['{0} : {1}'.format(key, value) for key, value in ports.items()]]
    for i in cell_text:
        while len(i) < length:
            i.append('')
    
    #transpose table
    cell_text = [[cell_text[j][i] for j in range(len(cell_text))] for i in range(len(cell_text[0]))]
    cell_text.insert(0,columns)

    #prints table
    print('6. Frequency Tabulation')
    for i in cell_text:
        print ("{:<16} {:<29} {:<19} {:<9}".format(i[0],i[1],i[2],i[3]))
    print('\n')
    
    #report of MMSI 205792000
    events = create_events(nav_objs1, navcodes)
    print('7. Report for MMSI 205792000')
    for event in events:
        print('MMSI:',event[0])
        print('Timestamp of Last Event:',str(datetime.datetime.fromtimestamp(event[1]/1000.0)))
        print('Navigation Code:',event[2])
        print('Navigation Description:',event[3])
        print('Lead Time(Milliseconds):',event[4])
        print('\n')
        
    #report of MMSI 413970021
    events = create_events(nav_objs2, navcodes)
    print('8. Report for MMSI 413970021')
    for event in events:
        print('MMSI:',event[0])
        print('Timestamp of Last Event:',str(datetime.datetime.fromtimestamp(event[1]/1000.0)))
        print('Navigation Code:',event[2])
        print('Navigation Description:',event[3])
        print('Lead Time(Milliseconds):',event[4])
        print('\n')
    
        
def create_events(nav_objs, nav_codes):
    #keeps only top nav codes, sorts by timestamp, and creates contiguous events with the same nav code for reporting
    top_codes = list(nav_codes.keys())[0:5] 
    for obj in nav_objs: #takes only the top 5 nav codes
        if obj.getNavcode() not in top_codes: #limit to only the top 5 navigation codes
            nav_objs.remove(obj)

    nav_objs = sorted(nav_objs, key = lambda x: x.getTimestamp()) #sorts the nav obj list by ascending timestamp
    events = [] #use a list to ensure contiguous events
    new_obj = True
    code = None
    first = None
    last_obj = None
    for obj in nav_objs: #creates events
        if new_obj:
            new_obj = False
            code = obj.getNavcode()
            first = obj.getTimestamp()
        if obj.getNavcode() != code or obj is nav_objs[-1]:
            events.append([last_obj.getMmsi(),last_obj.getTimestamp(),last_obj.getNavcode(),last_obj.getNavdesc(),last_obj.getTimestamp()-first])
            first = obj.getTimestamp()
            code = obj.getNavcode()
        last_obj = obj
    return events
    
def update_navcodes(navcodes,df):
    #updates the frequency tabulation of navigation codes variables in the dataset
    data = df['navigation.navCode'].values[0]
    if data in navcodes.keys():
        navcodes[data] += 1
    else:
        navcodes[data] = 1
    return dict(sorted(navcodes.items(), key = lambda x: x[1], reverse = True))

def update_navdesc(navdesc,df):
    #updates the frequency tabulation of navigation description variables in the dataset
    data = df['navigation.navDesc'].values[0]
    if data in navdesc.keys():
        navdesc[data] += 1
    else:
        navdesc[data] = 1
    return dict(sorted(navdesc.items(), key = lambda x: x[1], reverse = True))

def update_regions(regions,df):
    #updates the frequency tabulation of regions variables in the dataset
    data = df['olson_timezone'].values[0]
    if data in regions.keys():
        regions[data] += 1
    else:
        regions[data] = 1
    return dict(sorted(regions.items(), key = lambda x: x[1], reverse = True))

def update_ports(ports,df):
    #updates the frequency tabulation of ports variables in the dataset
    data = df['port.name'].values[0]
    if data in ports.keys():
        ports[data] += 1
    else:
        ports[data] = 1
    return dict(sorted(ports.items(), key = lambda x: x[1], reverse = True))

def most_sparse(sparse,c):
    #calculates the top 3 most sparse variables
    top = {}
    for key, value in sparse.items():
        if len(top) >= 3 and c - value > list(top.values())[-1]: #more sparse variables will have a lower frequency
            top.popitem()
            top[key] = c - value
            top = dict(sorted(top.items(), key = lambda x: x[1], reverse = True)) #keeps the list in descending order
        elif len(top) < 3: #initially populates the list
            top[key] = c - value
            top = dict(sorted(top.items(), key = lambda x: x[1], reverse = True))
    return list(top.keys())
            

def update_sparse(sparse, df):
    #updates the frequency tabulation of sparse variables in the dataset
    for i in df:
        if i in sparse.keys():
            sparse[i] += 1
        else:
            sparse[i] = 1
    return sparse

def get_times(times):
    #create a histogram of the times
    min_time = min(times)
    max_time = max(times)
    num_bins = int((max_time - min_time)/(3.6*10**6)) #divide range into hour blocks
    n, bins, patches = plt.hist(times,bins=num_bins) #returns the value of each bin and the boundaries of the bins
    #plt.title('Timestamps of Data')
    #plt.xlabel('Timestamps per hour')
    #plt.ylabel('Count')
    #plt.show()
    
    #remove all zeros from the data structures
    bins = np.delete(bins, 0) #bins has one more value than n
    for x in range(len(n)): #threw off clusters
        if n[x] < 1:
            bins[x] = 0
    n = n[n != 0]
    bins = bins[bins != 0]
    hist = np.column_stack((bins,n))
    
    #calculate the knee to find the optimal number of clusters
    squared_distances = []
    K = range(1,min(len(hist),20))
    for num_clusters in K:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(hist)
        squared_distances.append(kmeans.inertia_)
    kn = KneeLocator(K, squared_distances, curve='convex', direction='decreasing')
    #plt.plot(K,squared_distances,'bx-')
    #plt.xlabel('K') 
    #plt.ylabel('Sum of squared distances') 
    #plt.title('Elbow Method For Optimal K')
    #plt.xlim(0,20)
    #plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    #plt.show()
    
    #use the optimal K value and kmeans to create the clusters
    kmeans = KMeans(n_clusters=kn.knee)
    kmeans.fit(hist)
    y_kmeans = kmeans.predict(hist)
    #plt.scatter(hist[:, 0], hist[:, 1], c=y_kmeans, s=50, cmap='viridis')
    #plt.show()
    
    
    #correlate clusters to time periods
    new = True
    group = None
    start = 0
    end = 0
    main_times = []
    last_val = 0
    for i in range(len(y_kmeans)):
        if n[i] > 0:
            if new: #only executes on first run
                start = i
                group = y_kmeans[i]
                new = False
            elif y_kmeans[i] != group: #adds the min and max timestamp from eac cluster to the data structure
                end = last_val
                main_times.append([bins[start],bins[end]])
                start = i
                group = y_kmeans[i]
            last_val = i
    end = last_val
    main_times.append([bins[start],bins[end]])
    #print(y_kmeans)
    #print(n)
    #print(bins)
    #print(main_times)
    return main_times

class nav_obj:
    def __init__(self,mmsi,timestamp,navcode,navdesc):
        self.mmsi = mmsi
        self.timestamp = timestamp
        self.navcode = navcode
        self.navdesc = navdesc
        
    def getMmsi(self):
        return self.mmsi
    
    def setMmsi(self,m):
        self.mmsi = m
        
    def getTimestamp(self):
        return self.timestamp
    
    def setTimestamp(self,t):
        self.timestamp = t
        
    def getNavcode(self):
        return self.navcode
    
    def setNavcode(self, n):
        self.navcode = n
        
    def getNavdesc(self):
        return self.navdesc
    
    def setNavdesc(self,n):
        self.navdesc = n
        
if __name__ == '__main__':
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print('Please enter the JSON Files'' Directory Path')
        sys.exit(0)
    if not os.path.isdir(sys.argv[1]):
        print('Please enter a valid directory')
        sys.exit(0)
    iterations = 10000 #default
    if len(sys.argv) == 3:
        try:
            iterations = int(sys.argv[2]) if int(sys.argv[2]) > 100 else 10000
        except ValueError:
            print('Please enter a valid integer')
            sys.exit(0)
    
    path = sys.argv[1]
    files = []
    for file in os.listdir(path):
        if file.endswith('.json'):
            files.append(os.path.join(path, file))
    data = report(files, iterations)