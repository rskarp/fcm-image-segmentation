# Riley Karp
# 12/04/2022
# EN.525.770.82
# Course Project

import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import cv2
import time
import os
import concurrent.futures
from scipy.spatial import distance
from scipy import io
from enum import Enum

Algorithm = Enum('Algorithm', ['FCM', 'IT2FCM', 'CPIT2FCM', 'KMeans'])
dir_path = os.path.dirname(os.path.realpath(__file__))

# Generate graphical data sets
class Datasets:
    # Based on plots in figure 4
    def get_square_clusters():
        c1 = np.array([[0.5,2.2],[0.7,2.2],[0.9,2.2],
            [0.5,2.0],[0.7,2.0],[0.9,2.0],
            [0.5,1.8],[0.7,1.8],[0.9,1.8]])
        c2 = np.array([[3.,3.],[4.8,3.],[6.6,3.],
            [3.,2.],[4.8,2.],[6.6,2.],
            [3.,1.],[4.8,1.],[6.6,1.]])
        return c1,c2

    # Based on plots in figure 5
    def get_sphere_clusters(n=200):
        stdx = 10**0.5
        stdy = 0.1**0.5
        mu1 = [0,0]
        mu2 = [0,2]
        mu3 = [0,4]
        c1 = np.array([np.random.normal(mu1[0], stdx,size=[n,1]),np.random.normal(mu1[1], stdy,size=[n,1])])[:,:,0].T
        c2 = np.array([np.random.normal(mu2[0], stdx,size=[n,1]),np.random.normal(mu2[1], stdy,size=[n,1])])[:,:,0].T
        c3 = np.array([np.random.normal(mu3[0], stdx,size=[n,1]),np.random.normal(mu3[1], stdy,size=[n,1])])[:,:,0].T
        return c1,c2,c3

    # Based on plots in figures 6-8
    def get_rod_clusters(cluster_separation,n=100):
        stdx = 0.01**0.5
        stdy = 4**0.5
        mu1 = [10, 10]
        mu2 = [10+cluster_separation, 10]
        c1 = np.array([np.random.normal(mu1[0], stdx,size=[n,1]),np.random.normal(mu1[1], stdy,size=[n,1])])[:,:,0].T
        c2 = np.array([np.random.normal(mu2[0], stdx,size=[n,1]),np.random.normal(mu2[1], stdy,size=[n,1])])[:,:,0].T
        return c1,c2

class FCM:
    m = 2
    maxIters = 300
    stopDelta = 10e-4

    # Initialize random cluster centers (cxf matrix)
    def initialize_cluster_centers(self, data, c, f):
        return np.array([np.random.uniform(np.min(data[:,i]), np.max(data[:,i]), c) for i in range(f)]).T

    # Calculate distance from each point to each cluster (nxc matrix)
    def calculate_distances(self,data, cluster_centers):
        return np.array([[np.linalg.norm(x-c) for c in cluster_centers] for x in data])

    # Calculate membership grades of each point for each cluster (nxc matrix)
    def calculate_partition_matrix(self,D):
        n,c = D.shape 
        exp = float(2/(self.m-1))
        U = np.zeros(D.shape)
        for i in range(n):
            for j in range(c):
                d = float(D[i,j])
                U[i,j] = 1/np.sum([np.power(d/dc,exp) for dc in D[i,:]])
        return U
    
    # Calculate new cluster centers (cxf matrix)
    def calculate_new_centers(self,data,U):
        n,c = U.shape
        f = data.shape[1]
        V = np.zeros([c,f])
        for j in range(c):
            num = np.sum([np.power(U[i,j],self.m)*data[i,:] for i in range(n)], axis=0)
            den = np.sum([np.power(U[i,j],self.m) for i in range(n)], axis=0)
            V[j,:] = num/den
        return V

    # Get list of cluster labels (nx1 matrix)
    def get_data_labels(self, U):
        return np.array([np.argmax(ux) for ux in U])

    # Get data separated into clusters (returns list of length c where each element
    # is an array of points in that cluster)
    def get_cluster_data(self, c, labels, data):
        cluster_data = []
        for i in range(c):
            indices = np.where(labels == i)
            pts = np.concatenate([data[x,:] for x in indices],axis=0)
            cluster_data.append(pts)
        return cluster_data

    def run(self,data, c):
        # Get num points & num features
        n,f = data.shape 
        # Randomly generate cluster centers within the range of the given data (cxf matrix)
        V = self.initialize_cluster_centers(data,c,f)
        # Calculate distance from each point to each cluster center (nxc matrix)
        D = self.calculate_distances(data,V)
        # Calculate membership grades of each point for each cluster (nxc matrix)
        U = self.calculate_partition_matrix(D)

        # Initialize iteration parameters
        delta = sys.maxsize
        iter = 0
        oldV = np.ones(V.shape)*sys.maxsize

        # Iterate to minimize objective cost function
        while delta > self.stopDelta and iter < self.maxIters:
            if iter%9 == 0:
                print(f'Iteration: {iter}')
            # Update cluster centers
            V = self.calculate_new_centers(data,U)
            # Calculate new distances to clusters
            D = self.calculate_distances(data,V)
            # Calculate new membership grades
            U = self.calculate_partition_matrix(D)
            # Calculate delta
            delta = np.linalg.norm(oldV-V) 
            oldV = V
            # Increment counter
            iter += 1
        
        print(f'Final Iteration: {iter}')

        # Get list of cluster labels (nx1 matrix)
        labels = self.get_data_labels(U)
        # Get data separated into clusters
        cluster_data = self.get_cluster_data(c, labels, data)
        
        # Return cluster centers, membership grades, cluster data, and data labels
        return V, U, cluster_data, labels

class IT2FCM:
    m = 2
    m1 = 2
    m2 = 8
    maxIters = 300
    stopDelta = 10e-4

    # Initialize random cluster centers (cxf matrix)
    def initialize_cluster_centers(self, data, c, f):
        return np.array([np.random.uniform(np.min(data[:,i]), np.max(data[:,i]), c) for i in range(f)]).T

    # Calculate lower and upper membership grades of each point for each cluster (nxc matrices)
    def calculate_membership_grades(self,D):
        n,c = D.shape 
        exp1 = float(2/(self.m1-1))
        exp2 = float(2/(self.m2-1))
        u_l = np.array(
            [ [ max(1/np.sum([np.power(float(D[i,j])/dc,exp1) for dc in D[i,:]]),1/np.sum([np.power(float(D[i,j])/dc,exp2) for dc in D[i,:]])) 
                for j in range(c) ] 
            for i in range(n) ] )
        u_u = np.array(
            [ [ max(1/np.sum([np.power(float(D[i,j])/dc,exp1) for dc in D[i,:]]),1/np.sum([np.power(float(D[i,j])/dc,exp2) for dc in D[i,:]])) 
                for j in range(c) ] 
            for i in range(n) ] )
        return u_l, u_u

    # Calculate new center and membership grades for given feature of given cluster
    def update_center(self, dto):
        sorted_data_ids = dto['sorted_data_ids']
        dataPts = dto['dataPts']
        dataIds = dto['dataIds']
        vjl_R = dto['vjl_R']
        vjl_L = dto['vjl_L']
        u_l = dto['u_l']
        u_u = dto['u_u']
        n = dto['n']
        j = dto['j']
        l = dto['l']

        iter = 0
        stop = False
        UR = np.zeros((n,))
        UL = np.zeros((n,))
        while (stop == False and iter < self.maxIters):
            iter += 1
            k_R = np.where([dataPts[dataIds[i]] <= vjl_R and vjl_R <= dataPts[dataIds[i+1]] for i in range(n)])[0][0]
            k_L = np.where([dataPts[dataIds[i]] <= vjl_L and vjl_L <= dataPts[dataIds[i+1]] for i in range(n)])[0][0]
            for i,id in enumerate(sorted_data_ids):
                if i <= k_R:
                    UR[id] = u_l[id]
                else:
                    UR[id] = u_u[id]
                if i <= k_L:
                    UL[id] = u_u[id]
                else:
                    UL[id] = u_l[id]
            num_R = np.sum([np.power(UR[i],self.m)*dataPts[i] for i in range(n)])
            den_R = np.sum([np.power(UR[i],self.m) for i in range(n)])
            newVjl_R = num_R/den_R
            num_L = np.sum([np.power(UL[i],self.m)*dataPts[i] for i in range(n)])
            den_L = np.sum([np.power(UL[i],self.m) for i in range(n)])
            newVjl_L = num_L/den_L
            delta_R = abs(vjl_R-newVjl_R)
            delta_L = abs(vjl_L-newVjl_L)
            if (delta_R < self.stopDelta and delta_L < self.stopDelta):
                stop = True
            else:
                vjl_R = newVjl_R
                vjl_L = newVjl_L
        
        return j, l, UL, UR, vjl_R, vjl_L

    # Use Karnik-Mendel iterative algorithm to find cluser interval centers
    def calculate_cluster_bounds(self, data, u_l, u_u):
        start = time.perf_counter()
        n,f = data.shape
        c = u_l.shape[1]
        U = (u_l + u_u)/2.0
        UL = np.zeros(U.shape)
        UR = np.zeros(U.shape)
        VL = np.zeros((c,f))
        VR = np.zeros((c,f))
        
        # Calculate interval type 1 cluster centroids
        centroids = np.zeros([c,f])
        for j in range(c):
            num = np.sum([U[i,j]*data[i,:] for i in range(n)], axis=0)
            den = np.sum([U[i,j] for i in range(n)], axis=0)
            centroids[j,:] = num/den

        sorted_data_ids = np.argsort(data, kind ='mergesort', axis = 0)
        dtos = []
        for j in range(c):
            for l in range(f):
                vjl_R = centroids[j,l]
                vjl_L = centroids[j,l]
                dataPts = data[:,l]
                dataIds = sorted_data_ids[:,l]

                dtos.append({
                    'sorted_data_ids': sorted_data_ids,
                    'dataPts': dataPts,
                    'dataIds': dataIds,
                    'vjl_R': vjl_R,
                    'vjl_L': vjl_L,
                    'u_u': u_u[:,j],
                    'u_l': u_l[:,j],
                    'n': n,
                    'j': j,
                    'l': l
                })
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            pool = executor.map(self.update_center, dtos)
            for (j, l, ULj, URj, vjl_R, vjl_L) in pool:
                VR[j,l] = vjl_R
                VL[j,l] = vjl_L
                UR[:,j] = URj
                UL[:,j] = ULj
        end = time.perf_counter()
        print(f'Calculated centers in {round(end-start, 2)} second(s)') 

        return UL,UR,VL,VR

    # Calculate Euclidean distance between old and new cluster centers
    def get_delta(self, oldV, V):
        return np.linalg.norm(oldV-V)

    # Get list of cluster labels (nx1 matrix)
    def get_data_labels(self, U):
        return np.array([np.argmax(ux) for ux in U])

    # Get data separated into clusters (returns list of length c where each element
    # is an array of points in that cluster)
    def get_cluster_data(self, c, labels, data):
        cluster_data = []
        for i in range(c):
            indices = np.where(labels == i)
            pts = np.concatenate([data[x,:] for x in indices],axis=0)
            cluster_data.append(pts)
        return cluster_data

    def run(self,data, c):
        # Get num points & num features
        n,f = data.shape 
        # Randomly generate cluster centers within the range of the given data (cxf matrix)
        V = self.initialize_cluster_centers(data,c,f)

        # Initialize iteration parameters
        oldV = np.ones(V.shape)*sys.maxsize
        delta = self.get_delta(oldV,V)
        iter = 0

        start = time.time()
        # Iterate to minimize objective cost function
        while delta > self.stopDelta and iter < self.maxIters:
            if iter%4 == 0:
                end = time.time()
                print(f'Iteration: {iter} , total time: {end-start} seconds ({(end-start)/60} minutes)')
            oldV = V
            # Calculate new distances to clusters (nxc matrix)
            startdist = time.time()
            D = distance.cdist(data,V,'euclidean')
            enddist = time.time()
            print(f'Calculated distances in: {enddist-startdist} seconds ({(enddist-startdist)/60} minutes)')
            # Calculate upper/lower membership grades (nxc matrices)
            startdist = time.time()
            u_l, u_u = self.calculate_membership_grades(D)
            enddist = time.time()
            print(f'Calculated membership grades in: {enddist-startdist} seconds ({(enddist-startdist)/60} minutes)')
            # Calculate new cluster center intervals
            UL,UR,VL,VR = self.calculate_cluster_bounds(data, u_l, u_u)
            # Defuzzify cluster centers
            V = (VL+VR)/2.0

            # Calculate delta
            delta = self.get_delta(oldV,V)
            # Increment counter
            iter += 1
        
        print(f'Final Iteration: {iter}')

        # Defuzzify membership grades
        U = (UL + UR)/2.0

        # Get list of cluster labels (nx1 matrix)
        labels = self.get_data_labels(U)
        # Get data separated into clusters
        cluster_data = self.get_cluster_data(c, labels, data)

        # Return cluster centers, membership grades, cluster data, and data labels
        return V, U, cluster_data, labels

class CPIT2FCM:
    m = 2
    maxIters = 300
    stopDelta = 10e-4

    # Initialize random cluster centers (cxf matrix)
    def initialize_cluster_centers(self, data, c, f):
        return np.array([np.random.uniform(np.min(data[:,i]), np.max(data[:,i]), c) for i in range(f)]).T

    # Calculate distance from each point to each cluster (nxc matrix)
    def calculate_distances(self,data, cluster_centers):
        c,f = cluster_centers.shape
        v_l = cluster_centers - 0.10
        v_u = cluster_centers + 0.10
        
        D_l = distance.cdist(data,v_l,'mahalanobis')
        D_u = distance.cdist(data,v_u,'mahalanobis')
        return D_l,D_u

    # Calculate lower and upper membership grades of each point for each cluster (nxc matrices)
    def calculate_membership_grades(self,D_l,D_u):
        n,c = D_l.shape 
        exp = 2.0/(self.m-1)
        u_l = np.array(
            [ [ max(1/np.sum([np.power(float(D_l[i,j])/dc,exp) for dc in D_l[i,:]]),1/np.sum([np.power(float(D_u[i,j])/dc,exp) for dc in D_u[i,:]])) 
                for j in range(c) ] 
            for i in range(n) ] )
        u_u = np.array(
            [ [ max(1/np.sum([np.power(float(D_l[i,j])/dc,exp) for dc in D_l[i,:]]),1/np.sum([np.power(float(D_u[i,j])/dc,exp) for dc in D_u[i,:]])) 
                for j in range(c) ] 
            for i in range(n) ] )
        return u_l, u_u

    # Calculate new center and membership grades for given feature of given cluster
    def update_center(self, dto):
        sorted_data_ids = dto['sorted_data_ids']
        dataPts = dto['dataPts']
        dataIds = dto['dataIds']
        vjl_R = dto['vjl_R']
        vjl_L = dto['vjl_L']
        u_l = dto['u_l']
        u_u = dto['u_u']
        n = dto['n']
        j = dto['j']
        l = dto['l']

        iter = 0
        stop = False
        UR = np.zeros((n,))
        UL = np.zeros((n,))
        while (stop == False and iter < self.maxIters):
            iter += 1
            k_R = np.where([dataPts[dataIds[i]] <= vjl_R and vjl_R <= dataPts[dataIds[i+1]] for i in range(n)])[0][0]
            k_L = np.where([dataPts[dataIds[i]] <= vjl_L and vjl_L <= dataPts[dataIds[i+1]] for i in range(n)])[0][0]
            for i,id in enumerate(sorted_data_ids):
                if i <= k_R:
                    UR[id] = u_l[id]
                else:
                    UR[id] = u_u[id]
                if i <= k_L:
                    UL[id] = u_u[id]
                else:
                    UL[id] = u_l[id]
            num_R = np.sum([np.power(UR[i],self.m)*dataPts[i] for i in range(n)])
            den_R = np.sum([np.power(UR[i],self.m) for i in range(n)])
            newVjl_R = num_R/den_R
            num_L = np.sum([np.power(UL[i],self.m)*dataPts[i] for i in range(n)])
            den_L = np.sum([np.power(UL[i],self.m) for i in range(n)])
            newVjl_L = num_L/den_L
            delta_R = abs(vjl_R-newVjl_R)
            delta_L = abs(vjl_L-newVjl_L)
            if (delta_R < self.stopDelta and delta_L < self.stopDelta):
                stop = True
            else:
                vjl_R = newVjl_R
                vjl_L = newVjl_L
        
        return j, l, UL, UR, vjl_R, vjl_L

    # Use Karnik-Mendel iterative algorithm to find cluser interval centers
    def calculate_cluster_bounds(self, data, u_l, u_u):
        start = time.perf_counter()
        n,f = data.shape
        c = u_l.shape[1]
        U = (u_l + u_u)/2.0
        UL = np.zeros(U.shape)
        UR = np.zeros(U.shape)
        VL = np.zeros((c,f))
        VR = np.zeros((c,f))

        # Calculate interval type 1 cluster centroids
        centroids = np.zeros([c,f])
        for j in range(c):
            num = np.sum([U[i,j]*data[i,:] for i in range(n)], axis=0)
            den = np.sum([U[i,j] for i in range(n)], axis=0)
            centroids[j,:] = num/den

        # Sort each feature of data in ascending order
        sorted_data_ids = np.argsort(data, kind ='mergesort', axis = 0)
        dtos = []
        for j in range(c):
            for l in range(f):
                vjl_R = centroids[j,l]
                vjl_L = centroids[j,l]
                dataPts = data[:,l]
                dataIds = sorted_data_ids[:,l]

                dtos.append( {
                    'sorted_data_ids': sorted_data_ids,
                    'dataPts': dataPts,
                    'dataIds': dataIds,
                    'vjl_R': vjl_R,
                    'vjl_L': vjl_L,
                    'u_u': u_u[:,j],
                    'u_l': u_l[:,j],
                    'n': n,
                    'j': j,
                    'l': l
                })
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            pool = executor.map(self.update_center, dtos)
            for (j, l, ULj, URj, vjl_R, vjl_L) in pool:
                VR[j,l] = vjl_R
                VL[j,l] = vjl_L
                UR[:,j] = URj
                UL[:,j] = ULj
        end = time.perf_counter()
        print(f'Calculated centers in {round(end-start, 2)} second(s)') 

        return UL,UR,VL,VR

    # Calculate magnitude of difference vector between old and new cluster centers
    def get_delta(self, oldV, V):
        return np.linalg.norm(oldV-V)

    # Get list of cluster labels (nx1 matrix)
    def get_data_labels(self, U):
        return np.array([np.argmax(ux) for ux in U])

    # Get data separated into clusters (returns list of length c where each element
    # is an array of points in that cluster)
    def get_cluster_data(self, c, labels, data):
        cluster_data = []
        for i in range(c):
            indices = np.where(labels == i)
            pts = np.concatenate([data[x,:] for x in indices],axis=0)
            cluster_data.append(pts)
        return cluster_data

    def run(self,data, c):
        # Get num points & num features
        n,f = data.shape 
        # Randomly generate cluster centers within the range of the given data (cxf matrix)
        V = self.initialize_cluster_centers(data,c,f)

        # Initialize iteration parameters
        oldV = np.ones(V.shape)*sys.maxsize
        delta = self.get_delta(oldV,V)
        iter = 0

        start = time.time()
        # Iterate to minimize objective cost function
        while delta > self.stopDelta and iter < self.maxIters:
            if iter%4 == 0:
                end = time.time()
                print(f'Iteration: {iter} , total time: {end-start} seconds ({(end-start)/60} minutes)')
            oldV = V
            # Calculate new distances to clusters
            startdist = time.time()
            D_l,D_u = self.calculate_distances(data,V)
            enddist = time.time()
            print(f'Calculated distances in: {enddist-startdist} seconds ({(enddist-startdist)/60} minutes)')
            # Calculate upper/lower membership grades (nxc matrices)
            startdist = time.time()
            u_l, u_u = self.calculate_membership_grades(D_l,D_u)
            enddist = time.time()
            print(f'Calculated membership grades in: {enddist-startdist} seconds ({(enddist-startdist)/60} minutes)')
            # Calculate new cluster center intervals
            UL,UR,VL,VR = self.calculate_cluster_bounds(data, u_l, u_u)
            # Defuzzify cluster centers
            V = (VL+VR)/2.0

            # Calculate delta
            delta = self.get_delta(oldV,V)
            # Increment counter
            iter += 1
        
        print(f'Final Iteration: {iter}')

        # Defuzzify membership grades
        U = (UL + UR)/2.0

        # Get list of cluster labels (nx1 matrix)
        labels = self.get_data_labels(U)
        # Get data separated into clusters
        cluster_data = self.get_cluster_data(c, labels, data)

        # Return cluster centers, membership grades, cluster data, and data labels
        return V, U, cluster_data, labels

class ImageSegmentation:
    # Save the given image (2D array) to the specified filename in the current directory
    def saveImage(self,image,filename):
        result = cv2.imwrite(dir_path+'/images/results/'+filename,image)
        if result==True:
            print('File saved successfully')
        else:
            print('Error in saving file')

    # Save and display the given ground truth image
    # in_path: the path with filename to a .mat file (from BSDS)
    #  out_filename: the filename used to save the ground truth image
    def getGroundTruthImage(self, in_path, out_filename):
        file = io.loadmat(in_path)
        segmap = file['groundTruth'][0][0][0][0][0]
        image = segmap.astype(np.float64)
        hi = np.max(image)
        lo = np.min(image)
        image = (((image - lo)/(hi-lo))*255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.saveImage(image,out_filename)
        plt.imshow(image)
        plt.show()

    # Read an image file from the given path
    # Returns a 2D array of pixel values to be used for clustering
    def getPreparedImageData(self, path):
        # Read image file
        image = cv2.imread(path)
        # Convert color channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image.shape)
        # Reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values = image.reshape((-1, 3))
        # Convert values to float
        pixel_values = np.float32(pixel_values)
        return image, pixel_values

    # Runs the given clustering algorithms on the given original image file
    # Saves and displays the segmented image after clustering if showImages or saveImage is true
    def generateProccessedImage(self, numClusters, in_path, algos=[], showImages=False, saveImage=False, out_filename=None):
        # Get image data
        image, image_data = self.getPreparedImageData(in_path)

        # Run Clustering Algorithms
        fcm = FCM()
        it2fcm = IT2FCM()
        cpit2fcm = CPIT2FCM()
        for a in algos:
            print(f'Algorithm: {a.name}')
            start = time.time()
            if a == Algorithm.FCM:
                V,U,cdata,clabels = fcm.run(image_data,numClusters)
            elif a == Algorithm.IT2FCM:
                V,U,cdata,clabels = it2fcm.run(image_data,numClusters)
            elif a == Algorithm.CPIT2FCM:
                V,U,cdata,clabels = cpit2fcm.run(image_data,numClusters)
            elif a == Algorithm.KMeans:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 10e-4)
                _,clabels,V = cv2.kmeans(image_data,numClusters, None, criteria, 300, cv2.KMEANS_RANDOM_CENTERS)
            end = time.time()
            print(f'{a} time elapsed: {end-start} seconds ({(end-start)/60} minutes)')

            # Generate image from cluster data
            centers = np.uint8(V)
            labels = clabels.astype(int)
            print(f'Unique cluster labels: {np.unique(labels)}')
            labels = labels.flatten()
            segImg = centers[labels.flatten()]
            segImg = segImg.reshape(image.shape)
            segImg = cv2.cvtColor(segImg, cv2.COLOR_RGB2BGR)

            # Save image
            if saveImage==True and out_filename != None:
                self.saveImage(segImg,f'{out_filename}_{a.name}_{int(end)}.png')

            # Show image
            if showImages:
                cv2.imshow(out_filename, segImg)
                cv2.waitKey(0)

def run_graphical_experiments():
    # Generate datasets
    c1,c2 = Datasets.get_square_clusters()
    squareData = np.concatenate((c1,c2),axis=0)
    c1,c2,c3 = Datasets.get_sphere_clusters()
    sphereData = np.concatenate((c1,c2,c3),axis=0)
    c1,c2 = Datasets.get_rod_clusters(2)
    rodData2 = np.concatenate((c1,c2),axis=0)
    c1,c2 = Datasets.get_rod_clusters(2.5)
    rodData25 = np.concatenate((c1,c2),axis=0)
    c1,c2 = Datasets.get_rod_clusters(3)
    rodData3 = np.concatenate((c1,c2),axis=0)

    squares = {
        'name': 'squares',
        'data': squareData,
        'C': 2,
        'xlim': [0,7],
        'ylim': [0,4],
        'yticks': range(0,5)
    }
    spheres = {
        'name': 'spheres',
        'data': sphereData,
        'C': 3,
        'xlim': [-15,15],
        'ylim': [-2,6]
    }
    rods2 = {
        'name': 'rods2',
        'data': rodData2,
        'C': 2,
        'xlim': [5,17],
        'ylim': [5,17],
        'xticks': range(5,18,2),
        'yticks': range(5,18,2)
    }
    rods25 = {
        'name': 'rods25',
        'data': rodData25,
        'C': 2,
        'xlim': [5,17],
        'ylim': [5,17],
        'xticks': range(5,18,2),
        'yticks': range(5,18,2)
    }
    rods3 = {
        'name': 'rods3',
        'data': rodData3,
        'C': 2,
        'xlim': [5,17],
        'ylim': [5,17],
        'xticks': range(5,18,2),
        'yticks': range(5,18,2)
    }

    datasets = [squares, spheres, rods2, rods25, rods3]
    algos = [Algorithm.FCM, Algorithm.IT2FCM, Algorithm.CPIT2FCM, Algorithm.KMeans]

    fcm = FCM()
    it2fcm = IT2FCM()
    cpit2fcm = CPIT2FCM()

    # Run all algorithms on all graphical datasets
    for dataset in datasets:
        print(f'Dataset: {dataset["name"]}')
        for a in algos:
            print(f'Algorithm: {a.name}')
            start = time.time()
            if a == Algorithm.FCM:
                V,U,cdata,clabels = fcm.run(dataset['data'],dataset['C'])
            elif a == Algorithm.IT2FCM:
                V,U,cdata,clabels = it2fcm.run(dataset['data'],dataset['C'])
            elif a == Algorithm.CPIT2FCM:
                V,U,cdata,clabels = cpit2fcm.run(dataset['data'],dataset['C'])
            elif a == Algorithm.KMeans:
                d = dataset['data']
                d = np.float32(d)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 10e-4)
                _,clabels,V = cv2.kmeans(d,dataset['C'], None, criteria, 300, cv2.KMEANS_RANDOM_CENTERS)
                cdata = []
                for i in range(dataset['C']):
                    indices = np.where(clabels == i)
                    pts = np.concatenate([d[x,:] for x in indices],axis=0)
                    cdata.append(pts)
            end = time.time()
            print(f'{a} time elapsed: {end-start} seconds ({(end-start)/60} minutes)')
    
            # Plot data points
            for idx,c in enumerate(cdata):
                plt.scatter(c[:,0],c[:,1], marker='.', label=f'Data{idx+1}')
            # Plot cluster centers
            plt.scatter(V[:,0],V[:,1], marker='*', label='Centers')
            plt.legend()
            plt.title(f'{a.name} - {dataset["name"]}')
            if 'xlim' in dataset:
                plt.xlim(dataset['xlim'])
            if 'ylim' in dataset:
                plt.ylim(dataset['ylim'])
            if 'xticks' in dataset:
                plt.xticks(dataset['xticks'])
            if 'yticks' in dataset:
                plt.yticks(dataset['yticks'])
            plt.savefig(dir_path+'/images/results/'+f'{dataset["name"]}_{a.name}_{int(end)}.png')
            # plt.show()
            plt.close()

def run_single_image(img):
    algos = [Algorithm.FCM, Algorithm.IT2FCM, Algorithm.CPIT2FCM, Algorithm.KMeans]
    
    seg = ImageSegmentation()
    print(f'Image: {img["name"]}')
    seg.generateProccessedImage( numClusters=img['C'], 
        in_path=img['image'], 
        algos=algos, 
        showImages=False, 
        saveImage=True, 
        out_filename=img['name'])
    return f'Done {img["name"]}'

def run_image_experiments():
    # Berkley Segmentation Datasets (BSDS) #100075, #3096, #41004, #60079, #100080
    # https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
    bear = {
        'name':'bear',
        'image':'images/100075.jpg',
        'truth':'images/100075.mat',
        'C': 2
    }
    plane = {
        'name':'plane',
        'image':'images/3096.jpg',
        'truth':'images/3096.mat',
        'C': 2
    }
    moose = {
        'name':'moose',
        'image':'images/41004.jpg',
        'truth':'images/41004.mat',
        'C': 3
    }
    parachute = {
        'name':'parachute',
        'image':'images/60079.jpg',
        'truth':'images/60079.mat',
        'C': 3
    }
    bearSit = {
        'name':'bearSit',
        'image':'images/100080.jpg',
        'truth':'images/100080.mat',
        'C': 4
    }

    images = [bear, plane, moose, parachute, bearSit]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        pool = executor.map(run_single_image, images)
        for res in pool:
            print(res)

def run_all_experiments():
    start = time.time()
    run_graphical_experiments()
    endg = time.time()
    print(f'Graphical Experiments time elapsed: {endg-start} seconds ({(endg-start)/60} minutes)\n')
    starti = time.time()
    run_image_experiments()
    end = time.time()
    print(f'Image Experiments time elapsed: {end-starti} seconds ({(end-starti)/60} minutes)\n')
    print(f'Total time elapsed: {end-start} seconds ({(end-start)/60} minutes)')

if __name__ == '__main__':
    run_all_experiments()
