import numpy as np
from scipy.spatial import distance
from plyfile import PlyData, PlyElement


def open_ply(file_path):
    rdata = PlyData.read(file_path)
    points = []
    for i in range(len(rdata.elements[0].data)):
        point = rdata.elements[0].data[i]
        a = np.array(list(point))
        points.append(a)
    data = np.array(points)
    return data


def write_ply(name, data):
    tuples = []
    for point_i in range(data.shape[0]):
        tuples.append(tuple(data[point_i, :9]))

    described_data = np.array(
        tuples,
        dtype=[
            ("x", "double"),
            ("y", "double"),
            ("z", "double"),
            ("nx", "double"),
            ("ny", "double"),
            ("nz", "double"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    element = PlyElement.describe(described_data, "vertex")
    PlyData([element], text=False).write(name)


def main():
    #read in point cloud
    points = open_ply("data/PointCloud.ply")
    
    #identify black(noise) points in original point cloud
    noise_indices = np.where((points[:,-1]==0) & (points[:,-2]==0) & (points[:,-3]==0))[0]
    noise_points = points[noise_indices]
    
    #identify colored points in original point cloud
    colored_indices = np.setdiff1d(np.arange(points.shape[0]),noise_indices)
    colored_points = points[colored_indices]
    
    #output .ply file for all colored points before filter is applied for visual inspection
    write_ply("data/unfiltered_colors.ply", colored_points)
    
    #identify the set of unique, non black colors in the point cloud 
    unique_colors = np.unique(colored_points[:,-3:],axis=0)
    #separate the point cloud by color
    unique_colored_points = [colored_points[(colored_points[:,-3]==color[0]) & (colored_points[:,-2]==color[1]) & (colored_points[:,-1]==color[2])] for color in unique_colors]
    
    filtered_points = [] #variable for collecting points that pass the filter
    noise_points = [noise_points] #variable for collecting points that fail filter and become noise
    for color,points in zip(unique_colors,unique_colored_points): #filter point cloud by color
        
        #output .ply file for each unfiltered color segment for visual inspection
        write_ply("data/segment"+str(color)+".ply", points)
        
        #mean of each normal component
        mean_normal = np.mean(points[:,3:6],axis=0)
    
        #covariance matrix of the normal vectors
        S = np.cov(points[:,3:6].T)
        #inverse of S
        S_inv = np.linalg.inv(S)
        #calculate Mahalanobis distance of each normal vector, multivariate generalization of mean +/- standard deviation
        Mahalanobis_distances = np.array([distance.mahalanobis(normal,mean_normal,S_inv) for normal in points[:,3:6]])
        
        #locate points that pass the filter
        filtered_indices = np.where(Mahalanobis_distances<2)[0]
        #locate points that are outliers
        noise_indices = np.setdiff1d(np.arange(points.shape[0]),filtered_indices)
        
        #output .ply file for each filtered color segment for visual inspection
        write_ply("data/filtered_segment"+str(color)+".ply", points[filtered_indices])
        
        #collect inliers and outliers
        filtered_points.append(points[filtered_indices])
        noise_points.append(points[noise_indices])
        
    #reshape final inliers and outliers
    filtered_points = np.concatenate(filtered_points)
    noise_points = np.concatenate(noise_points)
    
    #output .ply file of all inlier points and outlier points for visual inspection
    write_ply("data/filtered_points.ply", filtered_points)
    write_ply("data/noise_points.ply", noise_points)
    
    #set new outlier colors to black
    noise_points[:,-3:] = 0
    #combine all inlier and outlier points
    all_points = np.append(filtered_points,noise_points,axis=0)
    
    #output .ply file of combined inlier and outlier points for visual inspection
    write_ply("data/Result.ply", all_points)
    
    return 0


if __name__ == "__main__":
    main()
