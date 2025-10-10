import numpy as np, open3d as o3d

pcd = o3d.geometry.PointCloud()

pts = np.array([[0, 0, 0],
                [1, 0, 0],
                [0, 1, 0]], dtype=np.float64)
pts = np.ascontiguousarray(pts)

pcd.points = o3d.utility.Vector3dVector(pts)   # should NOT crash

print(pcd)