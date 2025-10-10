import open3d as o3d

print ("hello world ...")

pcd = o3d.geometry.PointCloud()

print ("hello world 1")
# crash due to libomp.
# brew install libomp
# maybe:
# export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:${DYLD_LIBRARY_PATH}"


# Define a callback function for a key press
def change_color(vis):
    # Change the color of the point cloud
    pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    return False  # Return False to indicate no further updates are needed

def doOpen3d():
    print("Open3D function called")

    # Create a simple point cloud
    
    pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]])


    # Initialize the visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    # Register the callback for the 'C' key (ASCII code 67)
    vis.register_key_callback(67, change_color)

    print ("hello world 2")

    # Run the visualizer
    vis.run()
    vis.destroy_window()


print ("hello world 3")

doOpen3d()
print ("hello world 4")
doOpen3d()