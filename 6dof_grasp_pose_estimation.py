import open3d as o3d
from open3d import *
import numpy as np
import random
import math
import matplotlib as plt
import copy
import pickle
import glob
import os

# Versions 
print("opend3D version: ", open3d.__version__)
print("numpy version: ", np.__version__)
print("matplotlib version: ", plt.__version__)

# Parameters
point_step_size=1
down_sample_factor =20
colliding_points_allowed = 1
degree_step_size = 30
grasp_width_initial=0.02
grasp_width_step=0.02
grasp_width_final= 0.06


# segments and removes planar surface using RANSAC
def remove_plane(pc_points,visualize):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.007, # fixed parameters
                                            ransac_n=3,
                                            num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    #inlier means points on the plane
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    if visualize:
        o3d.visualization.draw_geometries([outlier_cloud, inlier_cloud])
        o3d.visualization.draw_geometries([outlier_cloud])

    #outlier_pc = np.asarray(outlier_cloud.points)

    return outlier_cloud


def backproject(depth_cv, intrinsic_matrix, return_finite_depth=True, return_selection=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X



def getPointCloud():  

    list_of_files = glob.glob('/home/panda_lmt_furkan/locomo_project/*npy') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    data = np.load(latest_file, allow_pickle=True, encoding = 'latin1')
    keys = []
    if len(data.shape) == 0:
        data = data.item()
        keys = data.keys()

    depth = data['depth']
    cam_K = data['K'].reshape(3,3)
    rgb = data['rgb']


    # Removing points that are farther than 1 meter or missing depth
    # values.
    #depth[depth ==0] = np.nan
    #depth[depth > 1.2] = np.nan 
    return_selection=True
    if return_selection:
        pc, selection = backproject(depth, cam_K, return_finite_depth=True, return_selection=True)
    else:
        pc = backproject(depth, cam_K, return_finite_depth=True, return_selection=False)
    pc_colors = rgb.copy()
    pc_colors = np.reshape(pc_colors[:,:,:3], [-1, 3])/256.0
    pc_colors = pc_colors[selection, :]

    #red=pc_colors[:,0]
    #blue=pc_colors[:,1]
    #pc_colors[:,0]=blue
    #pc_colors[:,1]=red
    ### furkend

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors =  o3d.utility.Vector3dVector(pc_colors)
    
    return pcd


# Upload Point cloud

pcd = getPointCloud()

print('PLY file loaded')
print('Shape of points', np.asarray(pcd.points).shape)
print('Shape of colors', np.asarray(pcd.colors).shape)

# save points of the pcd as numpy arrray
points = np.asarray(pcd.points)
# crop pcd
bb= o3d.geometry.OrientedBoundingBox()
# location of the bb 
# x = goes to the right as it increases, y = goes up as it decreases, z = goes forward as it decreases
tr_bb=np.asarray([-0.2,0.1,0.88])

rot_bb=np.zeros([3,3])
rot_bb = np.asarray(np.matmul([[ 1.0000000, 0.0000000, 0.0000000],
                                [0.0000000, 0.5000000, 0.8660254],
                                [0.0000000, -0.8660254, 0.5000000 ]], 

                                [[0.6533192, -0.7738907, 0.0000000],
                                [0.7738907, 0.6533192, 0.0000000],
                                [0.0000000, 0.0000000, 1.0000000 ]]))

bb.center=tr_bb
bb.R=rot_bb
# size of the bb
bb.extent=(0.6,0.6,0.6) 
bb.color = (0, 0, 0)

#o3d.visualization.draw_geometries([pcd,bb])

pcd=pcd.crop(bb)

# Remove ground surface plane
cropped_pcd_without_plane = remove_plane(copy.deepcopy(pcd),visualize=True)
# Define a smaller ROI pcd:
bb.extent=(0.32,0.33,0.5) 
downpcd_narrower=cropped_pcd_without_plane.crop(bb)

# downsample using farthest points
downpcd_narrower = downpcd_narrower.farthest_point_down_sample(len(downpcd_narrower.points)//down_sample_factor)
print("number of points to visit: "+str(len(downpcd_narrower.points)/point_step_size))
o3d.visualization.draw_geometries([downpcd_narrower],point_show_normal=True)

 

# Estimate the surface normals for each point
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=12))
#pcd.orient_normals_consistent_tangent_plane(k=12)

# to obtain a consistent normal orientation
#pcd.orient_normals_towards_camera_location(pcd.get_center())
pcd.orient_normals_towards_camera_location(camera_location= ([0., 0., 0.]))
# flip the normals to make them point outward
pcd.normals = o3d.utility.Vector3dVector( - np.asarray(pcd.normals))

normals = pcd.normals

o3d.visualization.draw_geometries([downpcd_narrower],point_show_normal=True)


# Gripper class and Grasp configurations class
class Grasp:
    grasp_position=None
    grasp_position_ind=None
    grasp_orientation=None
    grasp_width = None
    grasp_GScore_score = None
    def __init__(self,grasp_position, grasp_position_ind, grasp_orientation, grasp_width, grasp_GScore_score):
        self.grasp_position=grasp_position
        self.grasp_position_ind=grasp_position_ind
        self.grasp_orientation=grasp_orientation
        self.grasp_width=grasp_width
        self.grasp_GScore_score=grasp_GScore_score

class Gripper: 
    cord_mesh = None
    finger_width = None
    finger_height = None
    grasp_width = None
    grasp_point = None
    gripper_body_bbox = None
    finger1bbox = None
    finger2bbox = None    
    finger_body1bbox = None
    finger_body2bbox = None
    

    def rotate_around_x(degree):
         # convert to radian 
        radian = (degree/180)*math.pi

        return np.array([[1,0,0],[0, np.cos(radian),-np.sin(radian)],[0,np.sin(radian),np.cos(radian)]])


    def rotate_gripper(self, desired_normal,degree):

        self.f1normal, self.f2normal = self.get_normals()
        # Angle between gripper normal and the desired normal
        axis = np.cross(self.f1normal, desired_normal) 
        # Normalize the rotation axis
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(self.f1normal, desired_normal) / (np.linalg.norm(self.f1normal) * np.linalg.norm(desired_normal)))
        rotation_axis_angle = np.array([axis[0], axis[1], axis[2]]) * angle
        # Compute the rotation matrix to align the normals
        rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis_angle)

        # matrix for rotating around x axis
        radian = (degree/180)*math.pi
        rot_x=np.array([[1,0,0],[0, np.cos(radian),-np.sin(radian)],[0,np.sin(radian),np.cos(radian)]])
        # align and rotate
        rot_mat= np.matmul(rot_mat,rot_x)

        # Apply rotattion matrix
        self.finger1bbox.rotate(rot_mat,center=self.grasp_point)
        self.finger2bbox.rotate(rot_mat,center=self.grasp_point)
        self.finger_body1bbox.rotate(rot_mat,center=self.grasp_point)
        self.finger_body2bbox.rotate(rot_mat,center=self.grasp_point)
        
        self.gripper_body_bbox.rotate(rot_mat,center=self.grasp_point)
        return rot_mat

    # Calculate the normal vector
    def get_normals(self):

        # Get initial finger normals:
        f1c=self.finger1bbox.get_center()
        f2c=self.finger2bbox.get_center()
        
        f1normal = f2c-f1c
        f1normal /= np.linalg.norm(f1normal)

        f2normal = -f1normal
        
        return f1normal, f2normal 
 
    def __init__(self, x, y, z, o):

        self.cord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin=[0,0,0])

        # Gripper size 
        self.finger_width=0.0175 
        self.finger_height=0.0175 
        self.grasp_width= o 

        # initial position of the gripper on pcd
        self.grasp_point = np.asarray([x,y,z]) 

        # Gripper body 
        gripper_body_width,gripper_body_depth,gripper_body_height=0.21,0.04,0.08
        finger_body_width,finger_body_depth,finger_body_height=0.0075,0.0175,0.0475 #0.0175,0.0175,0.0475
        self.gripper_body_bbox=o3d.geometry.OrientedBoundingBox(self.grasp_point-[0,0,gripper_body_height/2+self.finger_height/2+0.03], np.eye(3), [gripper_body_width,gripper_body_depth,gripper_body_height])

        # Gripper finger boxes
        self.finger1bbox=o3d.geometry.OrientedBoundingBox(self.grasp_point-[self.grasp_width/4,0,0], np.eye(3), [self.grasp_width/2,self.finger_width,self.finger_height])
        self.finger2bbox=o3d.geometry.OrientedBoundingBox(self.grasp_point+[self.grasp_width/4,0,0], np.eye(3), [self.grasp_width/2,self.finger_width,self.finger_height])
        # Gripper finger body boxes
        self.finger_body1bbox=o3d.geometry.OrientedBoundingBox(self.grasp_point+[-self.grasp_width*3/4,0,-(finger_body_height-self.finger_height)/2], np.eye(3), [finger_body_width,finger_body_depth,finger_body_height])
        self.finger_body2bbox=o3d.geometry.OrientedBoundingBox(self.grasp_point+[self.grasp_width*3/4,0,-(finger_body_height-self.finger_height)/2], np.eye(3), [finger_body_width,finger_body_depth,finger_body_height])

# Get Points Inside Gripper
def get_points_inside_bounding_box(pcd,finger1bbox, finger2bbox=None):
    
    # Get the indices of the points within the bounding box
    indices1 = finger1bbox.get_point_indices_within_bounding_box(pcd.points)
    # Get the points inside the boxes
    indicesInsideBox1 = indices1

    pointsInsideBox1 = []
    pointsInsideBoxes = []

    for p in indicesInsideBox1: 
        pointsInsideBox1.append(pcd.points[p])
        
    if finger2bbox:
        indices2 = finger2bbox.get_point_indices_within_bounding_box(pcd.points)
        indicesInsideBox2 = indices2
        pointsInsideBox2 = []
        for p in indicesInsideBox2: 
            pointsInsideBox2.append(pcd.points[p])
        
        pointsInsideBoxes = pointsInsideBox1 + pointsInsideBox2

    else: 
        pointsInsideBoxes = pointsInsideBox1
        pointsInsideBox2 = []
        indicesInsideBox2 = []

    return pointsInsideBoxes, indicesInsideBox1, indicesInsideBox2


# Compute Grasp Score

def compute_GScore(indicesInsideBox1, indicesInsideBox2, f1normal, f2normal):
    GScores = []
    GScores_1 =  []
    GScores_2 =  []

    val_max = 1 / math.sqrt( (math.pow(2*math.pi,3)*np.linalg.det(np.eye(3)) ) ) # error = 0 --> e^0 = 1 
    for each in indicesInsideBox1:
        error = normals[each] + f1normal
        #print (error)
        #GScore calculate
        val = val_max*math.exp(-0.5*np.dot(np.dot((error),np.eye(3)), (error)))

        #normalize GScore value (val-min)/(max-min) :: max=0.063493 min=0
        GScore = val / val_max
        GScores_1.append(GScore)
        
    for each in indicesInsideBox2:
        error = normals[each] + f2normal
        #GScore calculate
        val = val_max*math.exp(-0.5*np.dot(np.dot((error),np.eye(3)), (error)))
        GScore = val / val_max
        GScores_2.append(GScore)

    GScore_values = np.append(GScores_1,GScores_2)
    GScore_of_the_pose = np.sum(GScore_values)/len(GScore_values)

    return GScore_of_the_pose

gist_rainbow = plt.colormaps['gist_rainbow']

GScores_of_Poses = []
point_indices_in_rotated_fingers = []
colliding_grasps=[]
noncolliding_grasps=[]
non_colliding_grasp_grippers=[]
pcd_colors = np.zeros((len(pcd.points), 3))

vis_list=[pcd]


# Grasp pose sampling

for i, position_ind in enumerate(range(0,len(np.array(downpcd_narrower.points)),point_step_size)): # i is for gripper index. position_ind = step_size*i

    if int(np.floor(100*i/len(np.array(downpcd_narrower.points))*point_step_size)) % 10==0:
        print("At position: % " + str(100*i/len(np.array(downpcd_narrower.points))*point_step_size))
    #if position_ind >
    for grasp_width in np.arange(grasp_width_initial,grasp_width_final,grasp_width_step):
        
        #gripper_position_array.append
        gripper=Gripper( np.asarray(downpcd_narrower.points)[position_ind][0],
                                            np.asarray(downpcd_narrower.points)[position_ind][1],
                                            np.asarray(downpcd_narrower.points)[position_ind][2], grasp_width)

        # get points inside boxes
        pointsInsideBoxes, indicesInsideBox1, indicesInsideBox2 = get_points_inside_bounding_box(pcd,gripper.finger1bbox, gripper.finger2bbox)
        
        # compute desired normal inside box
        

        normals_in_box = []
        _arr_normals = np.asarray(normals)
        all_indices = indicesInsideBox1
        if len(all_indices) > 15:
            for idx in all_indices:
                normals_in_box.append(_arr_normals[idx])
            # Calculate the average of normal vectors
            average_normal = np.mean(normals_in_box, axis=0)
            desired_normal = -1*average_normal

        
            # Apply the rotation matrix to the bounding box
            for degree in range (0,360, degree_step_size):

                gripper_xrotated = copy.deepcopy(gripper)
                grasp_rot_mat= gripper_xrotated.rotate_gripper(desired_normal,degree)


                # get normals to compute GScore
                f1normal_rotated, f2normal_rotated = gripper_xrotated.get_normals()

                pointsInsideBoxes, indicesInsideRotatedBox1, indicesInsideRotatedBox2 = get_points_inside_bounding_box(pcd,gripper_xrotated.finger1bbox, gripper_xrotated.finger2bbox)
                pointsInsideBody, _, __ = get_points_inside_bounding_box(pcd,gripper_xrotated.gripper_body_bbox)
                pointsInsideFingerBody1, _, __ = get_points_inside_bounding_box(pcd,gripper_xrotated.finger_body1bbox)
                pointsInsideFingerBody2, _, __ = get_points_inside_bounding_box(pcd,gripper_xrotated.finger_body2bbox)


                if len(pointsInsideBody+pointsInsideFingerBody1+pointsInsideFingerBody2)<colliding_points_allowed:
                    print("grasp found")
                    grasp_position = [np.asarray(downpcd_narrower.points)[position_ind][0],
                                                    np.asarray(downpcd_narrower.points)[position_ind][1],
                                                    np.asarray(downpcd_narrower.points)[position_ind][2]]
                    grasp_position_ind = position_ind
                    grasp_orientation = grasp_rot_mat
                    grasp_GScore_score=  compute_GScore(indicesInsideRotatedBox1, indicesInsideRotatedBox2, f1normal_rotated, f2normal_rotated)
                    
                    noncolliding_grasps.append(Grasp(grasp_position, grasp_position_ind,grasp_orientation, grasp_width, grasp_GScore_score))
                    rgb=gist_rainbow(np.asarray(grasp_GScore_score/2))[:3]
                    gripper_xrotated.gripper_body_bbox.color =rgb
                    gripper_xrotated.finger1bbox.color=rgb 
                    gripper_xrotated.finger2bbox.color=rgb
                    gripper_xrotated.finger_body1bbox.color=rgb 
                    gripper_xrotated.finger_body2bbox.color=rgb
                    pcd_colors[grasp_position_ind]=rgb
                    non_colliding_grasp_grippers.append(gripper_xrotated)
                    vis_list.append(gripper_xrotated.finger1bbox)
                    vis_list.append(gripper_xrotated.finger2bbox)
                    vis_list.append(gripper_xrotated.finger_body1bbox)
                    vis_list.append(gripper_xrotated.finger_body2bbox)
                    vis_list.append(gripper_xrotated.gripper_body_bbox)


o3d.visualization.draw_geometries(vis_list)

from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
now=now.strftime("%m-%d-%Y,%H-%M-%S")

#o3d.visualization.draw_geometries([pcd,non_colliding_grasp_grippers[2].finger1bbox,non_colliding_grasp_grippers[2].finger2bbox,non_colliding_grasp_grippers[2].gripper_body_bbox,non_colliding_grasp_grippers[0].finger1bbox,non_colliding_grasp_grippers[0].finger2bbox,non_colliding_grasp_grippers[0].gripper_body_bbox,non_colliding_grasp_grippers[1].finger1bbox,non_colliding_grasp_grippers[1].finger2bbox,non_colliding_grasp_grippers[1].gripper_body_bbox])
#non_colliding_grasp_grippers[0]
np.save("/home/panda_lmt_furkan/locomo_project/results/noncolliding_grasp_list_"+ now +".npy", noncolliding_grasps)


k = 10

GScores_of_Poses = []

for each in range(len(noncolliding_grasps)):

    GScores_of_Poses.append(noncolliding_grasps[each].grasp_GScore_score)


# Getting indices of k max values

highest_GScores_idx = np.flip( np.argsort(GScores_of_Poses)[-k:])

print("Highest Grasp Indices:",highest_GScores_idx)

GScores_of_Poses = np.array(GScores_of_Poses)

print("Highest Grasp Scores:" ,GScores_of_Poses[highest_GScores_idx])

vis_list_highest_k=[pcd]
best_grasps_list=[]
for each in range(len(highest_GScores_idx)):


    grasp=(noncolliding_grasps[highest_GScores_idx[each]])
    print(grasp.grasp_GScore_score)

    gripper_xrotated=Gripper(grasp.grasp_position[0], grasp.grasp_position[1],grasp.grasp_position[2], grasp_width)

    grasp_rot_mat=grasp.grasp_orientation
    gripper_xrotated.finger1bbox.rotate(grasp_rot_mat,center=grasp.grasp_position)
    gripper_xrotated.finger2bbox.rotate(grasp_rot_mat,center=grasp.grasp_position)
    gripper_xrotated.finger_body1bbox.rotate(grasp_rot_mat,center=grasp.grasp_position)
    gripper_xrotated.finger_body2bbox.rotate(grasp_rot_mat,center=grasp.grasp_position)
    gripper_xrotated.gripper_body_bbox.rotate(grasp_rot_mat,center=grasp.grasp_position)



    rgb=gist_rainbow(np.asarray(grasp.grasp_GScore_score/2))[:3]
    gripper_xrotated.gripper_body_bbox.color =rgb
    gripper_xrotated.finger1bbox.color=rgb 
    gripper_xrotated.finger2bbox.color=rgb
    gripper_xrotated.finger_body1bbox.color=rgb 
    gripper_xrotated.finger_body2bbox.color=rgb
    #pcd_colors[grasp_position_ind]=rgb
    #non_colliding_grasp_grippers.append(gripper_xrotated)
    vis_list_highest_k.append(gripper_xrotated.finger1bbox)
    vis_list_highest_k.append(gripper_xrotated.finger2bbox)
    vis_list_highest_k.append(gripper_xrotated.finger_body1bbox)
    vis_list_highest_k.append(gripper_xrotated.finger_body2bbox)
    vis_list_highest_k.append(gripper_xrotated.gripper_body_bbox)
    grasp_array = [grasp.grasp_position,grasp.grasp_orientation,grasp.grasp_width,grasp.grasp_GScore_score]
    best_grasps_list.append(grasp_array)
    

with open('/home/panda_lmt_furkan/locomo_project/results/best_grasp_list_'+ now, 'wb') as f:
    pickle.dump(best_grasps_list,f)

    
o3d.visualization.draw_geometries(vis_list_highest_k)

x=5
