import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image
import glob
import open3d as o3d
import blickfeld_scanner
from blickfeld_scanner.stream import point_cloud
from yolov4.tf import YOLOv4
import tensorflow as tf
import time
import statistics
import random
import os

class LiDAR2Camera(object):
    def __init__(self, calib_file):
        calibs = self.read_calib_file(calib_file)
        P = calibs["P0"]
        self.P = np.reshape(P, [3, 4])
        V2C = calibs["Tr_lidar_to_cam"]
        self.V2C = np.reshape(V2C, [3, 4])

    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    
def cart2hom(self, pts_3d):
    """ Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom
LiDAR2Camera.cart2hom = cart2hom

def project_lidar_to_image(self, pts_3d_lidar):
    '''
    Input: 3D points in Lidar Frame [nx3]
    Output: 2D Pixels in Image Frame [nx2]
    '''
    p_rt =  np.dot(self.P, np.vstack((self.V2C, [0, 0, 0, 1])))
    pts_3d_homo = np.column_stack([pts_3d_lidar, np.ones((pts_3d_lidar.shape[0],1))])
    p_rt_x = np.dot(p_rt, np.transpose(pts_3d_homo))
    pts_2d = np.transpose(p_rt_x)
    
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]
LiDAR2Camera.project_lidar_to_image = project_lidar_to_image

def show_lidar_on_image(self, pc_lidar, img):
    """ Project LiDAR points to image """

    pts_2d = self.project_lidar_to_image(pc_lidar)
    imgfov_pc_lidar=pc_lidar
    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    self.imgfov_pts_2d = pts_2d
    self.imgfov_pc_lidar = imgfov_pc_lidar
    for i in range(pts_2d.shape[0]):
        depth = imgfov_pc_lidar[i,1]
        color = cmap[int(240.0 / depth), :]
        cv2.circle(
            img,(int(np.round(pts_2d[i, 0])), int(np.round(pts_2d[i, 1]))),2,
            color=tuple(color),
            thickness=-1,
        )

    return img
LiDAR2Camera.show_lidar_on_image = show_lidar_on_image


def run_obstacle_detection(img):
    start_time=time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = yolo.resize_image(img)
    resized_image = resized_image / 255.
    input_data = resized_image[np.newaxis, ...].astype(np.float32)

    candidates = yolo.model.predict(input_data)
    _candidates = []
    result = img.copy()
    for candidate in candidates:
        batch_size = candidate.shape[0]
        grid_size = candidate.shape[1]
        _candidates.append(tf.reshape(candidate, shape=(1, grid_size * grid_size * 3, -1)))
        candidates = np.concatenate(_candidates, axis=1)
        pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0], iou_threshold=0.35, score_threshold=0.40)
        pred_bboxes = pred_bboxes[~(pred_bboxes==0).all(1)] 
        pred_bboxes = yolo.fit_pred_bboxes_to_original(pred_bboxes, img.shape)
        exec_time = time.time() - start_time
        result = yolo.draw_bboxes(img, pred_bboxes)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result, pred_bboxes


def rectContains(rect,pt, w, h, shrink_factor = 0):       
    x1 = int(rect[0]*w - rect[2]*w*0.5*(1-shrink_factor)) 
    y1 = int(rect[1]*h-rect[3]*h*0.5*(1-shrink_factor)) 
    x2 = int(rect[0]*w + rect[2]*w*0.5*(1-shrink_factor)) 
    y2 = int(rect[1]*h+rect[3]*h*0.5*(1-shrink_factor)) 
    
    return x1 < pt[0]<x2 and y1 <pt[1]<y2 


def filter_outliers(distances):
    inliers = []
    mu  = statistics.mean(distances)
    std = statistics.stdev(distances)
    for x in distances:
        if abs(x-mu) < std:
            inliers.append(x)
    return inliers


def get_best_distance(distances, technique="closest"):
    if technique == "closest":
        return min(distances)
    elif technique =="average":
        return statistics.mean(distances)
    elif technique == "random":
        return random.choice(distances)
    else:
        return statistics.median(sorted(distances))


def lidar_camera_fusion(self, pred_bboxes, image):
    img_bis = image.copy()
    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    distances = []
    for box in pred_bboxes:
        distances = []
        for i in range(self.imgfov_pts_2d.shape[0]):
            depth = self.imgfov_pc_lidar[i,1]
            if (rectContains(box, self.imgfov_pts_2d[i], image.shape[1], image.shape[0], shrink_factor=0.2)==True):
                distances.append(depth)
                color = cmap[int(240.0 / depth), :]
                cv2.circle(img_bis,(int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))),2,color=tuple(color),thickness=-1,)
        h, w, _ = img_bis.shape
        if (len(distances)>2):
            distances = filter_outliers(distances)
            best_distance = get_best_distance(distances, technique="average")
            cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (int(box[0]*w),int(box[1]*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3, cv2.LINE_AA)    
        distances_to_keep = []
    
    return img_bis, distances
LiDAR2Camera.lidar_camera_fusion = lidar_camera_fusion


def pipeline(self, image, point_cloud):
    "For a pair of 2 Calibrated Images"
    img = image.copy()
    self.show_lidar_on_image(point_cloud[:,:3], image)
    result, pred_bboxes = run_obstacle_detection(img)
    img_final, _ = self.lidar_camera_fusion(pred_bboxes, result)
    return img_final
LiDAR2Camera.pipeline = pipeline


def configure_point_cloud_stream(lidar_IP_adress):
    device = blickfeld_scanner.scanner(lidar_IP_adress)  
    reference_frame = point_cloud.REF_FRAME_XYZ
    stream = device.get_point_cloud_stream(filter=None, reference_frame=reference_frame,as_numpy=True)
    return stream

def load_yolo():
    global yolo,dir,lidar2cam
    dir = os.getcwd()
    yolo = YOLOv4(tiny=True)
    yolo.classes = dir+"/Yolov4/coco.names"
    yolo.make_model()
    yolo.load_weights(dir+"/Yolov4/yolov4-tiny.weights", weights_type="yolo")
    print("Yolo has been load...")





