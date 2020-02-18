#!/usr/bin/env python


import rospy
import numpy as np 
import tf
import cv2

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import trifocal_tensor_algebra as tta
import riseq_perception.vision_utils as cvutils

class ImageProcessor():

    def __init__(self):

        self.cam0_sub = rospy.Subscriber("/pelican/camera_nadir/image_raw", Image, self.cam0_cb)
        self.cam1_sub = rospy.Subscriber("/pelican/camera_nadir2/image_raw", Image, self.cam1_cb)
        self.cam2_sub = rospy.Subscriber("/pelican/camera_nadir3/image_raw", Image, self.cam2_cb)

        self.cam0_sub = rospy.Subscriber("/pelican/camera_nadir/camera_info", CameraInfo, self.cam0_params_cb)
        self.cam1_sub = rospy.Subscriber("/pelican/camera_nadir2/camera_info", CameraInfo, self.cam1_params_cb)
        self.cam2_sub = rospy.Subscriber("/pelican/camera_nadir3/camera_info", CameraInfo, self.cam2_params_cb)

        self.pub = rospy.Publisher("/image_processing/process", PoseStamped, queue_size =10)
        self.cam0_pub = rospy.Publisher("/threeview/cam0", Image, queue_size = 2)
        self.cam1_pub = rospy.Publisher("/threeview/cam1", Image, queue_size = 2)
        self.cam2_pub = rospy.Publisher("/threeview/cam2", Image, queue_size = 2)


        self.cam0_img_n = 0
        self.cam1_img_n = 0
        self.cam2_img_n = 0

        self.cam0_img = None
        self.cam1_img = None
        self.cam2_img = None
        self.cam0_params = None
        self.cam1_params = None
        self.cam2_params = None

        self.R0 = np.diag([1.0,1.0,1.0])
        self.t0 = np.array([0.,0.,0.])
        self.K0 = None
        self.R1 = np.diag([1.0,1.0,1.0])
        self.t1 = np.array([-0.3,0.,0.])
        self.K1 = None
        self.R2 = np.diag([1.0, 1.0, 1.0])
        self.t2 = np.array([-0.6, 0., 0.])
        self.K2 = None
        self.P0 = np.hstack((self.R0, self.t0.reshape(3,1)))
        self.P1 = np.hstack((self.R1, self.t1.reshape(3,1)))
        self.P2 = np.hstack((self.R2, self.t2.reshape(3,1)))

        self.process_time = rospy.Timer(rospy.Duration(0.05), self.process)
        self.bridge = CvBridge()

        self.vue_low = 40
        self.saturation_low = 50
        self.blue_low = 93
        self.blue_high = 131
        # Update HSV Boundaries
        self.HSVboundaries = [ #([165, 100, 40], [180, 255, 255]), #red upper range
                               #([0, 100, 40], [15, 255, 255]),    # red lower range
                               #([40, 50, 40], [80, 255, 255]),  # green range
                               ([self.blue_low, self.saturation_low, self.vue_low],[self.blue_high, 255, 255])]  # blue range


    def process(self,msg):

        if ( (self.cam0_img_n is not None) and (self.cam1_img_n is not None) and (self.cam2_img_n is not None) and (self.K0 is not None) ):
            

            cam0_mask = cvutils.filterColors(self.cam0_img, self.HSVboundaries, "RGB")
            cam0_out = cv2.bitwise_and(self.cam0_img, self.cam0_img, mask=cam0_mask)
            cam1_mask = cvutils.filterColors(self.cam1_img, self.HSVboundaries, "RGB")
            cam1_out = cv2.bitwise_and(self.cam1_img, self.cam1_img, mask=cam1_mask)
            cam2_mask = cvutils.filterColors(self.cam2_img, self.HSVboundaries, "RGB")
            cam2_out = cv2.bitwise_and(self.cam2_img, self.cam2_img, mask=cam2_mask)
                
            mask = np.zeros(self.cam0_img.shape[:2], dtype="uint8")
            mask[cam0_mask > 0] = 255
            a = np.where(mask == 255)
            x1 = (min(a[1]),max(a[0]))
            x2 = (max(a[1]), min(a[0]))
            #cam0_out = cv2.line(cam0_out, x2, x1, (0,255,0), 2)
            cam0_out = cv2.circle(cam0_out, x1,2,(255,0,0),2)
            cam0_out = cv2.circle(cam0_out, x2,2,(255,0,0),2)
            l = tta.get_line_equation(x1,x2)
            ab = l[0]/l[1]
            cb = l[2]/l[1]
            x1 = (0, int(-ab*0. -cb))
            x2 = (752, int(-ab*752 -cb))
            cam0_out = cv2.line(cam0_out, x1, x2, (0,255,0), 2)


            mask = np.zeros(self.cam1_img.shape[:2], dtype="uint8")
            mask[cam1_mask > 0] = 255
            a = np.where(mask == 255)
            x1 = (min(a[1]),max(a[0]))
            x2 = (max(a[1]), min(a[0]))
            #cam1_out = cv2.line(cam1_out, x2, x1, (0,255,0), 2)
            cam1_out = cv2.circle(cam1_out, x1,2,(255,0,0),2)
            cam1_out = cv2.circle(cam1_out, x2,2,(255,0,0),2)
            l_p = tta.get_line_equation(x1,x2)
            ab = l_p[0]/l_p[1]
            cb = l_p[2]/l_p[1]
            x1 = (0, int(-ab*0. -cb))
            x2 = (752, int(-ab*752 -cb))
            cam1_out = cv2.line(cam1_out, x1, x2, (0,255,0), 2)


            mask = np.zeros(self.cam2_img.shape[:2], dtype="uint8")
            mask[cam2_mask > 0] = 255
            a = np.where(mask == 255)
            x1 = (min(a[1]),max(a[0]))
            x2 = (max(a[1]), min(a[0]))
            #cam2_out = cv2.line(cam2_out, x2, x1, (0,255,0), 2)
            cam2_out = cv2.circle(cam2_out, x1,2,(255,0,0),2)
            cam2_out = cv2.circle(cam2_out, x2,2,(255,0,0),2)
            l_pp = tta.get_line_equation(x1,x2)
            ab = l_pp[0]/l_pp[1]
            cb = l_pp[2]/l_pp[1]
            x1 = (0, int(-ab*0. -cb))
            x2 = (752, int(-ab*752 -cb))
            cam2_out = cv2.line(cam2_out, x1, x2, (0,255,0), 2)

            trifocal_tensor = tta.get_trifocal_tensor(self.P1, self.P2)
            l_projected = tta.transport_line(l_p.reshape(3,1), l_pp.reshape(3,1), trifocal_tensor)
            print(l_projected)


            img_msg = self.bridge.cv2_to_imgmsg(cam0_out, "rgb8")
            self.cam0_pub.publish(img_msg)
            img_msg = self.bridge.cv2_to_imgmsg(cam1_out, "rgb8")
            self.cam1_pub.publish(img_msg)
            img_msg = self.bridge.cv2_to_imgmsg(cam2_out, "rgb8")
            self.cam2_pub.publish(img_msg)



    def get_line_coefficients(self, mask):
        return

    def cam0_cb(self, msg):
        self.cam0_img_n += 1
        self.cam0_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")


    def cam1_cb(self, msg):
        self.cam1_img_n += 1
        self.cam1_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def cam2_cb(self, msg):
        self.cam2_img_n += 1
        self.cam2_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def cam0_params_cb(self, msg):
        self.K0 = np.array(msg.K).reshape(3,3)

    def cam1_params_cb(self, msg):
        self.K1 = np.array(msg.K).reshape(3,3)


    def cam2_params_cb(self, msg):
        self.K2 = np.array(msg.K).reshape(3,3) 

def gate_pose_publisher():
    try:
        rospy.init_node("image_processing_node", anonymous = True)

        img_processor = ImageProcessor()

        rospy.loginfo('Processing Node Started')
        rospy.spin()
        rospy.loginfo('Processing Node Terminated')     

    except rospy.ROSInterruptException:
        print("ROS Terminated.")
        pass

if __name__ == '__main__':
    gate_pose_publisher()

