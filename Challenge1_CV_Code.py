# The calibration part of the code was written inspired on the tutorial given in the OpenCV documentation:
# https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

# Xabier Oyanguren Asua 1456628

import numpy as np
import cv2
import glob
import os
import shutil
import matplotlib.pyplot as plt

num_photos = 25
execution_path = "."
os.makedirs(f"{execution_path}/Chess_Board_Views/", exist_ok=True)
os.makedirs(f"{execution_path}/Chess_Boards_with_Detections/", exist_ok=True)
os.makedirs(f"{execution_path}/Calibrated_Parameters/", exist_ok=True)
os.makedirs(f"{execution_path}/Undistorted_Chess_Samples/", exist_ok=True)


# The left camera image will be the final deoth map so better the good resolution one
vidStream = cv2.VideoCapture(2)  # index of OBS - Nirie

vidStream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vidStream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# We will do the calibration using a reference 9x6 chess board
# First, we prepare the world coordinates of those points in the real world
# We can assume z=0 and only provide x,y-s.
# In chess board square units this would leave us with the points:
# (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
# if instead it was that the square of the board had 10 units (say cm)
# then we would calibrate things in cm with (0,0,0), (10,0,0), (20,0,0) etc.
# In our case the board squares have 2.4 cm each in width and height, so:
world_points_chess = np.zeros((6*9,3), np.float32)
world_points_chess[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*2.4 # in cm

# Arrays to store world points and image points from all the images.
world_points = [] # 3d point in real world space
image_points = [] # 2d points in image plane

# termination criteria for the optimization of corner detections
corner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 1e-9)

successful=1
while successful <= num_photos:
    # Capture num_photos views of the chess board - Better if 20 for example
    print(f"\n\nTAKING PHOTO {successful}/{num_photos} ########################")
    #input("Press ENTER to take the photos:")
    # instead of using .read() to get an image we decompose it into .grab and then .retrieve
    # so we can maximize the sinchronization

    if not vidStream.grab():
        print("[Error] Getting the image for this iteration. Retrying...")
        continue

    _, img = vidStream.retrieve()

    height, width, channels  = img.shape

    print(f"\nTaken image is: {width}x{height}x{channels}")

    # Save the taken image
    cv2.imwrite(f"{execution_path}/Chess_Board_Views/{successful:06}.png", img)


    found =True
    detected_corners_in_image=0
    gray=0
    # Read the image in numpy format
    full_img = cv2.imread(f"{execution_path}/Chess_Board_Views/{successful:06}.png")

    # Grayscale the image
    gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners in the image-> Telling that it should look for a 9x6 subchess
    # The detected corners will be the pixel numbers in pixel reference (9x6, 1, 2)
    found, detected_corners_in_image = cv2.findChessboardCorners(gray, (9,6), None)

    if found:
        print(f"\nDetected corners {detected_corners_in_image.shape} on image shape {full_img.shape}")

        # Improve/Refine detected corner pixels
        cv2.cornerSubPix(gray, detected_corners_in_image, (11,11), (-1,-1), corner_criteria)

        # If found-> add object points, image points (after refining them)
        # Draw and display the corners in the image to check if correctly detected -> full_img is modified!
        cv2.drawChessboardCorners(img, (9,6), detected_corners_in_image, found)
        cv2.imshow('Processed image is ok? Press SPACE if yes', img)
        ok = cv2.waitKey(2500) #########
        cv2.destroyAllWindows()

        if (ok!=32): ################
            print(f"\nVALIDATED chess table number {successful}/{num_photos}!\n")
            cv2.imwrite(f"{execution_path}/Chess_Boards_with_Detections/{successful:06}_with_detected_corners.png", img)

            # Valid detection of corners, so we add them to our list for calibration
            world_points.append(world_points_chess)

            image_points.append(detected_corners_in_image)
            successful+=1
        else:
            print("\nTrying AGAIN!\n")
    else:
        print("\nTrying AGAIN!\n")


print("\n\nCALIBRATING CAMERA! ###############################################")

camera_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-5)

# We input the world points for the chess points together with the corresponding image points
# in each image to the algorithm that calibrates the camera
ret, camera_matrix, distortion_coefficients, rot_vecs_views, \
        trans_vecs_views = cv2.calibrateCamera( world_points, image_points,
        gray.shape[::-1],  # shape tuple (nx, ny) of the images, just used to initialize the intrinsic param matrix
        None,None, criteria=camera_criteria)
# camera_matrix is the affine reference change from the projected to the pixel frame. The so called
# intrinsic camera matrix.
# it contains focal distances fx, fy and translation from the center of projection c1,c2 to
# the origin of pixels

# distoriton coefficients are the parameters to correct the intrinsic distortions of the camera
# [k1 k2 p1 p2 k3]
# k_j is the j-th radial distortion coefficient and p_j the j-th tangential distortion coefficient

# Rot vecs and trans vecs are the rotation and translation vectors to go from one image view of the
# board to the other ones

# We save the camera matrices and the distortion coefficents
print(f"The computed intrinsic camera matrix is: \n{camera_matrix}\n The distortion parameters are:\n[k1 k2 p1 p2 k3]={distortion_coefficients}\n")
np.save(f"{execution_path}/Calibrated_Parameters/Camera_Matrix_Cam_to_Pixel.npy", camera_matrix)
np.save(f"{execution_path}/Calibrated_Parameters/Distortion_Parameters.npy", distortion_coefficients)

# We compute the error if we re-projected it all with the found parameters
mean_error = 0
for i in range(len(world_points)):
    aprox_image_Points, _ = cv2.projectPoints(world_points[i], rot_vecs_views[i], trans_vecs_views[i], camera_matrix, distortion_coefficients)
    error = cv2.norm(image_points[i], aprox_image_Points, cv2.NORM_L2)/len(aprox_image_Points)
    mean_error += error
print( f"\nCAMERA CALIBRATED!\nMSE for reprojection given distortion correction is: {mean_error/len(world_points)}" )


print("\n\nUNDISTORTING THE CALIBRATION IMAGES #################################################################")

# Now that we have the camera matrix and the distortion parameters we can refine the camera matrix
# The thing is that when undistorting there will be some pixels that will be left in black due to
# the parts that were part of a distortion (like a fisheye, which left parts of the rectangular
# view outside the image, so we cannot reconstruct them when undistorting and blakc pixels will
# be set there.
# we can use the cv2.getOptimalNewCameraMatrix() function to refine the camera intrinsic matrix
# to get a matrix that undistorts taking into account the black regions or not. We will control
# this with a free scaling parameter alpha.
# With alpha=1 all source pixels are undistorted, which can lead to a non-rectangular image region - visible in the black pincushioning. Because the image has to be rectangular in memory, the gaps are padded with black pixels. Setting alpha=0 effectively increases the focal length to have a rectangular undistorted image where all pixels correspond to a pixel in the original. You lose data from the periphery, but you don't have any padded pixels without valid data.
alpha=1
refined_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (width,height), alpha, (width,height))
# the roi is a shape giving the subset of the image that does not have black pixels
x, y, w, h = roi

# we get all the chess board images we took
different_chess_views = sorted(glob.glob(f"{execution_path}/Chess_Board_Views/*.png"))

for file_name in different_chess_views:
    full_image = cv2.imread(file_name)
    # undistort the image
    undistorted = cv2.undistort(full_image, camera_matrix, distortion_coefficients, None, refined_camera_matrix)

    # crop the image to avoid black pixels
    undistorted = undistorted[y:y+h, x:x+w]

    # save the undistorted image
    cv2.imwrite(f"{execution_path}/Undistorted_Chess_Samples/{file_name.split('.')[-2].split('/')[-1]}_undistorted.png", undistorted)
