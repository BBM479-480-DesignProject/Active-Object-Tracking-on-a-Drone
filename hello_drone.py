import setup_path
import airsim

import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import numpy as np
import os
import tempfile
import pprint
import cv2
import torch
import numpy as np
import supervision as sv
import time
from ultralytics import YOLO
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from filter import PointEKF, BoundingBoxKalmanFilter, PointTrackerKalmanFilter
from reid import reid, reid_all


import threading
from helper_functions import calc_velocity, calc_difs, calc_euler, calc_arrays, plot_all, bounding,calc_measurement
# connect to the AirSim simulator
from airsim import DrivetrainType

timer_controller = False
tracking_time = 120
initializer = True
distance_x = 0
distance_y = 0


def set_distance_x(value):
    global distance_x
    distance_x = value


def set_distance_y(value):
    global distance_y
    distance_y = value

def get_distance_x():
    global distance_x
    return distance_x


def get_distance_y():
    global distance_y
    return distance_y

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client_2 = airsim.MultirotorClient()

client.confirmConnection()
client.enableApiControl(True)

def yaw_controller(distance):
    if -2.5 < distance < 2.5:
        return 0
    else:
        return distance

def altitude_controller(altitude):
    if altitude > 23.5:
        return 2
    elif altitude < 21.5:
        return -2
    else:
        return 0


def image_tracker():
    global distance_x
    global distance_y
    kalmans = []
    last_id = 0
    reid_dict = {}

    # Set device to CUDA if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    
    # Kalman filter attributes
    initial_state = np.array([240, 240])  # Initial state estimate
    initial_covariance = np.eye(2) * 1000  # Initial covariance estimate
    process_noise = np.eye(2) * 0.01  # Process noise covariance
    measurement_noise = np.eye(2) * 0.1  # Measurement noise covariance

    # Create Kalman filter object
    tracker = PointTrackerKalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise)


    # used for stopping the capturing streams
    counter = 0
    lost_counter = 0

    # Prompt the user to enter an integer for active_track
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Active track
    active_track = simpledialog.askinteger("Enter Active Track ID", "Please enter an integer for the active track ID:")
    active_track_xy = (240, 240)
    is_track = False
    active_track_text = "No active track"

    root.deiconify()

    # chose supervision library to draw cleaner boxes
    box_annotator = sv.BoundingBoxAnnotator(
        thickness=1,
    )

    label_annotator = sv.LabelAnnotator(
        text_thickness=1,
        text_scale=0.35
    )

    # model used in this project (yolov8s.pt for general purpose vehicles.pt for cars)
    model = YOLO('vehicle.pt')
    # model = YOLO("vehicles.pt")
    model.to(device)

    # used for capturing streams
    bbox = np.array([[1],[1]])

    # preparations to save output
    frame_width = 480#int(cap.get(3))
    frame_height = 480#int(cap.get(4))
    size = (frame_width, frame_height)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10.0, size)

    reid_val = active_track

    # Create canvas to display video
    canvas = tk.Canvas(root, width=frame_width, height=frame_height)
    canvas.pack()

    # Dictionary to keep track of buttons for each ID
    id_buttons = {}

    # Function to update video display
    def update_frame():
        nonlocal counter
        nonlocal active_track
        nonlocal tracker
        nonlocal active_track_xy
        nonlocal active_track_text
        nonlocal id_buttons
        nonlocal bbox
        nonlocal lost_counter
        nonlocal reid_val
        nonlocal kalmans
        nonlocal last_id
        nonlocal reid_dict

        responses = client_2.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGRA2BGR)

        print(frame.shape)

        if True:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            result = model.track(frame, persist=True, tracker="custom.yaml")[0]
            # Visualize the results on the frame
            detections = sv.Detections.from_ultralytics(result)

            for i in range(len(detections.class_id)):
                if detections.class_id[i] == 0:
                    detections.class_id[i] = 1

            if result.boxes.id is not None:

                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

                detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

                kalmans, last_id, detections, reid_dict = reid_all(kalmans, last_id, detections, reid_dict)
                print(kalmans)
                print(reid_dict)
                print(last_id)

                labels = [f"#{tracker_id}" for _, _, _, _, tracker_id, _ in detections]

                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)


                if active_track not in detections.tracker_id:
                    lost_counter += 1
                    if lost_counter == 90:
                        active_track_xy = np.sum(detections.xyxy, axis=0) / detections.xyxy.shape[0]
                        active_track_xy = [(active_track_xy[0] + active_track_xy[2]) / 2,
                                           ((active_track_xy[1] + active_track_xy[3]) / 2)]
                        active_track_text = "No active track"
                    else:
                        active_track_xy = tracker.get_predicted_state()

                else:
                    lost_counter = 0
                    for i in range(len(detections.tracker_id)):
                        if detections.tracker_id[i] == active_track:
                            active_track_xy = [(detections.xyxy[i][0] + detections.xyxy[i][2]) / 2,
                                               ((detections.xyxy[i][1] + detections.xyxy[i][3]) / 2) - 10]
                            active_track_text = "Active track: " + str(active_track)


            tracker.predict()  # Use 'kf' attribute to access Kalman filter methods

            if active_track_xy is not None:
                print(active_track_xy)
                tracker.update(active_track_xy)
            else:
                tracker.update(tracker.get_predicted_state())



            y_center, x_center = frame.shape[0] // 2, frame.shape[1] // 2

            pixel_distances = x_center - tracker.get_predicted_state()[0], y_center - tracker.get_predicted_state()[1]

            distance_data = client_2.getDistanceSensorData().distance
            set_distance_x(distance_data * pixel_distances[0] / 315)
            set_distance_y( max(distance_data * pixel_distances[1] / 315, 0))


            cv2.circle(frame, (int(tracker.get_predicted_state()[0]), int(tracker.get_predicted_state()[1])), 3, (0, 255, 2), -1)


            # Define the font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            font_color = (255, 0, 0)  # White color
            thickness = 1

            # Get the size of the text
            text_size = cv2.getTextSize(active_track_text, font, font_scale, thickness)[0]

            # Position of the text (upper left corner)
            x = 10  # distance from left edge
            y = 10 + text_size[1]  # distance from top edgey

            # Draw the text on the image
            cv2.putText(frame, active_track_text, (x, y), font, font_scale, font_color, thickness)

            # Display frame on canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            canvas.img_tk = img_tk  # Keep reference to avoid garbage collection
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

            # Update buttons based on current detections
            update_buttons(detections)
            out.write(frame)
            if not timer_controller:
                root.after(1, update_frame)
              # Update every 1 millisecond



    # Function to update buttons based on current detections
    def update_buttons(detections):
        nonlocal id_buttons
        if detections.tracker_id is not None:
            current_ids = set(detections.tracker_id)
            existing_ids = set(id_buttons.keys())
            removed_ids = existing_ids - current_ids

            # Remove buttons for IDs that are no longer visible
            for id in removed_ids:
                id_buttons[id].destroy()
                del id_buttons[id]

            # Create buttons for new IDs
            for id in current_ids:
                if id not in id_buttons:
                    button = tk.Button(root, text=f"Track {id}", command=lambda id=id: change_active_track(id))
                    button.pack(side=tk.LEFT)
                    id_buttons[id] = button

        else:
            # Remove all buttons if tracker_id is None
            for button in id_buttons.values():
                button.destroy()
            id_buttons = {}

    # Function to change active track
    def change_active_track(track_id):
        nonlocal active_track
        nonlocal reid_val
        active_track = track_id
        reid_val = track_id

    # Start updating the video display
    update_frame()

    root.mainloop()
    out.release()

    #cap.release()


def commander():
    global initializer
    global timer_controller
    global distance_y
    global distance_x
    print("Taking off...")
    client.armDisarm(True)
    client.takeoffAsync().join()
    time_arr = []
    altitudes = [[],[]]
    start_time = time.time()

    json_array_d = []  # gt
    json_array_v = []  # gt
    cv_array = []

    # initialization
    client.moveByVelocityBodyFrameAsync(0, 0, -30, 0.15, drivetrain=DrivetrainType.ForwardOnly).join()
    #client.moveByVelocityBodyFrameAsync(2, 0, 0, 0.8, drivetrain=DrivetrainType.ForwardOnly).join()
    #time.sleep(0.1)

    initializer = False
    while True:
        if timer_controller:
            break
        json_d = client.simGetVehiclePose('my_drone')
        json_v = client.simGetObjectPose("AI_CAR_PURPLE_2")
        #print(f"{json_v.position}         POSITION            ")
        json_array_d.append(json_d)
        json_array_v.append(json_v)
        distance_data = client.getDistanceSensorData().distance
        altitudes[0].append(distance_data)
        altitudes[1].append(abs(json_d.position.z_val-json_v.position.z_val))
        #cv_array.append([distance_x, distance_y, distance_data])
        cv_array.append([get_distance_x(), get_distance_y(), distance_data])

        #distance_x = yaw_controller(distance_x)
        set_distance_x(yaw_controller(get_distance_x()))

        vx = -get_distance_x() / 60
        vy = get_distance_y() / 2.8 + 1.15
        vz = altitude_controller(distance_data)

        client.moveByVelocityBodyFrameAsync(vy, vx, vz, 0.03, drivetrain=DrivetrainType.ForwardOnly,
                                            yaw_mode=airsim.YawMode(False, 0)).join()

        current_time = time.time()
        time_arr.append(current_time - start_time)

    ################
    poses_drone, poses_vehicle, orientation_drone, orientation_vehicle = calc_arrays(json_array_v, json_array_d)
    ################
    pose_difs, orientation_difs = calc_difs(poses_drone, poses_vehicle, orientation_vehicle, orientation_drone)
    ################
    distance_gt,distance_cv= calc_measurement(pose_difs,cv_array)
    ################
    euler_drone, euler_vehicle = calc_euler(orientation_drone, orientation_vehicle)
    ################
    velocities_d, velocities_v = calc_velocity(poses_drone, poses_vehicle, time_arr)
    ################
    array = [poses_drone, poses_vehicle, orientation_drone, orientation_vehicle, pose_difs, orientation_difs,
             velocities_d, velocities_v, euler_drone, euler_vehicle, altitudes,distance_gt,distance_cv]
    plot_all(time_arr, array)
    ################

def timer():
    global tracking_time
    global timer_controller
    counter = 0
    while 1:
        if counter >= tracking_time:
            timer_controller = True
            print("Time finished")
            break
        counter += 1
        time.sleep(1)



t0 = threading.Thread(target=image_tracker)
t1 = threading.Thread(target=commander)
t2 = threading.Thread(target=timer)

t0.start()
t1.start()
t2.start()
t0.join()
t1.join()
t2.join()



airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
