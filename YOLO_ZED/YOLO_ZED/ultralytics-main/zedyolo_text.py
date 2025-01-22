import cv2
import torch
import argparse
import math
import pyzed.sl as sl
from ultralytics import YOLO
import csv  # Add this for CSV file handling


def main():
    # Initialize ZED camera
    zed = sl.Camera()

    # Create initialization parameters and set depth mode and units
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera and check if it's successful
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Unable to open ZED camera:", repr(status))
        exit()

    # Create YOLO model
    model = YOLO('v8n_best.pt')

    # Create display window and set its size
    cv2.namedWindow('00', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('00', 1280, 720)

    # Create video output object
    out_video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (1280, 720))

    # Create runtime parameters object
    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()  # To store camera images
    depth = sl.Mat()  # To store depth maps
    point_cloud = sl.Mat()  # To store point cloud data

    i = 0
    # Open CSV file to write results
    with open('detection.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X Center", "Y Center", "Distance (m)", "3D Coordinates"])

        while i < 300:  # Process 500 frames; adjust as needed
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve image and depth information
                zed.retrieve_image(image, sl.VIEW.LEFT)  # Get left-eye image
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # Get depth map
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)  # Get point cloud data

                # Convert ZED camera image data to format suitable for YOLO detection
                im0 = image.get_data()[:, :, :3]  # Convert to 3-channel
                results = model.track(im0, save=False, conf=0.5)  # Perform YOLO detection
                annotated_frame = results[0].plot()  # Get annotated frame image
                boxes = results[0].boxes.xywh.cpu()  # Get detection box coordinates

                for box in boxes:
                    x_center, y_center, width, height = box.tolist()  # Get box center and size
                    x1 = x_center - width / 2  # Calculate top-left x-coordinate
                    y1 = y_center - height / 2  # Calculate top-left y-coordinate
                    x2 = x_center + width / 2  # Calculate bottom-right x-coordinate
                    y2 = y_center + height / 2  # Calculate bottom-right y-coordinate

                    if 0 < x2 < im0.shape[1]:  # Check if box is inside the image
                        err, point_cloud_value = point_cloud.get_value(int(x_center),
                                                                       int(y_center))  # Get 3D coordinates

                        if math.isfinite(point_cloud_value[2]):
                            # Calculate Euclidean distance and convert to meters
                            distance = math.sqrt(
                                point_cloud_value[0] ** 2 + point_cloud_value[1] ** 2 + point_cloud_value[
                                    2] ** 2) / 1000
                            text_coords = f"Coords: ({point_cloud_value[0]:.2f}, {point_cloud_value[1]:.2f}, {point_cloud_value[2]:.2f})"  # 3D coordinates
                            text_dis_avg = f"distance: {distance:.2f}m"  # Format distance string
                            # Display distance on image
                            cv2.putText(annotated_frame, text_coords, (int(x2 - 150), int(y1 + 60)), cv2.FONT_ITALIC,
                                        0.6, (0, 255, 255), 3)
                            cv2.putText(annotated_frame, text_dis_avg, (int(x2 - 5), int(y1 + 30)), cv2.FONT_ITALIC,
                                        1.2, (0, 255, 255), 3)

                            # Write results to CSV file
                            writer.writerow([x_center, y_center, distance, point_cloud_value])

                            # Print output to console
                            print(
                                f"目标在 ({x_center},{y_center}) 处的距离: {distance:.2f} 米, 坐标: {point_cloud_value}")
                        else:
                            print(f"无法计算 ({x_center},{y_center}) 处的距离")

                # Display annotated frame
                cv2.imshow('00', annotated_frame)
                # Write annotated frame to output video
                out_video.write(annotated_frame)
                # Check if user pressed 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                i += 1

    # Release resources
    out_video.release()
    cv2.destroyAllWindows()
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='')
    parser.add_argument('--svo', type=str, default=None, help='Optional SVO file')
    parser.add_argument('--img_size', type=int, default=416, help='Inference image size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='Object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()
