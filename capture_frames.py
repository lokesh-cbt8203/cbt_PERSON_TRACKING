# import cv2
# from ultralytics import YOLO
# import os
# def detect_and_display_elephants(video_path,output_video_path):#,yolo_model_path,output_video_path
#     # model = YOLO(yolo_model_path)

#     # Open the video
#     cap = cv2.VideoCapture(video_path)
#     output_folder = r'data'
#     os.makedirs(output_folder, exist_ok=True)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Initialize VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#     frame_index=0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_index+=1
#         # if frame_index>1000:
#         #     break


#         # yolomodel = model(frame)


#         # for output in yolomodel:
#         #     for detection in output.boxes:
#         #         confi = detection.conf[0]



#         #         class_name = model.names[0]

#         #         if confi > 0.50  :
#         #             x1, y1, x2, y2 = map(int, detection.xyxy[0])
#         #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         #             label = f"{class_name}: {confi:.2f}"
#         #             cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#         # frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
#         # cv2.imwrite(frame_filename, frame)
#         re=cv2.resize(frame,(800,800))
#         cv2.imshow("frames", re)


#         key= cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('s'):
#             frame_filename = os.path.join(output_folder, f"frame_1{frame_index}.jpg")
#             cv2.imwrite(frame_filename, frame)


#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()



# video_path = r"rtsp://admin:rolex@123@192.168.1.111:554/Streaming/channels/101"
# # yolo_model_path = r"D:\vChanel\spinning mill\t_shirt.pt"
# output_video_path = r'camera_zone21.mp4'

# elephant_count = detect_and_display_elephants(video_path,output_video_path)#,yolo_model_path,output_video_path



# import cv2
# from ultralytics import YOLO
# import os
# import datetime
# from twilio.rest import Client

# def detect_and_display_elephants(video_path, yolo_model_path, output_video_path):
#     # Initialize the YOLO model
#     model = YOLO(yolo_model_path)

#     # Open the video
#     cap = cv2.VideoCapture(video_path)
#     output_folder = 'link_videosr'
#     os.makedirs(output_folder, exist_ok=True)

#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Initialize VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     frame_index = 0
#     start_time = datetime.datetime.now()

#     # Dictionary to store detected classes
#     detected_classes = {}

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_index += 1

#         # if frame_index > 1000:
#         #     break

#         # Perform detection
#         yolomodel = model(frame)

#         for output in yolomodel:
#             for detection in output.boxes:
#                 confi = detection.conf[0]
#                 class_id = int(detection.cls[0])
#                 class_name = model.names[class_id]

#                 if confi > 0.50:
#                     x1, y1, x2, y2 = map(int, detection.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     label = f"{class_name}: {confi:.2f}"
#                     cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#                     # Update detected classes count
#                     if class_name in detected_classes:
#                         detected_classes[class_name] += 1
#                     else:
#                         detected_classes[class_name] = 1

#         frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
#         cv2.imwrite(frame_filename, frame)
#         re = cv2.resize(frame, (800, 800))
#         cv2.imshow("frames", re)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#     end_time = datetime.datetime.now()

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     # Print start and end times, and detected classes
#     print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     print("Detected Classes:")
#     for cls, count in detected_classes.items():
#         print(f"{cls}: {count}")

#     # Send WhatsApp message
#     account_sid = 'AC447cc058422bc9ee033166995a4bc1f7'
#     auth_token = '176079717949284266ccd4dfbd42ab97'
#     client = Client(account_sid, auth_token)

#     message_body = f"""
#     Detection Report:
#     Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
#     End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
#     Detected Classes:
#     """
#     for cls, count in detected_classes.items():
#         message_body += f"{cls}: {count}\n"

#     message = client.messages.create(
#         body=message_body,
#         from_='whatsapp:+14155238886',  # Your Twilio WhatsApp number
#         to='whatsapp:+919442493256'  # Recipient's number
#     )

#     print(f"WhatsApp message sent: {message.sid}")

# # Define paths and call the function
# video_path = "rtsp://rtspstream:1930feb0841b01a47d00bb674f92da03@zephyr.rtsp.stream/movie"
# yolo_model_path = r"rstpmdl.pt"
# output_video_path = r'rstp.mp4'

# detect_and_display_elephants(video_path, yolo_model_path, output_video_path)


# import cv2
# from ultralytics import YOLO
# import os
# import datetime
# import pywhatkit as kit  # You need to install pywhatkit

# def detect_and_display_elephants(video_path, yolo_model_path, output_video_path):
#     model = YOLO(yolo_model_path)

#     # Open the video
#     cap = cv2.VideoCapture(video_path)
#     output_folder = 'link_video'
#     os.makedirs(output_folder, exist_ok=True)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Initialize video writer
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     # Capture start time
#     start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
#     detected_classes = []

#     frame_index = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_index += 1
#         if frame_index > 1000:
#             break

#         yolomodel = model(frame)

#         for output in yolomodel:
#             for detection in output.boxes:
#                 confi = detection.conf[0]
#                 class_name = model.names[0]

#                 if confi > 0.50 and class_name == "person":
#                     x1, y1, x2, y2 = map(int, detection.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     label = f"{class_name}: {confi:.2f}"
#                     cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#                     # Track detected class
#                     detected_classes.append(class_name)

#         frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
#         cv2.imwrite(frame_filename, frame)
#         re = cv2.resize(frame, (800, 800))
#         cv2.imshow("frames", re)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#     # Capture end time
#     end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     # Create message
#     detections = ", ".join(set(detected_classes))
#     message = (f"Start Time: {start_time}\n"
#                f"End Time: {end_time}\n"
#                f"Detected Classes: {detections}")

#     # Send WhatsApp message
#     try:
#         kit.sendwhatmsg("+919442493256", message, datetime.datetime.now().hour, datetime.datetime.now().minute + 2)
#         print("Message sent successfully!")
#     except Exception as e:
#         print(f"Failed to send message: {e}")

# video_path = "rtsp://rtspstream:1930feb0841b01a47d00bb674f92da03@zephyr.rtsp.stream/movie"
# yolo_model_path = r"rstpmdl.pt"
# output_video_path = r'factory.avi'  # Ensure to specify a valid video file extension

# detect_and_display_elephants(video_path, yolo_model_path, output_video_path)





# import cv2
# from ultralytics import YOLO
# import os
# import datetime as dt
# from twilio.rest import Client

# def detect_and_display_elephants(video_path, yolo_model_path, output_video_path):
#     # Initialize the YOLO model
#     model = YOLO(yolo_model_path)

#     # Open the video
#     cap = cv2.VideoCapture(video_path)
#     output_folder = 'link_videosr'
#     os.makedirs(output_folder, exist_ok=True)

#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Initialize VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     frame_index = 0
#     start_time = dt.datetime.now()

#     # Dictionary to store detected classes
#     detected_classes = {}

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_index += 1

#         # Perform detection
#         yolomodel = model(frame)

#         for output in yolomodel:
#             for detection in output.boxes:
#                 confi = detection.conf[0]
#                 class_id = int(detection.cls[0])
#                 class_name = model.names[class_id]

#                 if confi > 0.50:
#                     x1, y1, x2, y2 = map(int, detection.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     label = f"{class_name}: {confi:.2f}"
#                     cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#                     # Update detected classes count
#                     if class_name in detected_classes:
#                         detected_classes[class_name] += 1
#                     else:
#                         detected_classes[class_name] = 1

#         frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
#         cv2.imwrite(frame_filename, frame)
#         re = cv2.resize(frame, (800, 800))
#         cv2.imshow("frames", re)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#     end_time = dt.datetime.now()

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     # Print start and end times, and detected classes
#     print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     print("Detected Classes:")
#     for cls, count in detected_classes.items():
#         print(f"{cls}: {count}")

#     # Send WhatsApp message
#     account_sid = 'AC3bbaef04b28b5ae6c6aa66506cf4db75'
#     auth_token = '95950c8efad971ca755d6b5d5b486823'
#     client = Client(account_sid, auth_token)

#     message_body = f"""
#     Detection Report:
#     Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
#     End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
#     Detected Classes:
#     """
#     for cls, count in detected_classes.items():
#         message_body += f"Detecetd! Found class: {cls} :{count} \n"#

#     try:
#         message = client.messages.create(
#             body=message_body,
#             from_='whatsapp:+14155238886.',  # Your Twilio WhatsApp number
#             to='whatsapp:+919442493256'  # Recipient's number
#         )
#         print(f"WhatsApp message sent: {message.sid}")
#     except Exception as e:
#         print(f"Failed to send WhatsApp message: {e}")

# # Define paths and call the function
# video_path = "rtsp://localhost:8554/sr"
# yolo_model_path = r"D:\vChanel\fast_api\opt.pt"
# output_video_path = r'rstp.mp4'

# detect_and_display_elephants(video_path, yolo_model_path, output_video_path)
import cv2
import os
import math

def detect_and_display_elephants(video_path, output_video_path):
    # Open the video or RTSP stream
    cap = cv2.VideoCapture(video_path)
    output_folder = r'gallery'
    os.makedirs(output_folder, exist_ok=True)

    if not cap.isOpened():
        print("❌ Error: Could not open video stream.")
        return

    # Get video properties safely
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sometimes RTSP doesn't provide FPS → fallback to 25
    if fps == 0 or fps is None or math.isnan(fps):
        fps = 25.0
        print(f"⚠️ FPS not detected, using default = {fps}")

    # Initialize VideoWriter for saving the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Stream ended or frame read failed.")
            break

        frame_index += 1
        # frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
        # cv2.imwrite(frame_filename, frame)
        # print(f"✅ Saved snapshot: {frame_filename}")
    # Show the resized frame for visualization
        re = cv2.resize(frame, (800, 800))
        cv2.imshow("Live Stream", re)

        # ✅ Save frame to video
        out.write(frame)

        # Optional: save image on 's' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 Quit pressed. Stopping...")
            break
        elif key == ord('s'):
            frame_filename = os.path.join(output_folder, f"frame_{frame_index}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"✅ Saved snapshot: {frame_filename}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"🎥 Video saved as: {output_video_path}")

# ---------------- Run ----------------
video_path = r"rtsp://admin:rolex@123@192.168.1.110:554/Streaming/channels/101"
output_video_path = r'save_video_for_strong_sort12345.mp4'

detect_and_display_elephants(video_path, output_video_path)
