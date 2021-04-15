import cv2
import numpy as np
import dlib
import time
import math
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

from fish_object import FishObject
from centretracker import CentreTracker

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

def run_inference_for_single_image(model, image):
    image = np.asarray(image)

    input_tensor = tf.convert_to_tensor(image)

    input_tensor = input_tensor[tf.newaxis, ...]

    output = model(input_tensor)

    num_detections = int(output.pop('num_detections'))
    output = {key: value[0, :num_detections].numpy()
              for key, value in output.items()}
    output['num_detections'] = num_detections

    output['detection_classes'] = output['detection_classes'].astype(
        np.int64)

    if 'detection_masks' in output:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output['detection_masks'], output['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            detection_masks_reframed > 0.5, tf.uint8)
        output['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output

def run_inference(model, category_index, cap,zone1_line ,zone3_line,detection_thresh, skip, anlys_per,fish_num):

    total_frames = 0
    zone1_BA = []
    zone2_BA = []
    zone3_BA = []
    pos_BA = []
    dist_BA =  []
    analysis_BA = []

    sum3 = "Possible causus of this behaviour include: \n" \
           "- Fish stress \n" \
           "- Parasite on the outside of the fish body that it is trying to remove\n" \
           "- Improper water qualityParasite on the outside of the fish body that it is trying to remove\n\n" \
           "Actions required: \n" \
           "- Make sure the fish is in the correct environemnt and is with other compatible fish\n" \
           "- Check to see if the fish is rubbing against sides of the tank - as this is signs of a parasite\n"\
           "- Check the ph, ammonia and nitrates levels of the water"

    sum2 = "Possible causus of this behaviour include: \n" \
           "- Sign of disease \n" \
           "- Possible swim bladder infection\n" \
           "- Poor food quality" \
           "- Poor water quality\n\n" \
           "Actions required: \n" \
           "- Quarintine the fish to avoid possible spread of infection\n" \
           "- Try fasting the fish for a day or two\n"\
           "- Check water varaibles such as the ph level and nitrates in the water" \
           "- Seek treatment"

    sum1 = "Possible causus of this behaviour include: \n" \
           "- The water oxygenation is low\n" \
           "- Poor water quality\n\n" \
           "Actions required: \n" \
           "- Make sure the water is being properly aerated\n" \
           "- Check the ph, ammonia and nitrates levels of the water"

    sum4 = "Possible causus of this behaviour include: \n" \
           "- Improper water temperature \n" \
           "- Overfeeding\n" \
           "- Poor water quality\n\n" \
           "Actions required: \n" \
           "- Check water tempreture\n" \
           "- Make sure the food is being consumed in the appropriate time\n"\
           "- Check water varaibles such as the ph level and nitrates in the water"

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []

    fishObjects = {}

    start = time.perf_counter()

    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break

        height, width, _ = image_np.shape
        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        rects = []

        if total_frames % skip == 0 and len(trackers)<fish_num:
            print("Detecting...")
            trackers = []

            #The detection using the single image detection function
            output = run_inference_for_single_image(model, image_np)

            for i, (y_min, x_min, y_max, x_max) in enumerate(output['detection_boxes']):
                if output['detection_scores'][i] > detection_thresh:
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(x_min * width), int(y_min * height), int(x_max * width),
                                          int(y_max * height))
                    tracker.start_track(rgb, rect)

                    #append lists with new detected fish variables
                    trackers.append(tracker)
                    zone1_BA.append(0)
                    zone2_BA.append(0)
                    zone3_BA.append(0)
                    dist_BA.append(0)
                    pos_BA.append([0,0])
                    analysis_BA.append('')

        else:
            for tracker in trackers:
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                x_min, y_min, x_max, y_max = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((x_min, y_min, x_max, y_max))

        objects = ct.update(rects)

        for (fishID, centre) in objects.items():
            to = fishObjects.get(fishID, None)

            if to is None:
                to = FishObject(fishID, centre)
            else:


                if centre[1] < zone1_line * height:
                    elapsed = (time.perf_counter() - start)
                    zone1_BA[fishID] = zone1_BA[fishID] + elapsed

                    if centre[0]>pos_BA[fishID][0] and centre[1] > pos_BA[fishID][1]:
                        mat = (centre[0] - pos_BA[fishID][0]) ^ 2 + (centre[1] - pos_BA[fishID][1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                    elif pos_BA[fishID][0]>centre[0] and centre[1] > pos_BA[fishID][1]:
                        mat = (pos_BA[fishID][0] - centre[0]) ^ 2 + (centre[1] - pos_BA[fishID][1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                    elif centre[0]>pos_BA[fishID][0] and pos_BA[fishID][1] > centre[1]:
                        mat = (centre[0] - pos_BA[fishID][0]) ^ 2 + (pos_BA[fishID][1] - centre[1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                    elif pos_BA[fishID][0]>centre[0] and pos_BA[fishID][1] > centre[1]:
                        mat = (pos_BA[fishID][0] - centre[0]) ^ 2 + (pos_BA[fishID][1] - centre[1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                #if in zone 3
                elif centre[1] > zone3_line * height:
                    elapsed = (time.perf_counter() - start)
                    zone3_BA[fishID] = zone3_BA[fishID] + elapsed

                    if centre[0] > pos_BA[fishID][0] and centre[1] > pos_BA[fishID][1]:
                        mat = (centre[0] - pos_BA[fishID][0]) ^ 2 + (centre[1] - pos_BA[fishID][1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                    elif pos_BA[fishID][0] > centre[0] and centre[1] > pos_BA[fishID][1]:
                        mat = (pos_BA[fishID][0] - centre[0]) ^ 2 + (centre[1] - pos_BA[fishID][1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                    elif centre[0] > pos_BA[fishID][0] and pos_BA[fishID][1] > centre[1]:
                        mat = (centre[0] - pos_BA[fishID][0]) ^ 2 + (pos_BA[fishID][1] - centre[1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                    elif pos_BA[fishID][0] > centre[0] and pos_BA[fishID][1] > centre[1]:
                        mat = (pos_BA[fishID][0] - centre[0]) ^ 2 + (pos_BA[fishID][1] - centre[1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                #if in zone 2
                else:
                    elapsed = (time.perf_counter() - start)
                    zone2_BA[fishID] = zone2_BA[fishID] + elapsed

                    if centre[0] > pos_BA[fishID][0] and centre[1] > pos_BA[fishID][1]:
                        mat = (centre[0] - pos_BA[fishID][0]) ^ 2 + (centre[1] - pos_BA[fishID][1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                    elif pos_BA[fishID][0] > centre[0] and centre[1] > pos_BA[fishID][1]:
                        mat = (pos_BA[fishID][0] - centre[0]) ^ 2 + (centre[1] - pos_BA[fishID][1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                    elif centre[0] > pos_BA[fishID][0] and pos_BA[fishID][1] > centre[1]:
                        mat = (centre[0] - pos_BA[fishID][0]) ^ 2 + (pos_BA[fishID][1] - centre[1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                    elif pos_BA[fishID][0] > centre[0] and pos_BA[fishID][1] > centre[1]:
                        mat = (pos_BA[fishID][0] - centre[0]) ^ 2 + (pos_BA[fishID][1] - centre[1]) ^ 2
                        dist_BA[fishID] = dist_BA[fishID] + math.sqrt(mat)

                pos_BA[fishID] = [centre[0], centre[1]]
                to.centroids.append(centre)


            fishObjects[fishID] = to

            #mark fish with ID and centre tracking point
            cv2.putText(image_np, "Fish {}".format(fishID), (centre[0] - 10, centre[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.circle(image_np, (centre[0], centre[1]), 4, (0, 255, 0), -1)

        # Blue zone 1 line
        cv2.line(image_np, (0, int(zone1_line * height)), (width, int(zone1_line * height)), (255, 0, 0), 5)
        # Red zone 3 line
        cv2.line(image_np, (0, int(zone3_line * height)), (width, int(zone3_line * height)), (0, 0, 255), 5)

        # display count and status
        start = time.perf_counter()
        cv2.imshow('Behaviour Analysis', image_np)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        if time.perf_counter()>anlys_per:
            print("End of period")
            break

        total_frames += 1

    cap.release()
    cv2.destroyAllWindows()

    #Fish behaviour analysis based on results for every ID still prsent
    for (fishID, centre) in objects.items():

        #calculate parameters and behaviour analysis summary
        totaltime = zone1_BA[fishID] + zone2_BA[fishID] + zone3_BA[fishID]
        zone1P = round((zone1_BA[fishID]/totaltime)*100)
        zone2P = round((zone2_BA[fishID]/totaltime)*100)
        zone3P = round((zone3_BA[fishID]/totaltime)*100)

        if zone1P > 50 and dist_BA[fishID] < 1100:
            analysis_BA[fishID] = "Extended periods at the top of the tank"

        elif zone3P > 60:
            analysis_BA[fishID] = "Extended periods at the bottom of the tank"

        elif dist_BA[fishID] > 1100 and zone3P < 60:
            analysis_BA[fishID] = "Eratic behaviour"

        elif dist_BA[fishID] < 350 and zone3P < 60:
            analysis_BA[fishID] = "Listelness behaviour"

        else:
            analysis_BA[fishID] = "No irregularities"


        #Display fish results
        print("FISH: ", fishID)
        print("------------------")

        print("Percentage of time in zone 1: ",zone1P)
        print("Percentage of time in zone 2: ",zone2P)
        print("Percentage of time in zone 3: ",zone3P)
        print("Distance travelled: ", round(dist_BA[fishID],2))
        print("Analysis of Behaviour: ", analysis_BA[fishID],"detected \n")


        if analysis_BA[fishID] == "Extended periods at the top of the tank":
            print(sum1)
        if analysis_BA[fishID] == "Extended periods at the bottom of the tank":
            print(sum2)
        if analysis_BA[fishID] == "Eratic behaviour":
            print(sum3)
        if analysis_BA[fishID] == "Listelness behaviour":
            print(sum4)

if __name__ == '__main__':

    #Paths to the trained model and label map
    PATH_TO_MODEL_DIR = 'C:/Users/Jordan/ComProject/TensorFlow/workspace/training_demo/exported-models/my_model'
    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    PATH_TO_LABELS = 'C:/Users/Jordan/ComProject/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt'

    #Loading in the model and labels
    tf.keras.backend.clear_session()
    model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    model = model.signatures['serving_default']
    print("Model loaded")

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    print("Labels loaded")

    #Inference adjustable conditions
    #-------------------------------

    #Path to the video file if used, if not webcam will be used
    PATH_TO_VIDEO = ''
    #Threshold for the detection confidence 0-1
    detection_thresh = 0.15
    #Positions of zone lines 0-1
    zone1_line = 0.45
    zone3_line =0.72
    #Frames to skip between applying the model for detection
    skip = 20
    #analyis period in seconds
    anlys_per = 60
    #number of fish in habitat to look for
    fish_num = 1

    #check for video file specification, otherwise use the webcam
    if PATH_TO_VIDEO != '':
        cap = cv2.VideoCapture(PATH_TO_VIDEO)
    else:
        cap = cv2.VideoCapture(0)
        print("Camera loaded")

    if not cap.isOpened():
        print("Could not open video file.")

    print("Running inference")
    run_inference(model, category_index, cap, zone1_line, zone3_line, detection_thresh, skip, anlys_per,fish_num)