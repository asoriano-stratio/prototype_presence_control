import datetime

import cv2
from arango import ArangoClient
from sklearn.neighbors import NearestNeighbors
import queue
from threading import Thread

from AlignedReIDModel import AlignedReIDModel
from helper_funcions import PeopleFeatures, TimeMeasurement, image_equalization_clahe, update_tracker, \
    draw_entrance_area, \
    filter_locations, do_face_recognition, update_tracker_v2
from people_db import PeopleDB
from people_tracker import PeopleTracker
from utils import video_frame_reader, show_image_opencv

# => Initializations

# · Arango
client = ArangoClient(protocol='http', host="localhost", port=8529)
db = client.db("spaceai", username="spaceai", password="spaceai")
coll = db.collection("all_stratians_one_face")
stratiansDb = coll.all().batch()
names = []
people_embeddings = []
knn_k = 1
for s in stratiansDb:
    names.append(s["normalized_name"])
    people_embeddings.append(s["embedding"])
knn = NearestNeighbors(n_neighbors=knn_k, algorithm="auto", metric='euclidean').fit(people_embeddings)

# · Image equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# · Face detector
haar_detector = cv2.CascadeClassifier("/home/asoriano/Escritorio/spaceai-evolution/evolution-vision/spaceai_vision_framework_project/resources/models/face_detection/haar/haarcascade_frontalface_default.xml")
face_recognition_model = {"names": names, "knn": knn, "haar": haar_detector}

# · People detection and feature extraction
people_model = PeopleFeatures(init_emb_model=False)
aligned_reid_model = AlignedReIDModel("/home/asoriano/Escritorio/spaceai-evolution/Models/AlignedReId/Market1501_Resnet50_Alignedreid(LS)/checkpoint_ep300.pth.tar")
people_db = PeopleDB(aligned_reid_model)
people_tracker = PeopleTracker(aligned_reid_model, people_db)


# · Time measurement
time_measurement = TimeMeasurement(timers_names=["pre-processing", "detection", "crop", "img_save", "embedding"])
# · Path for storing images
img_saver_path = "/home/asoriano/workspace/pocs/eval_tracker_person_embs/data/images"
# · Csv header
csv_header = "img_path,n_frame,top,right,bottom,left"
csv_path = "/home/asoriano/workspace/pocs/eval_tracker_person_embs/data/people_data.csv"
f = open(csv_path, 'w')
f.write(csv_header + "\n")
# - Tracker memory
tracker_memory = []
tracked_memory = []
tracked = []
tracked_path = "/home/asoriano/workspace/pocs/eval_tracker_person_embs/data/tracked"
# - Entrance area
x = 280
w = 200
y = 70
h = 340
entrance_area = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]  # t_l, t_r, b_l, b_r
loc_entrance_area = (y, x+w, y+h, x)

# => Open video
(height, width) = (None, None)
queue = queue.Queue(maxsize=1)
url = "/home/asoriano/workspace/spaceai-evolution/Videos/test_tracker_exploring_land/rsp1.avi"
url = "/home/asoriano/Escritorio/spaceai-evolution/Videos/test_entrance_hypatia/rsp1_ev_people.avi"
url = "tcp://10.90.0.152:5000"
url = "tcp://10.90.1.153:5000"
t = Thread(target=video_frame_reader, args=(url, queue, False))
t.daemon = True
t.start()

faces_locations = []

# => Processing video
global_frame_count = 0
fps_start = datetime.datetime.now()
fps_frame_count = 0
while True:

    # => Grab a single frame of video
    frame = queue.get(block=True, timeout=10)
    if height is None or width is None:
        (height, width) = frame.shape[:2]
    global_frame_count += 1
    fps_frame_count += 1

    # => Pre-processing
    time_measurement.initTimer("pre-processing")
    frame = image_equalization_clahe(clahe, frame)
    time_measurement.measureTime("pre-processing")

    # => Information extraction
    # · Detecting people
    time_measurement.initTimer("detection")
    people_locations = people_model.human_locations(frame)
    # locations = filter_locations(locations, height, width)
    time_measurement.measureTime("detection")

    faces_locations = []
    faces_recognitions = []
    if len(people_locations) > 0:

        # · Cropping people
        time_measurement.initTimer("crop")
        img_people = people_model.crop_human(frame, people_locations)
        time_measurement.measureTime("crop")

        # · Faces detection
        time_measurement.initTimer("face recognition")
        faces_people = []
        for img, loc in zip(img_people, people_locations):
            faces_img = do_face_recognition(img, face_recognition_model, loc)
            faces_people.append(faces_img)
        # for faces_locs, faces_encodings, recognitions in faces_people:
        #     faces_locations.extend(faces_locs)
        #     faces_recognitions.extend(recognitions)
        time_measurement.measureTime("face recognition")

        # · Getting features from people
        time_measurement.initTimer("people embedding")
        people_embeddings = [aligned_reid_model.get_embeddings(person) for person in img_people]
        time_measurement.measureTime("people embedding")

    else:
        img_people = []
        faces_people = []
        people_embeddings = []

    # => Tracker
    people_tracker.do_tracking(people_locations, people_embeddings, img_people, faces_people, global_frame_count)

    # => Draw
    people_tracker.draw_information(frame, global_frame_count)

    # for p in tracker_memory:
    #     for l in p["trajectory"]:
    #         top, right, bottom, left = l
    #         m = (int((right + left) / 2), int((top + bottom) / 2))
    #         m = (int((right + left) / 2), int((bottom)))
    #         cv2.circle(frame, m, 5, p["color"], -1)
    #
    # if len(tracked) > 0:
    #     for person in tracked:
    #         print("Tracked %s frames" % person["tracked"])
    #         n_faces = len([f for f in person["faces"] if len(f[0]) > 0])
    #         if n_faces > 0:
    #             print("Retrieved %s faces\n" % n_faces)
    #
    #         if n_faces > 3:
    #             tracked_memory.append(person)
    #
    # # => Draw on frame
    # for top, right, bottom, left in people_locations:
    #     cv2.rectangle(frame, (left, top), (left + (right - left), top + (bottom - top)), (0, 255, 0), 2)
    # for (top, right, bottom, left), recog in zip(faces_locations, faces_recognitions):
    #     cv2.rectangle(frame, (left, top), (left + (right - left), top + (bottom - top)), (0, 0, 255), 2)
    #     height, width = frame.shape[:2]
    #     small_name = recog[:18]
    #     if len(recog) != len(small_name):
    #         small_name += '...'
    #     box_w = 25 + int(len(small_name) * 12.5)
    #     box_h = 30
    #     box_x = left + int((right - left) / 2) - int(box_w / 2)
    #     box_y = bottom + 40
    #     if box_y + box_h > height:
    #         box_y = height - box_h
    #     cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), cv2.FILLED)
    #     cv2.putText(frame, small_name, (box_x + 20, box_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

    # => Processing FPS
    if global_frame_count % 100 == 0:
        elapsed_time = (datetime.datetime.now() - fps_start).total_seconds()
        estimated_fps = fps_frame_count / elapsed_time
        fps_frame_count = 0
        fps_start = datetime.datetime.now()
        print("Estimated fps over 100 frames: %s" % estimated_fps)

    # => Show frame
    frame = cv2.resize(frame, (int(1024), int(720)))
    show_image_opencv(frame, "debug")

    if global_frame_count % 100 == 0:
        print(time_measurement.timeMap)

    time_measurement.reset()

f.close()