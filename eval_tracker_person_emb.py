import datetime

import cv2
from arango import ArangoClient
from sklearn.neighbors import NearestNeighbors
import queue
from threading import Thread
from helper_funcions import PeopleFeatures, TimeMeasurement, image_equalization_clahe, update_tracker, \
    draw_entrance_area, \
    filter_locations, do_face_recognition
from utils import video_frame_reader, show_image_opencv

# => Initializations

# · Arango
client = ArangoClient(protocol='http', host="localhost", port=8529)
db = client.db("spaceai", username="spaceai", password="spaceai")
coll = db.collection("all_stratians_one_face")
stratiansDb = coll.all().batch()
names = []
embeddings = []
knn_k = 1
for s in stratiansDb:
    names.append(s["normalized_name"])
    embeddings.append(s["embedding"])
knn = NearestNeighbors(n_neighbors=knn_k, algorithm="auto", metric='euclidean').fit(embeddings)

haar_detector = cv2.CascadeClassifier("/home/asoriano/Escritorio/spaceai-evolution/evolution-vision/spaceai_vision_framework_project/resources/models/face_detection/haar/haarcascade_frontalface_default.xml")

face_recognition_model = {"names": names, "knn": knn, "haar": haar_detector}

# · People detection and feature extraction
people_model = PeopleFeatures()

# · Image equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
# videofile_path = "/home/asoriano/workspace/spaceai-evolution/Videos/test_tracker_exploring_land/rsp1.avi"
# videofile_path = "/home/asoriano/Escritorio/spaceai-evolution/Videos/test_entrance_hypatia/rsp1_ev_people.avi"
# #video_capture = cv2.VideoCapture(videofile_path)
#
# # video_capture = cv2.VideoCapture(0 + cv2.CAP_V4L)
# # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# #
# video_capture = cv2.VideoCapture("tcp://10.90.1.153:5000")
# #video_capture = cv2.VideoCapture("tcp://10.90.0.152:5000")

(height, width) = (None, None)
queue = queue.Queue(maxsize=1)
url = "tcp://10.90.1.153:5000"
t = Thread(target=video_frame_reader, args=(url, queue))
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

    #frame = cv2.resize(frame, (int(640), int(480)))


    # => Pre-processing
    time_measurement.initTimer("pre-processing")
    frame = image_equalization_clahe(clahe, frame)
    time_measurement.measureTime("pre-processing")

    # => Detecting people
    time_measurement.initTimer("detection")
    locations = people_model.human_locations(frame)
    locations = filter_locations(locations, height, width)
    time_measurement.measureTime("detection")

    if len(locations) > 0:

        # => Cropping people
        time_measurement.initTimer("crop")
        img_people = people_model.crop_human(frame, locations)
        faces_people = []
        for img in img_people:
            faces_img = do_face_recognition(img, face_recognition_model)
            faces_people.append(faces_img)

        time_measurement.measureTime("crop")

        # => Getting features from people
        time_measurement.initTimer("embedding")
        embeddings = [people_model.human_vector(person) for person in img_people]
        time_measurement.measureTime("embedding")

        # => Tracker
        tracker_memory, tracked = update_tracker(tracker_memory, people_model, locations, embeddings, img_people, faces_people, global_frame_count)

        for p in tracker_memory:
            for l in p["trajectory"]:
                top, right, bottom, left = l
                m = (int((right + left) / 2), int((top + bottom) / 2))
                m = (int((right + left) / 2), int((bottom)))
                cv2.circle(frame, m, 5, p["color"], -1)

        # for track in tracked:
        #     save_tracked_person_images(tracked_path, track)

        # => Saving people images
        # time_measurement.initTimer("img_save")
        # save_people_images(img_saver_path, f, frame_counter, height, width, locations, img_people, embeddings)
        # time_measurement.measureTime("img_save")
        # save_people_images_in_area(img_saver_path, f, frame_counter, loc_entrance_area, locations, img_people, embeddings)

    else:
        tracked = tracker_memory
        img_people = []
        embeddings = []
        tracker_memory = []

    if len(tracked) > 0:
        for person in tracked:
            print("Tracked %s frames" % person["tracked"])
            n_faces = len([f for f in person["faces"] if len(f[0]) > 0])
            if n_faces > 0:
                print("Retrieved %s faces\n" % n_faces)

    # => Draw on frame
    for top, right, bottom, left in locations:
        cv2.rectangle(frame, (left, top), (left + (right - left), top + (bottom - top)), (0, 255, 0), 2)
    for top, right, bottom, left in faces_locations:
        cv2.rectangle(frame, (left, top), (left + (right - left), top + (bottom - top)), (0, 0, 255), 2)

    draw_entrance_area(frame, entrance_area)

    # => Processing FPS
    if global_frame_count % 100 == 0:
        elapsed_time = (datetime.datetime.now() - fps_start).total_seconds()
        estimated_fps = fps_frame_count / elapsed_time
        fps_frame_count = 0
        fps_start = datetime.datetime.now()
        print("Estimated fps over 100 frames: %s" % estimated_fps)

    # => Show frame
    show_image_opencv(frame, "debug")
    print(time_measurement.timeMap)
    time_measurement.reset()

f.close()