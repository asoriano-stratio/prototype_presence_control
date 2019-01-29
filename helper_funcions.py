from copy import deepcopy

import cv2
import numpy as np
import tensorflow as tf
import nets.resnet_v1_50 as model
import heads.fc1024 as head
import time
import os
import face_recognition

id = 0


def detect_faces(face_detector_model, frame, face_recognition_model):

    if face_detector_model == "hog" or face_detector_model == "cnn":
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return face_recognition.face_locations(frame_rgb, number_of_times_to_upsample=0, model=face_detector_model)

    elif face_detector_model == "haar":
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_detector = face_recognition_model["haar"]
        faces = haar_detector.detectMultiScale(frame_gray, 1.3, 5)
        locations = []
        for (x, y, w, h) in faces:
            # (top, right, bottom, left)
            locations.append((y, x + w, y + h, x))
        return locations


def do_face_recognition(img, face_recognition_model, loc):
    knn = face_recognition_model["knn"]
    names = face_recognition_model["names"]

    locations = detect_faces("hog", img, face_recognition_model)
    faces_locations = []
    for img_loc in locations:
        i_top, i_right, i_bottom, i_left = loc
        top, right, bottom, left = img_loc
        faces_locations.append((top+i_top, right+i_left, bottom+i_top, left+i_left))


    recognitions = []
    faces_encodings = []
    if len(locations) > 0:
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces_encodings = face_recognition.face_encodings(frame_rgb, locations)
        distances, indices = knn.kneighbors(faces_encodings)
        for d, idx in zip(distances, indices):
            if d[0] < 0.6:
                recognitions.append(names[idx[0]])
            else:
                recognitions.append("unknown")
        #print(recognitions)
    return faces_locations, faces_encodings, recognitions


def filter_locations(locations, height, width, margin=20):

    out = []
    for l in locations:
        top, right, bottom, left = l
        if top > (0 + margin) and bottom < (height - margin) and left > (0 + margin) and right < (width - margin):
            out.append(l)
    return out

def draw_entrance_area(frame, entrance_area):

    t_l, t_r, b_l, b_r = entrance_area
    cv2.rectangle(frame, t_l, b_r, (0, 255, 255), 2)
    # cv2.circle(frame, (int((t_l[0]+b_r[0])/2), int((t_l[1]+b_r[1])/2)), 5, (0, 0, 255), -1)


def save_tracked_person_images(path, person):
    global id

    dir_path = os.path.join(path, "%s" % id)
    os.mkdir(dir_path)

    n_img = 0
    for img, loc in zip(person["img"], person["trajectory"]):
        image_path = os.path.join(dir_path, "%s.jpg" % n_img)
        cv2.imwrite(image_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        n_img += 1
    id = id + 1


def save_tracked_person_video(path, person):

    global id

    video_file = os.path.join(path, "%s.avi" % id)
    os.mkdir(os.path.join(path, "%s" % id))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(video_file, fourcc, 25.0, (640, 480))

    for img, loc in zip(person["img"], person["trajectory"]):
        blank_image = np.zeros((480, 640, 3), np.uint8)
        top, right, bottom, left = loc
        blank_image[top:bottom, left:right] = img
        writer.write(blank_image)
    writer.release()
    id = id + 1


def _get_iou(bb1, bb2):
    """ Calculate the Intersection over Union (IoU, in [0, 1]) of two bounding boxes. """

    (top1, right1, bottom1, left1) = bb1
    (top2, right2, bottom2, left2) = bb2

    # determine the coordinates of the intersection rectangle
    x_left = max(left1, left2)
    y_top = max(top1, top2)
    x_right = min(right1, right2)
    y_bottom = min(bottom1, bottom2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (right1 - left1) * (bottom1 - top1)
    bb2_area = (right2 - left2) * (bottom2 - top2)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def get_random_color():
    color = tuple(int(f) for f in np.random.choice(range(256), size=3))
    return color


def update_tracker(tracker_memory, people_model, locations, embeddings, img_people, faces_people, n_frame):

    def new_person(bbox, emb, img, face):
        global id

        person = {"id": id, "last_emb": emb, "memory": [], "n_frame": n_frame, "faces": [face],
                  "img": [deepcopy(img)], "bbox": bbox, "trajectory": [bbox], "tracked": 0, "color": get_random_color()}
        tracker_memory.append(person)

        id += 1

    for bbox, emb, img, face in zip(locations, embeddings, img_people, faces_people):

        # - Empty tracker
        if len(tracker_memory) == 0:
            new_person(bbox, emb, img, face)

        # - Tracker with data
        else:
            bboxes = [item["bbox"] for item in tracker_memory]
            ious = np.array([_get_iou(bb1, bbox) for bb1 in bboxes])
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            # · Bbox intersection
            if max_iou > 0.4:
                person = tracker_memory[max_iou_idx]

                # · Embeddings distance
                emb_dist = people_model.human_distance(person["last_emb"], emb)

                if emb_dist < 12:
                    person["memory"].append(emb)
                    person["last_emb"] = emb
                    person["n_frame"] = n_frame
                    person["bbox"] = bbox
                    person["tracked"] = person["tracked"] + 1
                    person["img"].append(deepcopy(img))
                    person["trajectory"].append(bbox)
                    person["faces"].append(face)
                else:
                    new_person(bbox, emb, img, face)
            else:
                new_person(bbox, emb, img, face)

    # - Update tracker memory: delete un-useful data
    tracked = [person for person in tracker_memory if person["n_frame"] < n_frame]
    updated_tracker = [person for person in tracker_memory if person["n_frame"] >= n_frame]
    tracker_memory = updated_tracker

    return tracker_memory, tracked


def update_tracker_v2(tracker_memory, alignedreid_model, locations, embeddings, img_people, faces_people, n_frame):

    def new_person(bbox, emb, img, face):
        global id

        person = {"id": id, "last_emb": emb, "memory": [], "n_frame": n_frame, "faces": [face],
                  "img": [deepcopy(img)], "bbox": bbox, "trajectory": [bbox], "tracked": 0, "color": get_random_color()}
        tracker_memory.append(person)

        id += 1

    for bbox, emb, img, face in zip(locations, embeddings, img_people, faces_people):

        # - Empty tracker
        if len(tracker_memory) == 0:
            new_person(bbox, emb, img, face)

        # - Tracker with data
        else:
            bboxes = [item["bbox"] for item in tracker_memory]
            ious = np.array([_get_iou(bb1, bbox) for bb1 in bboxes])
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            # · Bbox intersection
            if max_iou > 0.0:
                person = tracker_memory[max_iou_idx]

                # · Embeddings distance
                emb_dist = alignedreid_model.get_distance_from_emb(person["last_emb"], emb)

                if emb_dist < 0.6:
                    person["memory"].append(emb)
                    person["last_emb"] = emb
                    person["n_frame"] = n_frame
                    person["bbox"] = bbox
                    person["tracked"] = person["tracked"] + 1
                    person["img"].append(deepcopy(img))
                    person["trajectory"].append(bbox)
                    person["faces"].append(face)
                else:
                    new_person(bbox, emb, img, face)
            else:
                new_person(bbox, emb, img, face)

    # - Update tracker memory: delete un-useful data
    n_memory = 25
    tracked = [person for person in tracker_memory if person["n_frame"] < n_frame - n_memory]
    updated_tracker = [person for person in tracker_memory if person["n_frame"] >= n_frame-n_memory]
    tracker_memory = updated_tracker

    return tracker_memory, tracked

def save_people_images(path, f, n_frame, height, width, locations, images, embeddings):
    n_person = 1
    for loc, img, emb in zip(locations, images, embeddings):

        top, right, bottom, left = loc

        # Filter bbox on image borders
        if top > 0 and left > 0 and bottom < height and right < width:
            filename = "%s_%s_%s.jpg" % (str(n_frame), str(n_person), "_".join([str(l) for l in loc]))
            image_path = os.path.join(path, filename)
            cv2.imwrite(image_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            n_person += 1
            data = ",".join([filename, str(n_frame), str(top), str(right), str(bottom), str(left)])
            f.write(data + "\n")


def save_people_images_in_area(path, f, n_frame, area, locations, images, embeddings):
    n_person = 1
    for loc, img, emb in zip(locations, images, embeddings):

        top, right, bottom, left = loc
        topa, righta, bottoma, lefta = area

        # Filter bbox on image borders
        if top > topa and left > lefta and bottom < bottoma and right < righta:
            filename = "%s_%s_%s.jpg" % (str(n_frame), str(n_person), "_".join([str(l) for l in loc]))
            image_path = os.path.join(path, filename)
            cv2.imwrite(image_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            n_person += 1
            data = ",".join([filename, str(n_frame), str(top), str(right), str(bottom), str(left)])
            f.write(data + "\n")


def image_equalization_clahe(clahe, frame):
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


class TimeMeasurement:

    def __init__(self, timers_names=[]):
        # Initialization of timerDict
        self.timeMap = {}
        for timer_name in timers_names:
            self.timeMap[timer_name] = None

    def reset(self, timers_names=[]):
        self.__init__(timers_names)

    def initTimer(self, timerName):
        self.timeMap[timerName] = (time.time(), -1)
        return self.timeMap

    def measureTime(self, timerName):
        self.timeMap[timerName] = time.time() - self.timeMap[timerName][0]
        return self.timeMap


class PeopleFeatures:

    def __init__(self, init_emb_model=True):
        self.tf_sess = None
        self.endpoints = None
        self.tf_input_image = None

        people_det_model_prototxt_path = "/home/asoriano/Escritorio/spaceai-evolution/Models/MobileNetSSD_caffe/deploy.prototxt"
        people_det_model_caffemodel_path = "/home/asoriano/Escritorio/spaceai-evolution/Models/MobileNetSSD_caffe/MobileNetSSD_deploy.caffemodel"
        people_embeddings_path = "/home/asoriano/Escritorio/spaceai-evolution/Models/person_embedding_tf/checkpoint-25000"

        # - Reading people detection model
        self.people_det_model = cv2.dnn.readNetFromCaffe(people_det_model_prototxt_path, people_det_model_caffemodel_path)
        self.people_det_model_class_names = {0: 'background', 15: 'person'}

        # - People embeddings model
        if init_emb_model:
            tf.Graph().as_default()
            self.tf_sess = tf.Session()
            self.tf_input_image = tf.zeros([1, 256, 128, 3], dtype=tf.float32)
            endpoints, body_prefix = model.endpoints(self.tf_input_image, is_training=False)
            with tf.name_scope('head'):
                self.endpoints = head.head(endpoints, 128, is_training=False)
            tf.train.Saver().restore(self.tf_sess, people_embeddings_path)

    def human_locations(self, frame, thr=0.5, ):
        frame_resized = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        self.people_det_model.setInput(blob)
        # Prediction of network
        detections = self.people_det_model.forward()

        # Size of frame resize (300x300)
        cols = frame_resized.shape[1]
        rows = frame_resized.shape[0]
        output = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence of prediction
            if confidence > thr:  # Filter prediction
                class_id = int(detections[0, 0, i, 1])  # Class label

                # Object location
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                frame_height = frame.shape[0]
                frame_width = frame.shape[1]

                # Factor for scale to original size of frame
                heightFactor = frame_height / 300.0
                widthFactor = frame_width / 300.0
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom)
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop = int(widthFactor * xRightTop)
                yRightTop = int(heightFactor * yRightTop)

                xLeftBottom = xLeftBottom if xLeftBottom > 0 else 0
                yLeftBottom = yLeftBottom if yLeftBottom > 0 else 0
                xRightTop = xRightTop if xRightTop < frame_width else frame_width
                yRightTop = yRightTop if yRightTop < frame_height else frame_height

                if class_id in self.people_det_model_class_names:
                    output.append((yLeftBottom, xRightTop, yRightTop, xLeftBottom))

        return output

    def human_vector(self, img):
        resize_img = cv2.resize(img, (128, 256))
        resize_img = np.expand_dims(resize_img, axis=0)
        emb = self.tf_sess.run(self.endpoints['emb'], feed_dict={self.tf_input_image: resize_img})

        return emb

    def human_distance(self, enc1, enc2):
        return np.sqrt(np.sum(np.square(enc1 - enc2)))

    def crop_human(self, frame, locations):
        human_image = []
        for top, right, bottom, left in locations:
            sub_frame = frame[top:bottom, left:right, :]  # 3 channel image
            human_image.append(deepcopy(sub_frame))

        return human_image
