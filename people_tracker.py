import cv2

from RecognizedPerson import RecognizedPerson
from people_db import PeopleDB
import numpy as np


class PeopleTracker:

    def __init__(self, alignedReIdModel, people_db, distance_th=0.6, frames_memory=10):
        self.alignedReIdModel = alignedReIdModel
        self.people_db = PeopleDB(alignedReIdModel)
        self.tracker_memory = []
        self.current_n_frame = None
        self.distance_th = distance_th
        self.frames_memory = frames_memory
        self.people_db = people_db

    def _new_person(self, person_loc, person_emb, person_img, person_face, n_frame):
        faces_locs, faces_embs, face_recog = person_face
        person = RecognizedPerson(person_loc, person_emb, person_img, faces_locs, face_recog, n_frame)
        self.tracker_memory.append(person)

    def do_tracking(self, people_locs, people_embs, img_people, faces_people, n_frame):

        self.current_n_frame = n_frame

        # => Track people
        for person_loc, person_emb, person_img, person_face in zip(people_locs, people_embs, img_people, faces_people):

            # - Empty tracker
            if len(self.tracker_memory) == 0:
                self._new_person(person_loc, person_emb, person_img, person_face, n_frame)

            # - Tracker with memory
            else:
                # · Min emb. distance
                mem_embeddings = [person.last_emb for person in self.tracker_memory]
                distances = np.array([self.alignedReIdModel.get_distance_from_emb(emb, person_emb) for emb in mem_embeddings])
                min_dist_idx = int(np.argmin(distances))
                min_dist = distances[min_dist_idx]

                if min_dist < self.distance_th:
                    faces_locs, faces_embs, face_recog = person_face
                    self.tracker_memory[min_dist_idx].add_tracked_information(person_loc, person_emb, person_img, faces_locs, face_recog, n_frame)
                else:
                    self._new_person(person_loc, person_emb, person_img, person_face, n_frame)

        # => Process tracker memory
        tracked = [person for person in self.tracker_memory if person.n_frame < n_frame - self.frames_memory]
        # for person in tracked:
        #     if len([f for f in person.faces_locations if len(f) > 0]) > 5:
        #         self.people_db.insert_person(person)
        #     else:
        #         self.people_db.recognize_person(person.emb_memory)


        self.tracker_memory = [person for person in self.tracker_memory if person.n_frame >= n_frame - self.frames_memory]

    def draw_information(self, frame, n_frame):

        people_last_frame = [person for person in self.tracker_memory if person.n_frame == n_frame]
        for person in people_last_frame:
            color = person.color
            # - Person location
            top, right, bottom, left = person.person_loc
            cv2.rectangle(frame, (left, top), (left + (right - left), top + (bottom - top)), color, 2)
            # - Person trajectory
            for l in person.trajectory:
                top, right, bottom, left = l
                m = (int((right + left) / 2), int((bottom)))
                cv2.circle(frame, m, 5, color, -1)
            # - Person face
            for (top, right, bottom, left), recog in zip(person.face_loc, person.faces_recognitions[-1]):
                cv2.rectangle(frame, (left, top), (left + (right - left), top + (bottom - top)), color, 2)
                height, width = frame.shape[:2]
                small_name = recog[:18]
                if len(recog) != len(small_name):
                    small_name += '...'
                box_w = 25 + int(len(small_name) * 12.5)
                box_h = 30
                box_x = left + int((right - left) / 2) - int(box_w / 2)
                box_y = bottom + 40
                if box_y + box_h > height:
                    box_y = height - box_h
                cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, small_name, (box_x + 20, box_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)




