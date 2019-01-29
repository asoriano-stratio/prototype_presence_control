import numpy as np


class RecognizedPerson:

    def __init__(self, person_loc, person_emb, person_img, face_loc, face_recog, n_frame):

        # - UID
        self.id = None
        self.color = tuple(int(f) for f in np.random.choice(range(256), size=3))

        # - Time related properties
        self.n_frame = n_frame
        self.n_frames_memory = [n_frame]

        # - Person detection/recognition related properties
        self.last_emb = person_emb
        self.person_loc = person_loc
        self.trajectory = [person_loc]
        self.emb_memory = [person_emb]
        self.person_img = [person_img]

        # - Faces detection/recognition related properties
        self.face_loc = face_loc
        self.faces_locations = [face_loc]
        self.faces_recognitions = [face_recog]
        self.name = None

    def add_tracked_information(self, person_loc, person_emb, person_img, face_loc, face_recog, n_frame):

        self.n_frame = n_frame
        self.n_frames_memory.append(n_frame)

        self.person_loc = person_loc
        self.trajectory.append(person_loc)

        self.emb_memory.append(person_emb)
        self.person_img.append(person_img)
        self.face_loc = face_loc
        self.faces_locations.append(face_loc)
        self.faces_recognitions.append(face_recog)
