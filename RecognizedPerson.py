

class RecognizedPerson:

    def __init__(self):

        # - UID
        self.id = None
        self.color = None

        # - Time related properties
        self.n_frame = None
        self.n_frames_memory = []

        # - Person detection/recognition related properties
        self.last_emb = None
        self.bbox = None
        self.emb_memory = []
        self.trajectory = []
        self.person_img = []

        # - Faces detection/recognition related properties
        self.faces_locations = []
        self.faces_recognitions = []
