from collections import Counter
import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools

from helper_funcions import TimeMeasurement


class PeopleDB:

    def __init__(self, alignedReIdModel):

        self.names = []
        self.people_embeddings = []
        self.metadata = []
        self.alignedReIdModel = alignedReIdModel
        self.knn_model = None
        self.time_measurement = TimeMeasurement(timers_names=["insert_person", "recognize_person"])

    def insert_person(self, person):
        self.time_measurement.initTimer("insert_person")

        # TODO - Ensure only one face rec and loc
        person.name = self.do_recognition(list(itertools.chain(*person.faces_recognitions)))
        print(" =======>> %s has been learnt" % person.name)

        self.names.extend(len(person.emb_memory)*[person.name])
        self.people_embeddings.extend(person.emb_memory)

        self.knn_model = NearestNeighbors(
            n_neighbors=1, algorithm="auto", metric=self.alignedReIdModel.get_distance_from_emb
        ).fit(self.people_embeddings)

        self.time_measurement.measureTime("insert_person")

    def do_recognition(self, recognitions):
        c = Counter(recognitions)
        most_common, num_most_common = c.most_common(1)[0]

        return most_common

    def recognize_person(self, person_emb):

        if self.knn_model is not None:
            distances, indices = self.knn_model.kneighbors(person_emb)
            predictions = np.array(self.names)[indices]
            decisions = []
            c = Counter(predictions.flatten().tolist())
            most_common, num_most_common = c.most_common(1)[0]
            decisions.append(most_common)

            print(" =======>> %s has been recognized by body features" % decisions)

            return decisions

        return None
