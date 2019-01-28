from sklearn.neighbors import NearestNeighbors


class PeopleDB:

    def __init__(self, alignedReIdModel):

        self.names = []
        self.embeddings = []
        self.metadata = []
        self.alignedReIdModel = alignedReIdModel
        self.knn_model = None

    def insert_person(self, person):



        knn_model = NearestNeighbors(
            n_neighbors=1, algorithm="auto", metric=self.alignedReIdModel.get_distance_from_emb).fit(train_db_emb)

