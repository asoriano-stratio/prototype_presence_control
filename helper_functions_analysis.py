from collections import Counter, defaultdict
import face_recognition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as imreader
import time
from sklearn.metrics import euclidean_distances, f1_score, accuracy_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
import itertools

# => Add a new function to Pandas dataframe object: set the diagonal values of a dataframe

def set_diag(self, values):
    n = min(len(self.index), len(self.columns))
    self.values[[np.arange(n)] * 2] = values
pd.DataFrame.set_diag = set_diag


def get_faces_under_th(names, embeddings, th):
    """ Returns all matches which distance is less than a threshold """

    d_mat, df = get_distance_matrix(names, embeddings)
    df.set_diag(None)
    i1, i2 = np.where(np.triu(df < th) == True)
    for n1, n2 in zip(i1, i2):
        print("Possible duplicate: %s - %s" % (names[n1], names[n2]))


def get_distance_matrix(names, embeddings):
    """ Returns the distance matrix between an array of vectors """

    euclidean_distances(embeddings, embeddings)
    return euclidean_distances(embeddings, embeddings)


def get_tri_distance_matrix(names, embeddings):
    d_matrix = np.triu(euclidean_distances(embeddings, embeddings))
    df = pd.DataFrame(data=d_matrix, index=names, columns=names)
    return d_matrix, df


def min_distance_classifier(known_face_encodings, query_face_encoding, tolerance, names):
    """ Minimum distance search """

    # · Min dist. search
    start_time = time.time()
    distances = face_recognition.face_distance(known_face_encodings, query_face_encoding)
    pos = distances.argmin()

    # · Execute decision
    if distances[pos] < tolerance:
        elapsed_time = time.time() - start_time
        return names[pos], elapsed_time
    else:
        elapsed_time = time.time() - start_time
        return "?", elapsed_time


def knn_classifier_decision(face_names, distances, indices, tolerance):
    names = []
    for idx, d in zip(indices, distances):
        if d < tolerance:
            names.append(face_names[idx])
        else:
            names.append("?")
    c = Counter(names)
    most_common, num_most_common = c.most_common(1)[0]

    return most_common

def knn_classifier(known_face_encodings, query_face_encoding, tolerance, face_names, k, algorithm):
    """ Knn classifier """

    # · Train classifier
    knn = NearestNeighbors(n_neighbors=k, algorithm=algorithm, metric='euclidean').fit(known_face_encodings)

    start_time = time.time()    
    
    # · Knn search
    distances, indices = knn.kneighbors(query_face_encoding.reshape(1, -1))  # Only one sample to classify
    distances = distances[0]
    indices = indices[0]
    
    # · Execute decision
    most_common = knn_classifier_decision(face_names, distances, indices, tolerance)
    elapsed_time = time.time() - start_time

    return most_common, elapsed_time, distances, indices


def distances_distributions(face_names, face_embeddings):
    dist_same = []
    dist_dist = []
    dist_matrix = get_distance_matrix(face_names, face_embeddings)
    for i in range(len(face_names) - 1):
        name_a = face_names[i]
        for j in range(i + 1, len(face_names)):
            name_b = face_names[j]
            distance = dist_matrix[i, j]
            if name_a == name_b:
                dist_same.append(distance)
            else:
                dist_dist.append(distance)

    return dist_same, dist_dist


def show_distance_distribution(distances, alpha=0.5, bins=10,
                               label='Distance distribution', title="Distance distributions over same person images"):
    plt.hist(distances, bins, alpha=alpha, density=True, label=label)
    plt.title(title)
    plt.show()


def show_distances_distributions(face_names, face_embeddings):

    dist_same, dist_dist = distances_distributions(face_names, face_embeddings)

    plt.hist(dist_same, 10, alpha=0.5, density=True, label='dist_same')
    plt.hist(dist_dist, 10, alpha=0.5, density=True, label='dist_dist')
    plt.legend(loc='upper right')
    plt.show()


# Split in unknown and known people
def split_unknown(unknown_persons, db_face_names, db_face_embeddings, db_img_paths, db_img_sizes):

    faces_unknown = []
    face_embeddings_unknown = []
    path_unknown = []
    sizes_unknown = []
    for u in unknown_persons:
        u_idx = [i for i, n in enumerate(db_face_names) if n == u]
        names = [u for _ in range(len(u_idx))]
        emb = [db_face_embeddings[i] for i in u_idx]
        faces_unknown.extend(names)
        face_embeddings_unknown.extend(emb)
        path_unknown.extend([db_img_paths[i] for i in u_idx])
        sizes_unknown.extend([db_img_sizes[i] for i in u_idx])

        db_face_names = [i for j, i in enumerate(db_face_names) if j not in u_idx]
        db_face_embeddings = [i for j, i in enumerate(db_face_embeddings) if j not in u_idx]
        db_img_paths = [i for j, i in enumerate(db_img_paths) if j not in u_idx]
        db_img_sizes = [i for j, i in enumerate(db_img_sizes) if j not in u_idx]

    return db_face_names, db_face_embeddings, db_img_paths, db_img_sizes, faces_unknown, face_embeddings_unknown, path_unknown, sizes_unknown


def get_unique_names_identifiers(face_names):
    # · Count number of embeddings per face
    counts = Counter(face_names)

    # · Assign a unique name per embedding (adding a suffix to every repeated face name)
    face_names_unique = face_names.copy()
    unique_to_name_mapper = {}
    name_to_unique_mapper = defaultdict(list)
    for s, num in counts.items():
        if num > 1:  # ignore strings that only appear once
            for suffix in range(1, num + 1):  # suffix starts at 1 and increases by 1 each time
                face_names_unique[face_names_unique.index(s)] = s + "_" + str(suffix)  # replace each appearance of s
                unique_to_name_mapper[s + "_" + str(suffix)] = s
                name_to_unique_mapper[s].append(s + "_" + str(suffix))

    return face_names_unique, unique_to_name_mapper, name_to_unique_mapper


def get_performance(k, thresholds, db_names, db_embeddings, names, embeddings, unknown_label):

    def get_measures(k, tolerance, db_names, db_embeddings, names, embeddings, unknown_label):
        # For each face
        labeled_name = []
        predicted_name = []
        all_distances = []
        all_indices = []

        for f_idx, original_name in enumerate(names):
            labeled_name.append(original_name)
            query_embedding = embeddings[f_idx]
            # => Predictions
            name_knn, t_knn, distances, indices = knn_classifier(db_embeddings, query_embedding, tolerance, db_names, k, "auto")
            predicted_name.append(name_knn if name_knn != "?" else unknown_label)
            all_distances.append(distances)
            all_indices.append(indices)

        f1 = f1_score(labeled_name, predicted_name, average='weighted')
        accuracy = accuracy_score(labeled_name, predicted_name)

        return f1, accuracy, labeled_name, predicted_name, all_distances, all_indices

    f1_th = []
    accuracy_th = []
    labeled_name_th = []
    predicted_name_th = []
    all_distances_th = []
    all_indices_th = []
    for th in thresholds:
        f1, accuracy, labeled_name, predicted_name, all_distances, all_indices = \
            get_measures(k, th, db_names, db_embeddings, names, embeddings, unknown_label)
        f1_th.append(f1)
        accuracy_th.append(accuracy)
        labeled_name_th.append(labeled_name)
        predicted_name_th.append(predicted_name)
        all_distances_th.append(all_distances)
        all_indices_th.append(all_indices)

    opt_idx = np.argmax(f1_th)
    # Threshold at maximal F1 score
    opt_tau = thresholds[opt_idx]

    # Plot F1 score and accuracy as function of distance threshold
    plt.plot(thresholds, f1_th, label='F1 score')
    plt.plot(thresholds, accuracy_th, label='Accuracy')

    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Th.')
    plt.xlabel('Distance threshold')
    plt.legend()
    plt.show()

    return opt_tau, f1_th, accuracy_th, labeled_name_th[opt_idx], predicted_name_th[opt_idx], all_distances_th[opt_idx], all_indices_th[opt_idx]


def analyze_face_cluster(faces_names, faces_embeddings, image_paths, images_sizes):

    dist_matrix = np.array(get_distance_matrix(faces_names, faces_embeddings))
    faces_clusters = []
    for name in set(faces_names):
        cluster = {}
        indexes = np.array([i for i, n in enumerate(faces_names) if name == n])
        n_d_matrix = dist_matrix[indexes[:, None], indexes]
        cluster["name"] = name
        cluster["indexes"] = indexes
        cluster["num_samples"] = len(indexes)
        cluster["distance_matrix"] = n_d_matrix
        cluster["images_analysis"] = []
        for i in range(indexes.size):
            image_analysis = {}
            index = list(range(indexes.size))
            distances = n_d_matrix[i, :]
            distances = np.delete(distances, i)
            del index[i]
            idx_min = np.argmin(distances)
            min_d = distances[idx_min]
            idx_min = index[idx_min]
            img_index = indexes[i]
            other_indexes = list(indexes[index])
            image_analysis["index"] = img_index
            image_analysis["size"] = images_sizes[img_index]
            image_analysis["path"] = image_paths[img_index]
            image_analysis["distances"] = distances
            image_analysis["sizes"] = np.array(images_sizes)[other_indexes]
            image_analysis["indexes"] = other_indexes
            image_analysis["others_path"] = np.array(image_paths)[other_indexes]
            image_analysis["min_distance"] = {"index": indexes[idx_min], "distance": min_d, "path": image_paths[indexes[idx_min]], "size": images_sizes[indexes[idx_min]]}
            cluster["images_analysis"].append(image_analysis)

        faces_clusters.append(cluster)

    return faces_clusters


def show_two_images(path1, path2, title=None):
    f, axarr = plt.subplots(1, 2)
    f.show()
    axarr[0].imshow(imreader.imread(path1))
    if title:
        axarr[0].set_title(title)
    axarr[1].imshow(imreader.imread(path2))



def draw_face_clusters(face_clusters, d):
    for cluster in face_clusters:
        for image_analysis in cluster['images_analysis']:
            if image_analysis['min_distance']["distance"] > d:
                show_two_images(image_analysis["path"], image_analysis["min_distance"]["path"])
                

def plot_confusion_matrix(cm, classes,normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
def get_knn_output(knn_model, tolerance, face_names_db, faces_encodings):
    """ K-nearest neighbours search """

    # · Getting the k-nearest neighbors
    distances, indices = knn_model.kneighbors(faces_encodings)

    # · Building output information
    knn_outputs = []
    for i in range(len(faces_encodings)):
        # · Distances to neighbors
        knn_output = {"distances": distances[i]}
        knn_output["indices"] = indices[i]
        knn_output["threshold"] = tolerance
        
        names = []
        normalized_distance_to_th = []
        known = defaultdict(list)
        sum_d = 0
        n_unk = 0
        for idx, d in zip(indices[i], distances[i]):
            if d < tolerance:
                names.append(face_names_db[idx])
                known[face_names_db[idx]].append(d)
                sum_d += d
                normalized_distance_to_th.append((tolerance-d)/tolerance)
            else:
                names.append("?")
                normalized_distance_to_th.append((d-tolerance)/(1-tolerance))
                n_unk += 1
        knn_output["possible_names"] = names
        knn_output["normalized_distance_to_th"] = normalized_distance_to_th
        # · Getting probabilities of being unknown or known
        knn_output["unk_prob"] = n_unk / len(distances[i])
        knn_output["known_prob"] = 1.0 - n_unk / len(distances[i])

        # · Final decision: Unknown or known with name
        if knn_output["unk_prob"] > 0.5:
            knn_output["name"] = "?"
        else:
            known = {name: np.sum(np.array(dists)/sum_d) for name, dists in known.items()}
            idx = np.argmax(list(known.values()))
            knn_output["known_name_prob"] = known
            knn_output["name"] = list(known.keys())[idx]

        knn_outputs.append(knn_output)

    return knn_outputs    
   
    
def get_performance_v2(k, thresholds, db_names, db_embeddings, names, embeddings, unknown_label):

    def get_measures(knn_model, tolerance, db_names, db_embeddings, names, embeddings, unknown_label):
        knn_output = get_knn_output(knn_model, tolerance, db_names, embeddings)
        predicted_name = [ k_out['name'] if k_out['name'] != "?" else unknown_label for k_out in knn_output]
        f1 = f1_score(names, predicted_name, average='weighted')
        accuracy = accuracy_score(names, predicted_name)
        precission = precision_score(names, predicted_name, average='weighted')
        recall = recall_score(names, predicted_name, average='weighted')
        
        return f1, accuracy, precission, recall, names, predicted_name, knn_output    
    
    # Training KNN model
    knn_model = NearestNeighbors(n_neighbors=k, algorithm="auto", metric='euclidean').fit(db_embeddings)
    
    f1_th = []
    accuracy_th = []
    precission_th = [] 
    recall_th = []
    labeled_name_th = []
    predicted_name_th = []
    knn_outputs_th = []
    for th in thresholds:
        f1, accuracy, precission, recall, labeled_name, predicted_name, knn_outputs = \
            get_measures(knn_model, th, db_names, db_embeddings, names, embeddings, unknown_label)
        f1_th.append(f1)
        accuracy_th.append(accuracy)
        precission_th.append(precission)
        recall_th.append(recall)
        labeled_name_th.append(labeled_name)
        predicted_name_th.append(predicted_name)
        knn_outputs_th.append(knn_outputs)

    opt_idx = np.argmax(f1_th)
    # Threshold at maximal F1 score
    opt_tau = thresholds[opt_idx]

    # Plot F1 score and accuracy as function of distance threshold
    plt.plot(thresholds, f1_th, label='F1 score')
    plt.plot(thresholds, accuracy_th, label='Accuracy')
    plt.plot(thresholds, precission_th, label='Precission')
    plt.plot(thresholds, recall_th, label='Recall')

    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Th.')
    plt.xlabel('Distance threshold')
    plt.legend()
    plt.show()

    return opt_tau, f1_th, accuracy_th, precission_th, recall_th, labeled_name_th[opt_idx], predicted_name_th[opt_idx], knn_outputs_th[opt_idx]


def error_analysis(labeled_name, predicted_name, knn_output):
    analysis = {}
    analysis["unrecognized_errors"] = []
    analysis["misclassified_errors"] = []
    analysis["false_positive_errors"] = []
    analysis["true_recognized"] = []
    analysis["true_unknown"] = []

    for test_idx, (tag, pred, knn_output) in enumerate(zip(labeled_name, predicted_name, knn_output)):
        if tag != pred:
            if tag == "unknown":
                analysis["false_positive_errors"].append((test_idx, knn_output))
            if tag != "unknown" and pred != "unknown":
                analysis["misclassified_errors"].append((test_idx, knn_output))
            if pred == "unknown":
                analysis["unrecognized_errors"].append((test_idx, knn_output))
        else:
            if tag == "unknown":
                analysis["true_unknown"].append((test_idx, knn_output))
            else:
                analysis["true_recognized"].append((test_idx, knn_output))
                
    t = len(analysis["false_positive_errors"]) + len(analysis["unrecognized_errors"]) + len(analysis["misclassified_errors"])


    print("Total evaluation samples: " + str(len(labeled_name)))
    print("Total errors: " + str(t))
    print("Correct recognized: " + str(len(analysis["true_recognized"])))
    print("Correct unknown: " + str(len(analysis["true_unknown"])))
    print("Error - unrecognized (someone --> unknown): " + str(len(analysis["unrecognized_errors"])))
    print("Error - missclasified (someone --> other): " + str(len(analysis["misclassified_errors"])))
    print("Error - false positive (unknown --> someone): " + str(len(analysis["false_positive_errors"])))
    
    return analysis


def draw_error_analysis(analysis, train_img_paths, train_img_sizes, eval_img_paths, eval_img_sizes):
    print("-----------------------------------------------------------")
    print(" => Unrecognized errors")
    print("-----------------------------------------------------------")

    for unrecognized_error in analysis["unrecognized_errors"]:
        test_idx, knn_output = unrecognized_error
        plt.imshow(plt.imread(eval_img_paths[test_idx]))
        plt.title("Idx: " + str(test_idx) + " " + eval_img_paths[test_idx])
        plt.xlabel(eval_img_sizes[test_idx])
        plt.show()
        print("_________________________________________________________________________________________")            

    print("-----------------------------------------------------------")
    print(" => Misclassified errors")
    print("-----------------------------------------------------------")

    for misclassified_error in analysis["misclassified_errors"] :
        test_idx, knn_output = misclassified_error
        k = len(knn_output["indices"])
        f, ax = plt.subplots(1, k+1, figsize=(k*10,6)) 
        ax[0].imshow( plt.imread((eval_img_paths[test_idx])) )
        ax[0].set_xlabel(eval_img_sizes[test_idx])
        ax[0].set_title("Idx: " + str(test_idx))
        for i, idx in enumerate(knn_output["indices"]):
            ax[i+1].imshow( plt.imread((train_img_paths[idx])) ) 
            ax[i+1].set_xlabel(train_img_sizes[idx])
            ax[i+1].set_title("Idx: " + str(idx))
        plt.show()
        print("_________________________________________________________________________________________")            

    print("-----------------------------------------------------------")
    print(" => False positive errors")
    print("-----------------------------------------------------------")

    for false_positive_error in analysis["false_positive_errors"]:
        test_idx, knn_output = false_positive_error
        k = len(knn_output["indices"])
        f, ax = plt.subplots(1, k+1, figsize=(k*10,6)) 
        ax[0].imshow( plt.imread((eval_img_paths[test_idx])) )
        ax[0].set_xlabel(eval_img_sizes[test_idx])
        ax[0].set_title("Idx: " + str(test_idx))
        for i, idx in enumerate(knn_output["indices"]):
            ax[i+1].imshow( plt.imread((train_img_paths[idx])) ) 
            ax[i+1].set_xlabel(train_img_sizes[idx])
            ax[i+1].set_title("Idx: " + str(idx))
        plt.show()
        print("_________________________________________________________________________________________")            