import math
import numpy
from sklearn import neighbors
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

# Load face encodings
with open('dataset_faces.dat', 'rb') as f:
    all_face_encodings = pickle.load(f)


def train(model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(train_X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(train_X, train_y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(knn_clf=None, model_path=None, distance_threshold=0.44, n_neighbours=None):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param n_neighbours:
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Find encodings for faces in the test iamge
    faces_encodings = test_X

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=n_neighbours)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(test_y))]

    # Predict classes and remove classifications that aren't within the threshold
    return [pred if rec else "unknown" for pred, rec in
            zip(knn_clf.predict(faces_encodings), are_matches)]


print("Training KNN classifier...")
X = []
y = []
for keys in all_face_encodings:
    if len(all_face_encodings[keys]) < 128:
        for i in range(0, len(all_face_encodings[keys])):
            y.append(keys)
            X.append((all_face_encodings[keys])[i])
    elif len(all_face_encodings[keys]) == 128:
        y.append(keys)
        X.append(all_face_encodings[keys])
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=101)

error = []
# Calculating error for K values
for i in range(1, 13):
    classifier = train(model_save_path="test_accuracy_knn_model.clf", n_neighbors=i)
    predictions = predict(model_path="test_accuracy_knn_model.clf", n_neighbours=i)
    print(predictions)
    print(test_y)
    error.append(numpy.mean(predictions != test_y))

plt.plot(range(1, 13), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

