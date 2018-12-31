import face_recognition
import pickle
import math
from sklearn import neighbors


def train(model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


all_face_encodings = {}

image = face_recognition.load_image_file("randomImage.jpg")

all_face_encodings["randomDude"] = face_recognition.face_encodings(image)[0]

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
classifier = train(model_save_path="trained_knn_model.clf", n_neighbors=1)
print("Training complete!")

with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)
