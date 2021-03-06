import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition

# Load face encodings
with open('dataset_faces.dat', 'rb') as f:
    all_face_encodings = pickle.load(f)


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


def add_face_encodings(folder_name):
    for file_name in os.listdir("C:/Users/jabez/PycharmProjects/face_attendance/" + folder_name):
        image = face_recognition.load_image_file(folder_name + '/' + file_name)
        try:
            person_name = file_name[:-10]
            if person_name in all_face_encodings.keys():
                if len(all_face_encodings[person_name]) == 128:
                    all_face_encodings[person_name] = [all_face_encodings[person_name],
                                                       face_recognition.face_encodings(image)[0]]
                else:
                    (all_face_encodings[person_name]).append(face_recognition.face_encodings(image)[0])
                print("added for re-training")
            else:
                all_face_encodings[person_name] = face_recognition.face_encodings(image)[0]
                print("added to database")
        except IndexError:
            print("skipping image bc of poor details proceed to next retry identification processing")
    file_list = [f for f in os.listdir("C:/Users/jabez/PycharmProjects/face_attendance/" + folder_name)]
    for f in file_list:
        os.remove(os.path.join("C:/Users/jabez/PycharmProjects/face_attendance/" + folder_name, f))


face_encoding_to_insert = (all_face_encodings['jabez'])[0]
(all_face_encodings['jabez']).append(face_encoding_to_insert)
#  pickle face encodings
with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)
print("Training KNN classifier...")
X = []
y = []
for keys in all_face_encodings:
    print(len(all_face_encodings[keys]))
    if len(all_face_encodings[keys]) < 128:
        for i in range(0, len(all_face_encodings[keys])):
            y.append(keys)
            X.append((all_face_encodings[keys])[i])
    elif len(all_face_encodings[keys]) == 128:
        y.append(keys)
        X.append(all_face_encodings[keys])
print(y)
classifier = train(model_save_path="trained_knn_model.clf", n_neighbors=2)
print("Training complete!")
