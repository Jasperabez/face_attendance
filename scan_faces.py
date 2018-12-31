import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import hashids
import uuid


# save matches, rechecks, and unknowns to respective folder for human decision making
def save_to_file(loc, X_img, file_name, name=None):
    top, right, bottom, left = loc
    face_img = X_img[top - 15:bottom + 15, left - 15:right + 15]
    pil_image = Image.fromarray(face_img)
    if name is None:
        pil_image.save(file_name + "/" + (hashids.Hashids(salt=str(uuid.uuid4()), )).encode(1, 2, 3) + ".jpg")
    else:
        pil_image.save(file_name + "/" + name + (hashids.Hashids(salt=str(uuid.uuid4()), )).encode(1, 2, 3) + ".jpg")


# get person's name and loc, including rechecks and unknown also save the images to respective folders
def predict(X_img_path, model_path=None, distance_threshold=0.36, recheck_distance_threshold=0.44, n_neighbors=1):
    if model_path is None:
        raise Exception("Must supply knn classifier either through model_path")

    # Load a trained KNN model (if one was passed in)
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the matches and close matches for faces
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=n_neighbors)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    are_close_matches = [distance_threshold < closest_distances[0][i][0] <= recheck_distance_threshold for i in
                         range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return_value = []
    for pred, loc, closet, recheck in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches,
                                          are_close_matches):
        if closet:
            return_value.append((pred, loc))
            save_to_file(loc, X_img, "matches", pred)
        elif recheck:
            return_value.append((pred + '?', loc))
            save_to_file(loc, X_img, "recheck", pred)
        else:
            return_value.append(("unknown", loc))
            save_to_file(loc, X_img, "unknown")
    return return_value


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


# get face encoding for unknown image, using knn clf
img_folder = 'pictures'
for img_path in image_files_in_folder(img_folder):
    y_predict = predict(img_path, model_path="trained_knn_model.clf", )
