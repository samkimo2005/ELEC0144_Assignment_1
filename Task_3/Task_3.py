# Code Map

# Step 0: Setup
from pathlib import Path
from unicodedata import name
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Import and preprocess the input videos

def preprocessing(video_path):
    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if not video.isOpened() or total_frames == 0:
        video.release()
        raise ValueError("Cannot open the video" + str(video))

    # Take 12 frames evenly from the middle 80% of the video.
    frame_numbers = np.linspace(0.1 * (total_frames - 1),
                                0.9 * (total_frames - 1), 12).astype(int)
    frames = []

    for number in frame_numbers:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(number))
        success, frame = video.read()
        if not success:
            video.release()
            raise ValueError("Cannot read a frame from: " + str(video_path))

        # Resize while keeping the image's original proportions.
        height, width = frame.shape[:2]
        scale = min(224 / width, 224 / height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        frame = cv2.resize(frame, (new_width, new_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add black padding to obtain a 224 x 224 image.
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        x = (224 - new_width) // 2
        y = (224 - new_height) // 2
        image[y:y + new_height, x:x + new_width] = frame

        frames.append(image / 255.0)

    video.release()
    return np.array(frames, dtype=np.float32)

# Step 2: Extract visual features from the frames of the videos
def feature_extraction(frames):
    edge_frames = [] # Restore the frames for edge detection
    gray_frames = [] # Restore the frames as gray-scale images

    for frame in frames:
        # Convert RGB frames into gray-scale
        image = (frame * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        gray_frames.append(gray.astype(np.float32)/255.0)

        # Detect the edges of the fingers and objects
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        edges = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA) # Resize to reduce the unnecessary features
        edge_frames.append(edges)

    # Feature 1: average positions of edges throughout the grip.
    edge_mean = np.mean(edge_frames, axis = 0)

    # Feature 2: change in edges between the start and the end of the grip
    first_edges = np.mean(edge_frames[:3], axis = 0)
    last_edges = np.mean(edge_frames[-3:], axis = 0)
    edge_change = np.abs(last_edges - first_edges)

    # Feature 3: Average movement between the start and the end of the object
    movements = []
    for i in range(1, len(gray_frames)):
        difference = np.abs(gray_frames[i] - gray_frames[i-1])
        difference = cv2.resize(difference, (16, 16), interpolation = cv2.INTER_AREA)
        movements.append(difference)
    movement_mean = np.mean(movements, axis = 0)

    # 256 features for each 16 x 16 image, so for each grip, 256 x 3 = 768 features for each grip
    # flatten(): Convert a 2D matrix into a 1D array with the same total number of elements
    features = np.concatenate(
        (edge_mean.flatten(),
         edge_change.flatten(),
         movement_mean.flatten())
         )
    return features.astype(np.float32)

# Step 3: Construct the classification methods
# Method 1: Classification using Multilayer Propagation
def classification_mlp():
   return MLPClassifier(
                hidden_layer_sizes= (256,),
                activation = "relu",
                max_iter = 1000,
                random_state = 42
            )


# Method 2, 3 and 4: Classification using CNN: 1. basic 2. batch normalisation 3. data augmentation
def classification_cnn(method = "basic"):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3))
    ])

    # Data Augmentation
    if method == "augmentation":
        model.add(tf.keras.layers.RandomBrightness(
            0.1, value_range = (0.0, 1.0)
        ))

        model.add(tf.keras.layers.RandomContrast(0.1))

    # First Convolutional Layer
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding = "same"))

    if method == "batch":
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Second Convolutional Layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))

    if method == "batch":
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Classification layer
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Method 5: Classification using GoogLeNet (pre-trained CNN)
def classification_googlenet():
    # Load pre-trained GoogLeNet
    googlenet = hub.load(
        "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/5"
    )

    # Define the binary classifier
    classifier = LogisticRegression(
        max_iter = 1000,
        random_state  = 42
    )
    
    return googlenet, classifier

# Step 4: Training

def model_training(
        training_features, validation_features, 
        training_label, validation_label,
        training_frames, validation_frames,
        mlp, 
        cnn_basic, cnn_batchnorm, cnn_augmentation, 
        googlenet, google_classifier):

    results = {}

    # Convert the list of features into NumPy arrays
    X_train_mlp = np.array(training_features)
    X_val_mlp = np.array(validation_features)

    # Convert the labels
    y_train = np.array(training_label)
    y_val = np.array(validation_label)

    # --------------------------------------------------
    # Method 1: MLP
    # --------------------------------------------------

    X_train_mlp = np.array(training_features, dtype=np.float32)
    X_val_mlp = np.array(validation_features, dtype=np.float32)

    # Standardise features using training data only
    scaler = StandardScaler()
    X_train_mlp = scaler.fit_transform(X_train_mlp)
    X_val_mlp = scaler.transform(X_val_mlp)

    # Train MLP
    mlp.fit(X_train_mlp, y_train)

    # Validate MLP
    mlp_predictions = mlp.predict(X_val_mlp)
    mlp_accuracy = accuracy_score(y_val, mlp_predictions)

    results["MLP"] = mlp_accuracy
    print("MLP Accuracy:", mlp_accuracy)

    # --------------------------------------------------
    # Prepare RGB frames for CNNs and GoogLeNet
    # --------------------------------------------------

    # Select three frames per grip from the original 12
    X_train_cnn = np.concatenate(
        [frames[[2, 5, 9]] for frames in training_frames],
        axis=0
    ).astype(np.float32)

    X_val_cnn = np.concatenate(
        [frames[[2, 5, 9]] for frames in validation_frames],
        axis=0
    ).astype(np.float32)

    # Each grip's label is repeated for its three frames
    y_train_cnn = np.repeat(y_train, 3)
    y_val_cnn = np.repeat(y_val, 3)

    # --------------------------------------------------
    # Methods 2, 3 and 4: CNNs
    # --------------------------------------------------

    cnn_methods = {
        "Basic CNN": cnn_basic,
        "CNN + BatchNorm": cnn_batchnorm,
        "CNN + Augmentation": cnn_augmentation
    }

    for name, model in cnn_methods.items():

        print("Training:", name)

        # Train the CNN
        model.fit(
            X_train_cnn,
            y_train_cnn,
            validation_data=(X_val_cnn, y_val_cnn),
            epochs=10,
            batch_size=16
        )

        # Predict probabilities for validation frames
        probabilities = model.predict(
            X_val_cnn,
            batch_size=16,
            verbose=0
        )

        # Average three frame predictions per grip
        probabilities = probabilities.reshape(len(y_val), 3)
        grip_probabilities = np.mean(probabilities, axis=1)

        # Convert probabilities to Light (0) or Firm (1)
        predictions = (grip_probabilities >= 0.5).astype(int)

        # Calculate grip-level validation accuracy
        accuracy = accuracy_score(y_val, predictions)

        results[name] = accuracy
        print(name, "Accuracy:", accuracy)

    # --------------------------------------------------
    # Method 5: Pre-trained GoogLeNet
    # --------------------------------------------------

    print("Training: GoogLeNet")

    # Extract pre-trained visual features in small batches
    def google_features(images):
        features = []

        for i in range(0, len(images), 16):
            batch = tf.convert_to_tensor(
                images[i:i + 16],
                dtype=tf.float32
            )

            output = googlenet(batch)
            features.append(output.numpy())

        return np.concatenate(features, axis=0)

    # Extract visual features from the RGB frames
    X_train_google = google_features(X_train_cnn)
    X_val_google = google_features(X_val_cnn)

    # Average the three feature vectors belonging to each grip
    X_train_google = X_train_google.reshape(
        len(y_train), 3, -1
    ).mean(axis=1)

    X_val_google = X_val_google.reshape(
        len(y_val), 3, -1
    ).mean(axis=1)

    # Train Logistic Regression on GoogLeNet features
    google_classifier.fit(X_train_google, y_train)

    # Validate
    google_predictions = google_classifier.predict(X_val_google)
    google_accuracy = accuracy_score(y_val, google_predictions)

    results["GoogLeNet"] = google_accuracy
    print("GoogLeNet Accuracy:", google_accuracy)

    # --------------------------------------------------
    # Compare the five methods
    # --------------------------------------------------

    print("\nValidation Results:")

    for name, accuracy in results.items():
        print(name, ":", f"{accuracy:.2%}")

    best_method = max(results, key=results.get)

    print("\nResults Summary:", results)
    print("\nBest Method:", best_method)

    return scaler, best_method
    
# Step 5: Final Evaluation
def evaluation(test_label, test_objects, predictions):

    # Overall accuracy
    overall_accuracy = accuracy_score(test_label, predictions)

    # Accuracy for each object
    object_accuracies = {}

    for object_name in ["foam_ball", "rigid_bottle"]:
        mask = np.array(test_objects) == object_name

        object_accuracies[object_name] = accuracy_score(
            np.array(test_label)[mask],
            predictions[mask]
        )

    # Confusion matrix
    matrix = confusion_matrix(
        test_label, predictions, labels=[0, 1]
    )

    print("\nFinal Test Results:")
    print("Overall Accuracy:", f"{overall_accuracy:.2%}")

    for name, accuracy in object_accuracies.items():
        print(name, "Accuracy:", f"{accuracy:.2%}")

    print("Confusion Matrix:\n", matrix)

    return overall_accuracy, object_accuracies, matrix


def main():
    location = Path(__file__).resolve().parent

    training_features = []
    validation_features = []
    test_features = []

    training_frames = []
    validation_frames = []
    test_frames = []

    training_label = []
    validation_label = []
    test_label = []

    test_objects = []

    # Step 1 & 2: Preprocessing & visual feature extraction
    for folder_name in ["Training", "Validation", "Test"]:
        folder = location / "input_data" / folder_name
        
        for video_path in sorted(folder.iterdir()):
            if video_path.suffix.lower() not in (".mp4", ".mov", ".avi"):
                continue

            # Step 1: Preprocessing
            frames = preprocessing(video_path)
            print(folder_name, video_path.name, frames.shape)

            if "_light_" in video_path.stem.lower():
                label = 0
            elif "_firm_" in video_path.stem.lower():
                label = 1
            else:
                raise ValueError("Unknown label: " + video_path.name)
             
            if "foam_ball" in video_path.stem.lower():
                object_type = "foam_ball"
            elif "rigid_bottle" in video_path.stem.lower():
                object_type = "rigid_bottle"
            else:
                raise ValueError("Unknown object: " + video_path.name)

            # Step 2: Visual feature extraction
            features = feature_extraction(frames)
            
            if folder_name == "Training":
                training_features.append(features)
                training_frames.append(frames)
                training_label.append(label)

            elif folder_name == "Validation":
                validation_features.append(features)
                validation_frames.append(frames)
                validation_label.append(label)

            elif folder_name == "Test":
                test_features.append(features)
                test_frames.append(frames)
                test_label.append(label)
                test_objects.append(object_type)

    # Step 3: Classification
    mlp = classification_mlp()

    cnn_basic = classification_cnn("basic")
    cnn_batchnorm = classification_cnn("batch")
    cnn_augmentation = classification_cnn("augmentation")

    googlenet, google_classifier = classification_googlenet()
    
    # Step 4: Training
    scaler, best_method = model_training(
        training_features, validation_features, 
        training_label, validation_label,
        training_frames, validation_frames,
        mlp, 
        cnn_basic, cnn_batchnorm, cnn_augmentation, 
        googlenet, google_classifier)

# Step 5: Evaluation
    models = {
    "MLP": mlp,
    "Basic CNN": cnn_basic,
    "CNN + BatchNorm": cnn_batchnorm,
    "CNN + Augmentation": cnn_augmentation,
    "GoogLeNet": google_classifier
    }
    best_model = models[best_method]

    test_accuracy, object_accuracies, matrix = evaluation(
        test_features, test_frames, test_label, test_objects,
        best_method, best_model, scaler, googlenet
    )

    print("Overall Accuracy:", test_accuracy)
    print("Object Accuracies:", object_accuracies)
    print("Confusion Matrix:\n", matrix)

if __name__ == "__main__":
    main()