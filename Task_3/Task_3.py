# Code Map

# Step 0: Setup
from pathlib import Path
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

# Step 1: Import and preprocess the input videos
def preprocessing(video_path):
    video = cv2.VideoCapture(video_path)
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
    first_edges = np.mean(edge_frames[:,3], axis = 0)
    last_edges = np.mean(edge_frames[-3,:], axis = 0)
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
    features = np.concatenate(edge_mean.flatten(),
                             edge_change.flatten(),
                             movement_mean.flatten())
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
    if method == "Augmentation":
        model.add(tf.keras.layers.RandomBrightness(
            0.1, value_range = (0.0, 1.0)
        ))

        model.add(tf.keras.layers.RandomContrast(0.1))

    # First Convolutional Layer
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding = "same"))

    if method == "Batch":
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Second Convolutional Layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same"))

    if method == "Batch":
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
def model_training():
    return 0

def evaluation():
    return 0

def main():
    location = Path(__file__).resolve().parent
    training_features = []
    test_features = []

    for folder_name in ["Training", "Test"]:
        folder = location / folder_name
        
        for video_path in sorted(folder.iterdir()):
            if video_path.suffix.lower() not in (".mp4", ".mov", ".avi"):
                continue

            # Step 1: Preprocessing
            frames = preprocessing(video_path)
            print(folder_name, video_path.name, frames.shape)

            # Step 2: Visual feature extraction
            features = feature_extraction(frames)
            
            if folder_name == "Training":
                training_features.append(features)
            else:
                test_features.append(features)

            # Step 3: Classification

            


if __name__ == "__main__":
    main()