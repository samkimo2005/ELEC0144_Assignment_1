# Code Map
# Step 0: Setup                Import required libraries.
# Step 1: Preprocessing        Extract and prepare video frames.
# Step 2: Feature Extraction   Extract edge and movement features.
# Step 3: Classification       Construct five classification methods.
# Shared Functions            Prepare inputs and predictions for CNNs and GoogLeNet.
# Step 4: Training             Train five models, validate and select the best method.
# Step 5: Evaluation           Calculate test accuracy and confusion matrix.
# Main                         Load datasets and execute the classification pipeline.
# AI use                       GPT-6

# Step 0: Setup
from pathlib import Path
from unicodedata import name
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import googlenet as torchvision_googlenet, GoogLeNet_Weights
import csv

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
    # PyTorch uses (channels, height, width) rather than (height, width, channels).
    layers = [nn.Conv2d(3, 16, (3, 3), padding=1)]

    if method == "batch":
        layers.append(nn.BatchNorm2d(16))

    layers.extend([nn.ReLU(), nn.MaxPool2d((2, 2))])

    layers.append(nn.Conv2d(16, 32, (3, 3), padding=1))

    if method == "batch":
        layers.append(nn.BatchNorm2d(32))

    layers.extend([
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    ])

    # For "augmentation", brightness and contrast are changed during training only.
    return nn.Sequential(*layers)

# Method 5: Classification using GoogLeNet (pre-trained CNN)
def classification_googlenet():
    # Load pre-trained GoogLeNet and use its 1024-dimensional visual features.
    googlenet = torchvision_googlenet(
        weights=GoogLeNet_Weights.DEFAULT,
        transform_input=False
    )
    googlenet.fc = nn.Identity()
    googlenet.eval()
    for parameter in googlenet.parameters():
        parameter.requires_grad = False

    # Define the binary classifier
    classifier = LogisticRegression(
        max_iter = 1000,
        random_state  = 42
    )
    
    return googlenet, classifier

# Shared operations for Training, Validation and Test

def select_cnn_frames(frame_lists):
    return np.concatenate(
        [frames[[2, 5, 9]] for frames in frame_lists], axis=0
    ).astype(np.float32)


def cnn_grip_predictions(model, images, number_of_grips):
    model.eval()
    device = next(model.parameters()).device
    probabilities = []

    with torch.no_grad():
        for i in range(0, len(images), 16):
            batch = torch.from_numpy(images[i:i + 16]).permute(0, 3, 1, 2).to(device)
            output = model(batch).squeeze(1)
            probabilities.extend(output.cpu().numpy())

    probabilities = np.array(probabilities).reshape(number_of_grips, 3)
    grip_probabilities = np.mean(probabilities, axis=1)
    return (grip_probabilities >= 0.5).astype(int)


def google_grip_features(googlenet, images, number_of_grips):
    googlenet.eval()
    device = next(googlenet.parameters()).device
    features = []

    # ImageNet normalisation required by the pretrained GoogLeNet weights.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for i in range(0, len(images), 16):
            batch = torch.from_numpy(images[i:i + 16]).permute(0, 3, 1, 2).to(device)
            batch = (batch - mean) / std
            output = googlenet(batch)
            features.append(output.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features.reshape(number_of_grips, 3, -1).mean(axis=1)

# Step 4: Training

def model_training(
        training_features, validation_features, 
        training_label, validation_label,
        training_frames, validation_frames,
        mlp, 
        cnn_basic, cnn_batchnorm, cnn_augmentation, 
        googlenet, google_classifier):

    results = {}

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
    X_train_cnn = select_cnn_frames(training_frames)
    X_val_cnn = select_cnn_frames(validation_frames)

    # Each grip's label is repeated for its three frames
    y_train_cnn = np.repeat(y_train, 3)

    # --------------------------------------------------
    # Methods 2, 3 and 4: CNNs
    # --------------------------------------------------

    cnn_methods = {
        "Basic CNN": cnn_basic,
        "CNN + BatchNorm": cnn_batchnorm,
        "CNN + Augmentation": cnn_augmentation
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images = torch.from_numpy(X_train_cnn).permute(0, 3, 1, 2)
    train_labels = torch.tensor(y_train_cnn, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels), batch_size=16, shuffle=True
    )
    augmentation = transforms.ColorJitter(brightness=0.1, contrast=0.1)

    for name, model in cnn_methods.items():

        print("Training:", name)
        model.to(device)
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the CNN for 10 epochs with batches of 16 frames
        for epoch in range(10):
            model.train()
            epoch_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                if name == "CNN + Augmentation":
                    images = augmentation(images)

                optimizer.zero_grad()
                outputs = model(images).squeeze(1)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(labels)

            print(f"Epoch {epoch + 1}/10 - loss: {epoch_loss / len(y_train_cnn):.4f}")

        # Predict one Light/Firm label for each validation grip
        predictions = cnn_grip_predictions(model, X_val_cnn, len(y_val))

        # Calculate grip-level validation accuracy
        accuracy = accuracy_score(y_val, predictions)

        results[name] = accuracy
        print(name, "Accuracy:", accuracy)

    # --------------------------------------------------
    # Method 5: Pre-trained GoogLeNet
    # --------------------------------------------------

    print("Training: GoogLeNet")

    # Extract and average pre-trained features for each grip
    X_train_google = google_grip_features(
        googlenet, X_train_cnn, len(y_train)
    )
    X_val_google = google_grip_features(
        googlenet, X_val_cnn, len(y_val)
    )

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

    return scaler, best_method, results
    

# Step 5: Final Evaluation
def evaluation(test_label, test_objects, predictions_by_method):

    test_results = {}
    labels = np.array(test_label)
    objects = np.array(test_objects)

    for method, predictions in predictions_by_method.items():

        # Overall accuracy
        overall_accuracy = accuracy_score(labels, predictions)

        # Accuracy for each object
        object_accuracies = {}

        for object_name in ["sponge", "bottle"]:
            mask = objects == object_name

            object_accuracies[object_name] = accuracy_score(
                labels[mask], predictions[mask]
            )

        # Confusion matrix
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])

        test_results[method] = (
            overall_accuracy, object_accuracies, matrix
        )

        print("\nTest Results:", method)
        print("Overall Accuracy:", f"{overall_accuracy:.2%}")

        for name, accuracy in object_accuracies.items():
            print(name, "Accuracy:", f"{accuracy:.2%}")

        print("Confusion Matrix:\n", matrix)

    return test_results


def main():
    location = Path(__file__).resolve().parent

    with open(location / "input_data" / "manifest.csv", "r", newline="", encoding="utf-8-sig") as file:
        manifest = {
            row["clip_id"].strip().lower(): row
            for row in csv.DictReader(file)
        }

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

            # Get each video's label and object from the manifest
            clip_id = video_path.stem.lower()
            record = manifest[clip_id]
            grip = record["grip"].strip().lower()
            if grip not in ("light", "firm"):
                raise ValueError("Unknown grip label: " + clip_id)
            label = 0 if grip == "light" else 1
            object_type = record["object"].strip().lower()

            # Step 1: Preprocessing
            frames = preprocessing(video_path)
            print(folder_name, video_path.name, frames.shape)

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
    scaler, best_method, validation_results = model_training(
        training_features, validation_features, 
        training_label, validation_label,
        training_frames, validation_frames,
        mlp, 
        cnn_basic, cnn_batchnorm, cnn_augmentation, 
        googlenet, google_classifier)

    # Step 5: Evaluation of all five trained methods
    X_test_mlp = scaler.transform(
        np.array(test_features, dtype=np.float32)
    )

    X_test_cnn = select_cnn_frames(test_frames)

    X_test_google = google_grip_features(
        googlenet, X_test_cnn, len(test_label)
    )

    predictions_by_method = {
        "MLP": mlp.predict(X_test_mlp),
        "Basic CNN": cnn_grip_predictions(cnn_basic, X_test_cnn, len(test_label)),
        "CNN + BatchNorm": cnn_grip_predictions(cnn_batchnorm, X_test_cnn, len(test_label)),
        "CNN + Augmentation": cnn_grip_predictions(cnn_augmentation, X_test_cnn, len(test_label)),
        "GoogLeNet": google_classifier.predict(X_test_google)
    }

    test_results = evaluation(
        test_label, test_objects, predictions_by_method
    )
    
    # Calculate and display Test metrics for every method
    test_results = evaluation(test_label, test_objects, predictions_by_method)

    # Step 6: Save Validation and Test results in an Excel-readable CSV spreadsheet
    with open(location / "Task_3_Results.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Method", "Metric", "Value"])

        for method, accuracy in validation_results.items():
            writer.writerow(["Validation", method, "Accuracy (%)", round(accuracy * 100, 2)])

        for method, (test_accuracy, object_accuracies, matrix) in test_results.items():
            writer.writerow(["Test", method, "Overall accuracy (%)", round(test_accuracy * 100, 2)])
            for object_name, accuracy in object_accuracies.items():
                writer.writerow(["Test", method, object_name + " accuracy (%)", round(accuracy * 100, 2)])

            for actual in range(2):
                for predicted in range(2):
                    actual_name = "Light" if actual == 0 else "Firm"
                    predicted_name = "Light" if predicted == 0 else "Firm"
                    writer.writerow([
                        "Test", method,
                        f"Actual {actual_name}, Predicted {predicted_name} (count)",
                        int(matrix[actual, predicted])
                    ])




if __name__ == "__main__":
    main()