import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Load the image
#image = cv2.imread("pC:\Users\9254g\Downloads\Real-Time-Face-Recognition-master\Real-Time-Face-Recognition-master\plant_leaf.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("C:/Users/9254g/OneDrive/Desktop/CV_Activity/leaf_image.jpg", cv2.IMREAD_GRAYSCALE)

# Resize for faster processing if needed
image = cv2.resize(image, (200, 200))

# Step 1: Extract GLCM Texture Features
# Define GLCM properties and distances
glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
contrast = graycoprops(glcm, 'contrast')[0, 0]
dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]
correlation = graycoprops(glcm, 'correlation')[0, 0]

# Form feature vector from texture properties
texture_features = np.array([contrast, dissimilarity, homogeneity, energy, correlation])

# Step 2: Texture-based Image Segmentation using K-means
# Reshape the image for clustering
pixels = image.reshape(-1, 1)

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
segmented_image = kmeans.labels_.reshape(image.shape)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(segmented_image, cmap='gray')
plt.show()

features = []  # Extracted texture features
labels = []    # 0 for healthy, 1 for pest-infested

# Add texture features and corresponding labels
features.append(texture_features)  # For a single image, repeat for multiple images
labels.append(0)  # Example label (0 for healthy)

# Train a classifier (SVM in this case)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")