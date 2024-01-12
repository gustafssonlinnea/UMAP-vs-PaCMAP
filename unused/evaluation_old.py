import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist
from tabulate import tabulate
from evaluation import calculate_svm_accuracy


def _calculate_knn_accuracy(X, y):
    loo = LeaveOneOut()
    
    knn_accuracies = []
    
    k = 3  # TODO: make k tunable

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        accuracy = _calculate_knn_accuracy_of_fold(X_train, y_train, X_test, y_test, k)
        knn_accuracies.append(accuracy)

    # Calculate the mean and standard deviation of the accuracies
    mean_accuracy = np.mean(knn_accuracies)
    std_accuracy = np.std(knn_accuracies)
    
    return mean_accuracy, std_accuracy


def _calculate_knn_accuracy_of_fold(X_train, y_train, X_test, y_test, k):
    # Assuming X_train, y_train are your training data and labels
    # Assuming X_test, y_test are your test data and labels

    # Create and train the KNN model
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_knn = knn_model.predict(X_test)

    # Calculate KNN accuracy
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    
    return knn_accuracy

def _calculate_svm_accuracy(X, y):  # X: embedding, y: labels
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm_accuracies = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        accuracy = _calculate_svm_accuracy_of_fold(X_train, y_train, X_test, y_test)
        svm_accuracies.append(accuracy)

    # Calculate the mean and standard deviation of the accuracies
    mean_accuracy = np.mean(svm_accuracies)
    std_accuracy = np.std(svm_accuracies)
    
    return mean_accuracy, std_accuracy


def _calculate_svm_accuracy_of_fold(X_train, y_train, X_test, y_test):
     # Use the Nystroem method to approximate the kernel matrix
    nystroem = Nystroem(random_state=42)
    X_train_transformed = nystroem.fit_transform(X_train)
    X_test_transformed = nystroem.transform(X_test)

    # Train nonlinear SVM model on the transformed features
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train_transformed, y_train)

    # Make predictions on the test set
    y_pred_svm = svm_model.predict(X_test_transformed)

    """
    # Create and train the SVM model with RBF kernel
    svm_model = SVC(kernel='rbf')  # You can adjust other parameters like C, gamma, etc.
    svm_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_svm = svm_model.predict(X_test)"""

    # Calculate SVM accuracy
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    
    return svm_accuracy


def _calculate_random_triplet_accuracy(embedding, num_samples=1000):
    """Calculates the random triplet accuracy of embedding by drawing num_samples samples.

    Args:
        embedding (np.ndarray): The reduced-dimensional representation.
        num_samples (int, optional): The number of random samples to draw. Defaults to 1000.

    Returns:
        float: The random triplet accuracy
    """
    num_points, num_dimensions = embedding.shape
    
    # Randomly sample triplets
    triplets = np.random.choice(num_points, (num_samples, 3))
    
    # Calculate pairwise distances in the high-dimensional space
    high_dim_distances = pdist(embedding, metric='euclidean')
    
    # Calculate pairwise distances in the low-dimensional space
    low_dim_distances = pdist(embedding[triplets], metric='euclidean')
    
    # Count the number of triplets where the relative distances are maintained
    consistent_triplets = np.sum(high_dim_distances[triplets[:, 0]] < high_dim_distances[triplets[:, 1]])
    consistent_triplets += np.sum(high_dim_distances[triplets[:, 1]] < high_dim_distances[triplets[:, 2]])
    consistent_triplets += np.sum(high_dim_distances[triplets[:, 0]] < high_dim_distances[triplets[:, 2]])
    
    # Calculate the Random Triplet Accuracy
    triplet_accuracy = consistent_triplets / (3 * num_samples)
    
    return triplet_accuracy


def _calculate_centroid_triplet_accuracy(embedding, labels, num_samples=1000):
    # embedding is the reduced-dimensional representation and labels is the array of true class labels (aka y_true)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Randomly sample classes for triplets
    triplet_classes = np.random.choice(unique_labels, (num_samples, 3))
    
    # Get indices corresponding to the sampled classes
    triplet_indices = [np.where(labels == triplet_classes[i, j])[0] for i in range(num_samples) for j in range(3)]
    
    # Calculate pairwise distances between centroids in the high-dimensional space
    high_dim_centroid_distances = pdist(np.array([np.mean(embedding[indices], axis=0) for indices in triplet_indices]), metric='euclidean')
    
    # Calculate pairwise distances between centroids in the low-dimensional space
    low_dim_centroid_distances = pdist(np.array([np.mean(embedding[indices], axis=0) for indices in triplet_indices]), metric='euclidean')
    
    # Count the number of preserved centroid triplets
    consistent_centroid_triplets = np.sum(high_dim_centroid_distances < low_dim_centroid_distances)
    
    # Calculate the Centroid Triplet Accuracy
    centroid_triplet_accuracy = consistent_centroid_triplets / num_samples
    
    return centroid_triplet_accuracy


def evaluate(embedding, labels):
    knn_accuracy = _calculate_knn_accuracy(embedding, labels)
    svm_accuracy, svm_accuracy_std = _calculate_svm_accuracy(embedding, labels)
    random_triplet_accuracy = _calculate_random_triplet_accuracy(embedding)
    centroid_triplet_accuracy = _calculate_centroid_triplet_accuracy(embedding, labels)
    
    results = [
        ['KNN Accuracy', knn_accuracy], 
        ['SVM Accuracy', svm_accuracy], 
        ['Random Triplet Accuracy', random_triplet_accuracy], 
        ['Centroid Triplet Accuracy', centroid_triplet_accuracy]
        ]
    
    print(tabulate(results, headers=['Metric', 'Score']))
