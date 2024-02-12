"""Code from the official PaCMAP implementation (January 2024)"""

import os
import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import numba

import sys
sys.path.insert(0, './utils')
from run_script import data_prep

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from collections import Counter
from numpy.random import default_rng
from sklearn.decomposition import TruncatedSVD

from CONSTANTS import *


@numba.njit()
def euclid_dist(x1, x2):
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i] - x2[i]) ** 2
    return np.sqrt(result)

def score(X, Y, i,j,k):
    yij = euclid_dist(Y[i], Y[j])
    yik = euclid_dist(Y[i], Y[k])
    if yik < yij:
        return 1
    else:
        return 0

def score_largely(X, Y, i,j,k):
    xij = euclid_dist(X[i], X[j])
    xik = euclid_dist(X[i], X[k])
    yij = euclid_dist(Y[i], Y[j])
    yik = euclid_dist(Y[i], Y[k])
    if (xik-xij)/(xik+1e-15) < 0.2: # when the triplet is less important in high-dim space
        if (yij-yik)/(yik+1e-15) < 0.2: # no violation or slight violation
            return 0
        else:
            return 1
    else: # when the triplet is important in high-dim space
        if yij < yik:
            return 0
        else:
            return 1

def eval_random(X, Y, num=20):
    n, x_dim = X.shape
    if x_dim > 100:
        X -= np.mean(X, axis=0)
        X = TruncatedSVD(n_components=100, random_state=0).fit_transform(X)
    res = 0
    for i in range(n):
        for j in range(num):
            selected = np.random.randint(0, n, 2)
            if euclid_dist(X[i], X[selected[0]]) < euclid_dist(X[i], X[selected[1]]):
                res += score(X, Y, i, selected[0], selected[1])
            else:
                res += score(X, Y, i, selected[1], selected[0])
    return res

def knn_clf(nbr_vec, y):
    '''
    Helper function to generate knn classification result.
    '''
    y_vec = y[nbr_vec]
    c = Counter(y_vec)
    return c.most_common(1)[0][0]
    
def knn_eval(X, y, n_neighbors=1):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by an k-nearest neighbor classifier.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        kwargs: Any keyword argument that is send into the knn clf.
    Output:
        acc: The avg accuracy generated by the clf, using leave one out cross val.
    '''
    sum_acc = 0
    max_acc = X.shape[0]
    # Train once, reuse multiple times
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices [:, 1:]
    distances = distances[:, 1:]
    for i in range(X.shape[0]):
        result = knn_clf(indices[i], y)
        if result == y[i]:
            sum_acc += 1
    avg_acc = sum_acc / max_acc
    return avg_acc

def knn_eval_series(X, y, n_neighbors_list=[1, 3, 5, 10, 15, 20, 25, 30]):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by an k-nearest neighbor classifier.
    A series of accuracy will be calculated for the given n_neighbors.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        n_neighbors_list: A list of int.
        kwargs: Any keyword argument that is send into the knn clf.
    Output:
        accs: The avg accuracy generated by the clf, using leave one out cross val.
    '''
    avg_accs = []
    for n_neighbors in n_neighbors_list:
        avg_acc = knn_eval(X, y, n_neighbors)
        avg_accs.append(avg_acc)
    return avg_accs

def faster_knn_eval_series(X, y, n_neighbors_list=[1, 3, 5, 10, 15, 20, 25, 30]):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by an k-nearest neighbor classifier.
    A series of accuracy will be calculated for the given n_neighbors.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        n_neighbors_list: A list of int.
        kwargs: Any keyword argument that is send into the knn clf.
    Output:
        accs: The avg accuracy generated by the clf, using leave one out cross val.
    '''
    avg_accs = []
    max_acc = X.shape[0]
    # Train once, reuse multiple times
    nbrs = NearestNeighbors(n_neighbors=n_neighbors_list[-1]+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices [:, 1:]
    distances = distances[:, 1:]
    for n_neighbors in n_neighbors_list:
        sum_acc = 0
        for i in range(X.shape[0]):
            indices_temp = indices[:, :n_neighbors]
            result = knn_clf(indices_temp[i], y)
            if result == y[i]:
                sum_acc += 1
        avg_acc = sum_acc / max_acc
        avg_accs.append(avg_acc)
    return avg_accs

def svm_eval(X, y, img_verbose=False, n_splits=5, **kwargs):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by an SVM with rbf kernel.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        kwargs: Any keyword argument that is send into the SVM.
    Output:
        acc: The (avg) accuracy generated by an SVM with rbf kernel.
    '''
    X = scale(X)
    skf = StratifiedKFold(n_splits=n_splits)
    sum_acc = 0
    max_acc = n_splits
    for train_index, test_index in skf.split(X, y):
        clf = SVC(**kwargs)
        clf.fit(X[train_index], y[train_index])
        acc = clf.score(X[test_index], y[test_index])
        sum_acc += acc
    avg_acc = sum_acc/max_acc
    return avg_acc

def faster_svm_eval(X, y, n_splits=5, **kwargs):
    '''
    This is an accelerated version of the svm_eval function.
    An accuracy is calculated by an SVM with rbf kernel.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        kwargs: Any keyword argument that is send into the SVM.
    Output:
        acc: The (avg) accuracy generated by an SVM with rbf kernel.
    '''

    X = X.astype(np.float)
    X = scale(X)
    skf = StratifiedKFold(n_splits=n_splits)
    sum_acc = 0
    max_acc = n_splits
    for train_index, test_index in skf.split(X, y):
        feature_map_nystroem = Nystroem(gamma=1/(X.var()*X.shape[1]), random_state=1, n_components=300)
        data_transformed = feature_map_nystroem.fit_transform(X[train_index])
        clf = LinearSVC(random_state=0, tol=1e-5, **kwargs)
        clf.fit(data_transformed, y[train_index])
        test_transformed = feature_map_nystroem.transform(X[test_index])
        acc = clf.score(test_transformed, y[test_index])
        sum_acc += acc
    avg_acc = sum_acc/max_acc
    return avg_acc


def centroid_triplet_eval(X, X_new, y):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An triplet satisfaction score is calculated by evaluating how many triplets
    of cluster centroids have been violated.
    Input:
        X: A numpy array with the shape [N, p]. The higher dimension embedding
           of some dataset. Expected to have some clusters.
        X_new: A numpy array with the shape [N, k]. The lower dimension embedding
               of some dataset. Expected to have some clusters as well.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset. Used to identify clusters
    Output:
        acc: The score generated by the algorithm.
    '''    
    cluster_mean_ori, cluster_mean_new = [], []
    categories = np.unique(y)
    num_cat = len(categories)
    mask = np.mask_indices(num_cat, np.tril, -1)
    for i in range(num_cat):
        label = categories[i]
        X_clus_ori = X[y == label]
        X_clus_new = X_new[y == label]
        cluster_mean_ori.append(np.mean(X_clus_ori, axis = 0))
        cluster_mean_new.append(np.mean(X_clus_new, axis = 0))
    cluster_mean_ori = np.array(cluster_mean_ori)
    cluster_mean_new = np.array(cluster_mean_new)
    ori_dist = euclidean_distances(cluster_mean_ori)[mask]
    new_dist = euclidean_distances(cluster_mean_new)[mask]
    dist_agree = 0. # two distance agrees
    dist_all = 0. # count
    for i in range(len(ori_dist)):
        for j in range(i+1, len(ori_dist)):
            if ori_dist[i] > ori_dist[j] and new_dist[i] > new_dist[j]:
                dist_agree += 1
            elif ori_dist[i] <= ori_dist[j] and new_dist[i] <= new_dist[j]:
                dist_agree += 1
            dist_all += 1
    return dist_agree/dist_all

def faster_centroid_triplet_eval(X, X_new, y):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An triplet satisfaction score is calculated by evaluating how many triplets
    of cluster median centroids have been violated.
    Input:
        X: A numpy array with the shape [N, p]. The higher dimension embedding
           of some dataset. Expected to have some clusters.
        X_new: A numpy array with the shape [N, k]. The lower dimension embedding
               of some dataset. Expected to have some clusters as well.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset. Used to identify clusters
    Output:
        acc: The score generated by the algorithm.
    '''    
    cluster_mean_ori, cluster_mean_new = [], []
    categories = np.unique(y)
    num_cat = len(categories)
    mask = np.mask_indices(num_cat, np.tril, -1)
    for i in range(num_cat):
        label = categories[i]
        X_clus_ori = X[y == label]
        X_clus_new = X_new[y == label]
        cluster_mean_ori.append(np.median(X_clus_ori, axis = 0))
        cluster_mean_new.append(np.median(X_clus_new, axis = 0))
    cluster_mean_ori = np.array(cluster_mean_ori)
    cluster_mean_new = np.array(cluster_mean_new)
    ori_dist = euclidean_distances(cluster_mean_ori)[mask]
    new_dist = euclidean_distances(cluster_mean_new)[mask]
    dist_agree = 0. # two distance agrees
    dist_all = 0. # count
    for i in range(len(ori_dist)):
        for j in range(i+1, len(ori_dist)):
            if ori_dist[i] > ori_dist[j] and new_dist[i] > new_dist[j]:
                dist_agree += 1
            elif ori_dist[i] <= ori_dist[j] and new_dist[i] <= new_dist[j]:
                dist_agree += 1
            dist_all += 1
    return dist_agree/dist_all

def random_triplet_eval(X, X_new, y):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An triplet satisfaction score is calculated by evaluating how many randomly
    selected triplets have been violated. Each point will generate 5 triplets.
    Input:
        X: A numpy array with the shape [N, p]. The higher dimension embedding
           of some dataset. Expected to have some clusters.
        X_new: A numpy array with the shape [N, k]. The lower dimension embedding
               of some dataset. Expected to have some clusters as well.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset. Used to identify clusters
    Output:
        acc: The score generated by the algorithm.
    '''    

    # Sampling Triplets
    # Five triplet per point
    anchors = np.arange(X.shape[0])
    rng = default_rng(RANDOM_SEED)
    triplets = rng.choice(anchors, (X.shape[0], 5, 2))
    triplet_labels = np.zeros((X.shape[0], 5))
    anchors = anchors.reshape((-1, 1, 1))
    
    # Calculate the distances and generate labels
    b = np.broadcast(anchors, triplets)
    distances = np.empty(b.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u,v) in b]
    labels = distances[:, :, 0] < distances[: , :, 1]

    
    # Calculate distances for LD
    b = np.broadcast(anchors, triplets)
    distances_l = np.empty(b.shape)
    distances_l.flat = [np.linalg.norm(X_new[u] - X_new[v]) for (u,v) in b]
    pred_vals = distances_l[:, :, 0] < distances_l[:, :, 1]
    correct = np.sum(pred_vals == labels)
    acc = correct/X.shape[0]/5
    return acc

def evaluate_output(X, X_new, y, name, baseline=False, labelled=True):
    results = {}
    results['name'] = name
    if labelled:
        if baseline:
            baseline_knn_accs = knn_eval_series(X, y)
            baseline_svm_acc = faster_svm_eval(X, y)
            results['baseline_knn'] = baseline_knn_accs
            results['baseline_svm'] = baseline_svm_acc
        knn_accs = knn_eval_series(X_new, y)
        svm_acc = faster_svm_eval(X_new, y)
        cte_acc = centroid_triplet_eval(X, X_new, y)
        results['knn'] = knn_accs
        results['svm'] = svm_acc
        results['cte'] = cte_acc
    rte_acc = random_triplet_eval(X, X_new, y)
    results['rte'] = rte_acc
    return results

def evaluate_output_non_svm(X, X_new, y, name, baseline=False, labelled=True):
    results = {}
    results['name'] = name
    if labelled:
        if baseline:
            baseline_knn_accs = knn_eval_series(X, y)
            results['baseline_knn'] = baseline_knn_accs
        knn_accs = knn_eval_series(X_new, y)
        cte_acc = centroid_triplet_eval(X, X_new, y)
        results['knn'] = knn_accs
        results['cte'] = cte_acc
    rte_acc = random_triplet_eval(X, X_new, y)
    results['rte'] = rte_acc
    return results

def evaluate_output_cte_only(X, X_new, y, name, baseline=False, labelled=True):
    results = {}
    results['name'] = name
    if labelled:
        knn_accs = knn_eval_series(X_new, y)
        cte_acc = centroid_triplet_eval(X, X_new, y)
        results['knn'] = knn_accs
        results['cte'] = cte_acc
    rte_acc = random_triplet_eval(X, X_new, y)
    results['rte'] = rte_acc
    return results

def evaluate_output_svm_only(X, X_new, y, name, baseline=False, labelled=True):
    results = {}
    results['name'] = name
    if labelled:
        if baseline:
            baseline_svm_acc = faster_svm_eval(X, y)
            results['baseline_svm'] = baseline_svm_acc
        svm_acc = faster_svm_eval(X_new, y)
        results['svm'] = svm_acc
    return results

def fetch_output(dataset_name='MNIST'):
    location = '../output'
    all_file = os.listdir(location)
    selected_file = []
    for file in all_file:
        if file[:len(dataset_name)] == dataset_name and file[len(dataset_name)+1] != 'h' and file[len(dataset_name)+1] != 'b':
            selected_file.append(file)
    return selected_file

def evaluate_category(dataset_name='MNIST', labelled=True, data_pca=True, svm=True, svm_only=False):
    if data_pca:
        print('data_pca')
    if svm:
        print('svm')
    if svm_only:
        print('svm_only')
    X, y = data_prep(dataset_name, 70000)
    if X.shape[1] > 100:
        if data_pca and dataset_name != 'Mouse_scRNA':
            pca = PCA(n_components=100)
            X = pca.fit_transform(X)
        elif data_pca and dataset_name == 'Mouse_scRNA':
            pca = PCA(n_components=1000)
            X = pca.fit_transform(X)
    location = '../output'
    selected_file = fetch_output(dataset_name)
    i = 0
    all_results = {}
    for file in selected_file:
        X_new = np.load(location + file)
        for j in range(5):
            if i == 0 and j == 0:
                if svm:
                    results = evaluate_output(X, X_new[j], y, file, baseline=True, labelled=labelled)
                elif svm_only:
                    results = evaluate_output_svm_only(X, X_new[j], y, file, baseline=True, labelled=labelled)
                else:
                    results = evaluate_output_non_svm(X, X_new[j], y, file, baseline=True, labelled=labelled)
                all_results[results['name'] + str(j)] = results
                if labelled:
                    if not svm_only:
                        all_results['baseline_knn'] = results['baseline_knn']
                    if svm or svm_only:
                        all_results['baseline_svm'] = results['baseline_svm']
            else:
                if svm:
                    results = evaluate_output(X, X_new[j], y, file, baseline=False, labelled=labelled)
                elif svm_only:
                    results = evaluate_output_svm_only(X, X_new[j], y, file, baseline=False, labelled=labelled)
                else:
                    results = evaluate_output_non_svm(X, X_new[j], y, file, baseline=False, labelled=labelled)
                all_results[results['name'] + str(j)] = results
        i += 1
    if data_pca:
        dataset_name += '_pca'
    if labelled:
        dataset_name += '_l'
    if svm_only:
        dataset_name += '_svm'
    
    with open(dataset_name, 'wb') as fp:
        pickle.dump(all_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Finished')
    return all_results

def fetch_LargeVis(dataset_name='MNIST'):
    location = '../output'
    all_file = os.listdir(location)
    selected_file = []
    for file in all_file:
        # To solve the error of LargeVis
        if file[len(dataset_name)+1] != 'L':
            continue
        if file[:len(dataset_name)] == dataset_name and file[len(dataset_name)+1] != 'h':
            selected_file.append(file)
    return selected_file

def evaluate_LargeVis(dataset_name='MNIST', labelled=True, data_pca=True, svm=True, svm_only=False):
    X, y = data_prep(dataset_name, 70000)
    if X.shape[1] > 100:
        if data_pca and dataset_name != 'Mouse_scRNA':
            pca = PCA(n_components=100)
            X = pca.fit_transform(X)
        elif data_pca and dataset_name == 'Mouse_scRNA':
            pca = PCA(n_components=1000)
            X = pca.fit_transform(X)
    location = '../output'
    selected_file = fetch_LargeVis(dataset_name)
    i = 0
    all_results = {}
    for file in selected_file:
        X_new = np.load(location + file)
        for j in range(5):
            if i == 0 and j == 0:
                if svm:
                    results = evaluate_output(X, X_new[j], y, file, baseline=True, labelled=labelled)
                elif svm_only:
                    results = evaluate_output_svm_only(X, X_new[j], y, file, baseline=True, labelled=labelled)
                else:
                    results = evaluate_output_non_svm(X, X_new[j], y, file, baseline=True, labelled=labelled)
                all_results[results['name'] + str(j)] = results
                if labelled:
                    if not svm_only:
                        all_results['baseline_knn'] = results['baseline_knn']
                    if svm or svm_only:
                        all_results['baseline_svm'] = results['baseline_svm']
            else:
                if svm:
                    results = evaluate_output(X, X_new[j], y, file, baseline=False, labelled=labelled)
                elif svm_only:
                    results = evaluate_output_svm_only(X, X_new[j], y, file, baseline=False, labelled=labelled)
                else:
                    results = evaluate_output_non_svm(X, X_new[j], y, file, baseline=False, labelled=labelled)
                all_results[results['name'] + str(j)] = results
        i += 1
    dataset_name += '_largevis'
    if data_pca:
        dataset_name += '_pca'
    if labelled:
        dataset_name += '_l'
    if svm_only:
        dataset_name += '_svm'
    elif svm == False:
        dataset_name += '_nonsvm'
    with open(dataset_name, 'wb') as fp:
        pickle.dump(all_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Finished')
    return all_results

def evaluate_npy(selected_file, dataset_name='MNIST', labelled=True, data_pca=True, svm=True):
    size_arg = 10000000
    if dataset_name == 's_curve' or dataset_name == 's_curve_hole':
        size_arg = 10000
    X, y = data_prep(dataset_name, size_arg)
    if X.shape[1] > 100:
        if data_pca and dataset_name != 'Mouse_scRNA':
            pca = PCA(n_components=100)
            X = pca.fit_transform(X)
        elif data_pca and dataset_name == 'Mouse_scRNA':
            pca = PCA(n_components=1000)
            X = pca.fit_transform(X)
    location = '../output'
    output_location = '../test_results/'
    for file in selected_file:
        all_results = {}
        X_new = np.load(location + file)
        for j in range(5):
            if svm:
                results = evaluate_output_svm_only(X, X_new[j], y, file, baseline=False, labelled=labelled)
            else:
                results = evaluate_output_non_svm(X, X_new[j], y, file, baseline=False, labelled=labelled)
            all_results[str(j)] = results
        outfilename = file[:-4]
        if svm:
            outfilename += '_svm'
        outfilename = output_location + outfilename + '.json'
        with open(outfilename, 'wb') as fp:
            pickle.dump(all_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print('Succesfully evaluated ' + file)
    print('Finished evaluation')

def evaluate_ctes(selected_file, dataset_name='MNIST', labelled=True, data_pca=True):
    size_arg = 10000000
    if dataset_name == 's_curve' or dataset_name == 's_curve_hole':
        size_arg = 10000
    X, y = data_prep(dataset_name, size_arg)
    if X.shape[1] > 100:
        if data_pca and dataset_name != 'Mouse_scRNA':
            pca = PCA(n_components=100)
            X = pca.fit_transform(X)
        elif data_pca and dataset_name == 'Mouse_scRNA':
            pca = PCA(n_components=1000)
            X = pca.fit_transform(X)
    location = '../output'
    output_location = '../test_results/'
    for file in selected_file:
        all_results = {}
        X_new = np.load(location + file)
        for j in range(5):
            results = centroid_triplet_eval(X, X_new[j], y)
            all_results[str(j)] = results
        outfilename = file[:-4]
        outfilename += '_cte'
        outfilename = output_location + outfilename + '.json'
        with open(outfilename, 'wb') as fp:
            pickle.dump(all_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print('Succesfully evaluated ' + file)
    print('Finished evaluation')
    
def evaluate_rtes(selected_file, dataset_name='MNIST', labelled=True, data_pca=True):
    size_arg = 10000000
    if dataset_name == 's_curve' or dataset_name == 's_curve_hole':
        size_arg = 10000
    X, y = data_prep(dataset_name, size_arg)
    if X.shape[1] > 100:
        if data_pca and dataset_name != 'Mouse_scRNA':
            pca = PCA(n_components=100)
            X = pca.fit_transform(X)
        elif data_pca and dataset_name == 'Mouse_scRNA':
            pca = PCA(n_components=1000)
            X = pca.fit_transform(X)
    location = '../output'
    output_location = '../test_results/'
    for file in selected_file:
        all_results = {}
        X_new = np.load(location + file)
        for j in range(5):
            results = random_triplet_eval(X, X_new[j], y)
            all_results[str(j)] = results
        outfilename = file[:-4]
        outfilename += '_rte'
        outfilename = output_location + outfilename + '.json'
        with open(outfilename, 'wb') as fp:
            pickle.dump(all_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print('Succesfully evaluated ' + file)
    print('Finished evaluation')
