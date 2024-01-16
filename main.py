import sys
import numpy as np
from PARAMETERS import*

# Insert module paths
module_paths = ['algorithms', 'evaluation', 'visualization']  # './path' if main is in a folder
sys.path[0:0] = module_paths

# Import packages
from sklearn import datasets
from umap_reducer import UMAP_reducer
from pacmap_reducer import PaCMAP_reducer
import evaluation
import visualization
from PARAMETERS import RANDOM_SEED


def get_dataset_by_name(name='Digits'):
    """Loads the dataset of choice.

    Args:
        name (str, optional): Name of the dataset. Defaults to 'Digits'.

    Returns:
        (array, array): X (data/features) and y (target/labels)
    """
    if name == 'Digits':
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target

    elif name == 'Iris':  # Low-dimensional (4D)
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

    elif name == 'CIFAR-10':
        cifar10 = datasets.fetch_openml(name='CIFAR_10', version=1)
        X = cifar10['data'], 
        y = cifar10['target'].astype(int)
        
    elif name == 'CIFAR-100':
        cifar100 = datasets.fetch_openml(name='CIFAR_100', version=1)
        X = cifar100['data']
        X.reshape(X.shape[0], -1)
        y = cifar100['target'].astype(int)
        
    elif name == 'LFW':
        lfw = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        X, y = lfw.data, lfw.target
        X = X.reshape(X.shape[0], -1)

    elif name == 'COIL-20':
        # loading preprocessed coil_20 dataset
        # you can change it with any dataset that is in the ndarray format, with the shape (N, D)
        # where N is the number of samples and D is the dimension of each sample
        X = np.load('./data/coil_20.npy', allow_pickle=True)
        X = X.reshape(X.shape[0], -1)
        y = np.load('./data/coil_20_labels.npy', allow_pickle=True)
    
    return X, y


def get_best_result_and_corresponding_parameters(
    Xs,
    results, 
    params,
    knn_neighbors=[1, 3, 5, 10, 15, 20, 25, 30]):
    print('\nCalculating best results')
    best_Xs = {}
    best_result = {}
    best_result['knn'] = float('-inf')
    best_result['svm'] = float('-inf')
    best_result['cte'] = float('-inf')
    best_result['rte'] = float('-inf')
    
    best_param = {}
    best_knn_neighbors = None
    
    for i, result in enumerate(results):
        knns = result['knn']
        svm = result['svm']
        cte = result['cte']
        rte = result['rte']
        
        for j, knn in enumerate(knns):
            if knn > best_result['knn']:
                best_Xs['knn'] = Xs[i]
                best_result['knn'] = knn
                best_param['knn'] = params[i]
                best_knn_neighbors = knn_neighbors[j]
        
        if svm > best_result['svm']:
            best_Xs['svm'] = Xs[i]
            best_result['svm'] = svm
            best_param['svm'] = params[i]
            
        if cte > best_result['cte']:
            best_Xs['cte'] = Xs[i]
            best_result['cte'] = cte
            best_param['cte'] = params[i]
        
        if rte > best_result['rte']:
            best_Xs['rte'] = Xs[i]
            best_result['rte'] = rte
            best_param['rte'] = params[i]
    
    print('Best results calculated\n')
    return best_Xs, best_result, best_param, best_knn_neighbors


def hyperparameter_tuning(X, 
                          y, 
                          algorithm='UMAP', 
                          ns_neighbors=[5, 10, 20, 50, 100]):
    print(f'\nHyperparameter tuning for {algorithm}:\n')
    Xs_new = []
    results = []

    if algorithm == 'UMAP':
        for n_neighbors in ns_neighbors:
            print(f'Calculating results for {n_neighbors} neighbors')
            umap = UMAP_reducer(n_neighbors=n_neighbors)
            umap.fit(X) 
            X_new_umap = umap.get_embedding()
            result_umap = evaluation.evaluate_output(X, X_new_umap, y, 'UMAP')
            Xs_new.append(X_new_umap)
            results.append(result_umap)

    elif algorithm == 'PaCMAP':
        for n_neighbors in ns_neighbors:
            print(f'Calculating results for {n_neighbors} neighbors')
            pacmap = PaCMAP_reducer(n_neighbors=n_neighbors)
            pacmap.fit(X) 
            X_new_pacmap = pacmap.get_embedding()
            result_pacmap = evaluation.evaluate_output(X, X_new_pacmap, y, 'PaCMAP')
            Xs_new.append(X_new_pacmap)
            results.append(result_pacmap)
    else:
        print('Not a valid algorithm for hyperparameter tuning (can only be "UMAP" or "PaCMAP")')

    best_Xs, best_result, best_param, best_knn_neighbors = get_best_result_and_corresponding_parameters(
        Xs_new, 
        results, 
        ns_neighbors)
    
    best_result['name'] = algorithm
    print(f'Done with hyperparameter tuning for {algorithm}\n')
    
    return best_Xs, best_result, best_param, best_knn_neighbors


def visualize_and_print_results(Xs_new_umap,
                                Xs_new_pacmap,
                                knn_neighbors_umap,
                                knn_neighbors_pacmap,
                                results_umap,
                                results_pacmap,
                                params_umap,
                                params_pacmap,
                                y,
                                dataset_name):
    # Print results of both algorithms
    evaluation.print_evaluation_results(results_umap)
    evaluation.print_evaluation_results(results_pacmap)

    print('knn_neighbors_umap:', knn_neighbors_umap)
    print('knn_neighbors_pacmap:', knn_neighbors_pacmap)
    
    print('params_umap', params_umap)
    print('params_pacmap', params_pacmap)
    
    metrics = ['knn', 'svm', 'cte', 'rte']
    title_metrics = ['KNN', 'SVM', 'centroid triplet', 'random triplet']
    
    # Plot all visualizations
    for metric, title_metric in zip(metrics, title_metrics):
        visualization.plot_embeddings(embeddings=[{'UMAP': Xs_new_umap[metric]},
                                                  {'PaCMAP': Xs_new_pacmap[metric]}],
                                    y=y, 
                                    data_title=dataset_name,
                                    fig_name=f'fig_{dataset_name}_{metric}',
                                    n_neighbors=[params_umap[metric],
                                                params_pacmap[metric]],
                                    #n_knn_neighbors=[knn_neighbors_umap, knn_neighbors_pacmap],
                                    metric=title_metric)


def main(dataset_name, data_path='data', figure_path='figures'):
    print('\nInitializing UMAP and PaCMAP calculations\n')
    np.random.seed(RANDOM_SEED)  # Set random seed
    X, y = get_dataset_by_name(dataset_name)  # Load data
    
    # UMAP
    Xs_new_umap, results_umap, params_umap, knn_neighbors_umap = hyperparameter_tuning(X, y, algorithm='UMAP')
    X_new_umap = Xs_new_umap['knn']
    
    # PaCMAP
    Xs_new_pacmap, results_pacmap, params_pacmap, knn_neighbors_pacmap = hyperparameter_tuning(X, y, algorithm='PaCMAP')
    
    visualize_and_print_results(Xs_new_umap,
                                Xs_new_pacmap,
                                knn_neighbors_umap,
                                knn_neighbors_pacmap,
                                results_umap,
                                results_pacmap,
                                params_umap,
                                params_pacmap,
                                y,
                                dataset_name)
    
    
if __name__ == '__main__':
    main('Digits')  # Choose the dataset you want to use
    
    #data_path = "./data/"
    #output_path = "./output/"
    #experiments.main(data_path, output_path,'MNIST', 10000000)
    """main(data_path, output_path,'FMNIST', 10000000)
    main(data_path, output_path,'coil_20', 10000000)
    main(data_path, output_path,'coil_100', 10000000)

    main(data_path, output_path,'Mouse_scRNA', 10000000)
    main(data_path, output_path,'mammoth', 10000000)
    main(data_path, output_path,'s_curve_hole', 10000)
    main(data_path, output_path,'20NG', 20000)
    main(data_path, output_path,'USPS', 20000)


    main(data_path, output_path,'Flow_cytometry', 10000000)
    main(data_path, output_path,'kddcup99', 10000000)"""    
