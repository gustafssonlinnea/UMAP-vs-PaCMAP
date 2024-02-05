# Insert module paths
import sys
module_paths = ['algorithms', 
                'data', 
                'evaluation',
                'results', 
                'utils',
                'visualization']  # './path' if main is in a folder
sys.path[0:0] = module_paths

# Import packages and modules
from umap_reducer import UMAP_reducer
from pacmap_reducer import PaCMAP_reducer
import evaluation
import visualization
import numpy as np
from load_data import get_dataset_by_name
import results as res
import utils
from CONSTANTS import *


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
    return res.Results(best_Xs, best_result, best_param, best_knn_neighbors)


def hyperparameter_tuning(X, 
                          y, 
                          algorithm='UMAP', 
                          ns_neighbors=[5, 10, 20, 50, 100]):
    print(f'\nHyperparameter tuning for {algorithm}:\n')
    Xs_new = []
    metric_results = []

    if algorithm == 'UMAP':
        for n_neighbors in ns_neighbors:
            print(f'Calculating results for {n_neighbors} neighbors')
            umap = UMAP_reducer(n_neighbors=n_neighbors)
            umap.fit(X) 
            X_new_umap = umap.get_embedding()
            result_umap = evaluation.evaluate_output(X, X_new_umap, y, 'UMAP')
            Xs_new.append(X_new_umap)
            metric_results.append(result_umap)

    elif algorithm == 'PaCMAP':
        for n_neighbors in ns_neighbors:
            print(f'Calculating results for {n_neighbors} neighbors')
            pacmap = PaCMAP_reducer(n_neighbors=n_neighbors)
            pacmap.fit(X) 
            X_new_pacmap = pacmap.get_embedding()
            result_pacmap = evaluation.evaluate_output(X, X_new_pacmap, y, 'PaCMAP')
            Xs_new.append(X_new_pacmap)
            metric_results.append(result_pacmap)
    else:
        print('Not a valid algorithm for hyperparameter tuning (can only be "UMAP" or "PaCMAP")')

    results = get_best_result_and_corresponding_parameters(
        Xs_new, 
        metric_results, 
        ns_neighbors)
    
    results.metric_results['name'] = algorithm
    print(f'Done with hyperparameter tuning for {algorithm}\n')
    
    return results


def visualize_and_print_results(results_umap,
                                results_pacmap,
                                y,
                                dataset_name,
                                cmap,
                                max_datapoints=10000):
    # Print results of both algorithms
    res.Results.print_results(results_umap, results_pacmap)
    
    metrics = ['knn', 'svm', 'cte', 'rte']
    title_metrics = ['KNN', 'SVM', 'centroid triplet', 'random triplet']
    
    # Create and save all visualizations
    for metric, title_metric in zip(metrics, title_metrics):
        embedding_umap = results_umap.Xs[metric]
        embedding_pacmap = results_pacmap.Xs[metric]
        
        if embedding_umap.shape[0] > max_datapoints and max_datapoints is not None:
            embedding_umap = embedding_umap[:max_datapoints, :max_datapoints]
            embedding_pacmap = embedding_pacmap[:max_datapoints, :max_datapoints]
            y = y[:max_datapoints]
        
        fig_path = f'figures/{dataset_name}'
        utils.create_directory_if_not_exists(fig_path)
        
        if cmap is None:
            if dataset_name != 'Circle2D' and embedding_umap.shape[0] <= 12:
                cmap = DISCRETE_CMAP
            elif dataset_name != 'Circle2D':
                cmap = CONTINUOUS_CMAP
            else:
                cmap = CYCLIC_CMAP
            
        visualization.plot_embeddings(embeddings=[{'UMAP': embedding_umap},
                                                  {'PaCMAP': embedding_pacmap}],
                                    y=y, 
                                    data_title=dataset_name,
                                    fig_name=f'fig_{dataset_name}_{metric}',
                                    fig_path=fig_path,
                                    n_neighbors=[results_umap.params[metric],
                                                results_pacmap.params[metric]],
                                    metric=title_metric,
                                    dot_size=int(1000 / np.sqrt(embedding_umap.shape[0])),
                                    cmap=cmap)


def main(dataset_name, 
         data_path='data', 
         figure_path='figures', 
         read_results=False, 
         cmap=None):

    X, y = get_dataset_by_name(dataset_name)  # Load data
    
    if not read_results: 
        print('\nInitializing UMAP and PaCMAP calculations ' \
        f'for {dataset_name} dataset\n')       

        # UMAP
        results_umap = hyperparameter_tuning(X, y, algorithm='UMAP')
        results_umap.write_results_to_file(f'res_UMAP_{dataset_name}')
        
        # PaCMAP
        results_pacmap = hyperparameter_tuning(X, y, algorithm='PaCMAP')
        results_pacmap.write_results_to_file(f'res_PaCMAP_{dataset_name}')
    else:
        print('Initializing reading of stored results ' \
            f'for {dataset_name} dataset\n')

        # UMAP
        results_umap = res.Results.read_results_from_file(
            file_name=f'res_UMAP_{dataset_name}')
        
        # PaCMAP
        results_pacmap = res.Results.read_results_from_file(
            file_name=f'res_PaCMAP_{dataset_name}')
    
    visualize_and_print_results(results_umap, 
                                results_pacmap, 
                                y, 
                                dataset_name, 
                                cmap)
    
    
if __name__ == '__main__':
    # read_results can only be set to True  
    # if the corresponding file exists in results/results_files
    
    #main('Iris', read_results=True)
    #main('Digits', read_results=True)
    #main('MNIST', read_results=True)
    #main('FMNIST', read_results=True)
    #main('CIFAR-10', read_results=True)
    #main('CIFAR-100', read_results=True)
    #main('Spiral3D', read_results=True)
    main('Circle2D', read_results=True)
    