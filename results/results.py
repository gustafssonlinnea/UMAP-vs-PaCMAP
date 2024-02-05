import pickle
import pandas as pd


class Results():
    def __init__(self, Xs, metric_results, params, n_knn_neighbors):
        self.Xs = Xs
        self.metric_results = metric_results
        self.params = params
        self.n_knn_neighbors = n_knn_neighbors
        
    def write_results_to_file(self,
                              file_name='result',
                              file_path='results/results_files'):
        # Writing to a file
        with open(f'{file_path}/{file_name}.pkl', 'wb') as file:
            data_to_save = {'results': self}
            pickle.dump(data_to_save, file)
        
        print(f'Results are saved to {file_path}/{file_name}.pkl')
    
    @staticmethod
    def read_results_from_file(file_name='result', 
                               file_path='results/results_files'):
        # Reading from a file
        with open(f'{file_path}/{file_name}.pkl', 'rb') as file:
            loaded_data = pickle.load(file)

        loaded_results = loaded_data['results']
        print(f'Results are loaded from {file_path}/{file_name}.pkl')
        
        return loaded_results

    @staticmethod
    def print_results(results_umap, results_pacmap):
        table = [[results_umap.metric_results['knn'], 
                    results_umap.params['knn'], 
                    results_pacmap.metric_results['knn'], 
                    results_pacmap.params['knn']],
                [results_umap.metric_results['svm'],
                    results_umap.params['svm'],
                    results_pacmap.metric_results['svm'],
                    results_pacmap.params['svm']],
                [results_umap.metric_results['cte'], 
                    results_umap.params['cte'],
                    results_pacmap.metric_results['cte'],
                    results_pacmap.params['cte']],
                [results_umap.metric_results['rte'], 
                    results_umap.params['rte'],
                    results_pacmap.metric_results['rte'],
                    results_pacmap.params['rte']],
                [results_umap.n_knn_neighbors, 
                    '', 
                    results_pacmap.n_knn_neighbors, 
                    '']]
        
        df = pd.DataFrame(table, 
                        columns=['UMAP', 'Neighbors', 'PaCMAP', 'Neighbors'], 
                        index=['KNN accuracy', 
                                'SVM accuracy', 
                                'Centroid triplet accuracy', 
                                'Random triplet accuracy',
                                'KNN neighbors'],
                        dtype='object')
        print('\n', df)
        
        return df
