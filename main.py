from sklearn import datasets
# from sklearn.datasets import fetch_openml
from UMAP import UMAP
from PaCMAP import PaCMAP
import evaluation
import visualization


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


def main(dataset_name, figure_path='figures', figure_name='fig'):
    # Load data
    X, y = get_dataset_by_name(dataset_name)
    
    # UMAP
    umap = UMAP()
    umap.fit(X) 
    #umap.plot_embedding(y, dataset_name)
    X_new_umap = umap.get_embedding()
    results_umap = evaluation.evaluate_output(X, X_new_umap, y, 'UMAP')
    evaluation.print_evaluation_results(results_umap)
    # visualization.plot_embedding(X_new_umap, y, 'UMAP', dataset_name)
    #visualization.plot_embeddings(y, dataset_name, {'UMAP': X_new_umap})
    
    # PaCMAP
    pacmap = PaCMAP()
    pacmap.fit(X)
    #pacmap.plot_embedding(y, dataset_name)
    X_new_pacmap = pacmap.get_embedding()
    results_pacmap = evaluation.evaluate_output(X, X_new_pacmap, y, 'PaCMAP')
    evaluation.print_evaluation_results(results_pacmap)
    #visualization.plot_embedding(X_new_pacmap, y, 'PaCMAP', dataset_name)
    
    # Plot embeddings    
    visualization.plot_embeddings(y, 
                                  dataset_name,
                                  100,
                                  0.3,
                                  figure_path,
                                  figure_name,
                                  {'UMAP': X_new_umap}, {'PaCMAP': X_new_pacmap})
    
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
