from sklearn.datasets import load_digits
import umap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

    digits = load_digits()
    # print(digits.DESCR)

    """fig, ax_array = plt.subplots(20, 20)
    axes = ax_array.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(digits.images[i], cmap='gray_r')
    plt.setp(axes, xticks=[], yticks=[], frame_on=False)
    plt.tight_layout(h_pad=0.5, w_pad=0.01)
    plt.show()"""

    """digits_df = pd.DataFrame(digits.data[:,1:11])
    digits_df['digit'] = pd.Series(digits.target).map(lambda x: 'Digit {}'.format(x))
    sns.pairplot(digits_df, hue='digit', palette='Spectral')"""



    """
    UMAP(a=None, angular_rp_forest=False, b=None,
        force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
        local_connectivity=1.0, low_memory=False, metric='euclidean',
        metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
        n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
        output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
        set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
        target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
        transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)
    """
    # help(umap.UMAP)
    reducer = umap.UMAP(min_dist=0.1, n_components=2, n_epochs=500, n_neighbors=1000)
    reducer.fit(digits.data)

    embedding = reducer.transform(digits.data)
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert(np.all(embedding == reducer.embedding_))
    print(embedding.shape)

    two_d = False
    if reducer.n_components == 2:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
        
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        plt.title('UMAP projection of the Digits dataset', fontsize=24)
        plt.show()
    elif reducer.n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D  # Importing the 3D plotting toolkit

        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # Adding a 3D subplot

    # Scatter plot
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=digits.target, cmap='Spectral', s=5)

        # Set aspect ratio
        ax.set_aspect('equal', 'datalim')

        # Add colorbar
        colorbar = fig.colorbar(scatter, ax=ax, boundaries=np.arange(11)-0.5)
        colorbar.set_ticks(np.arange(10))

        # Set title
        ax.set_title('UMAP projection of the Digits dataset', fontsize=24)
        plt.show()

if __name__ == "__main__":
    main()
