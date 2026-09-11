from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

class PcaUmap:
    @staticmethod
    def pca_nd(X, n_components: int=2):
        pca = PCA(n_components)
        X_pca = pca.fit_transform(X)

        plt.scatter(X_pca[:, 0], X_pca[:, 1])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA 2D")
        plt.show()

    @staticmethod
    def pca_class_colored(X, y):
        plt.figure(figsize=(8, 6))
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y,
            cmap="tab10",
            alpha=0.7
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA 2D — kolorowanie klasą")
        plt.colorbar(scatter, label="Klasa")
        plt.show()

    @staticmethod
    def pca_lambda_colored(X, lambda_values):
        plt.figure(figsize=(8, 6))
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=lambda_values,
            cmap="viridis",
            alpha=0.7
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA 2D — kolorowanie λ")
        plt.colorbar(scatter, label="λ")
        plt.show()

    @staticmethod
    def umap_nd(X, y):
        reducer = umap.UMAP(
            n_components=2,
            random_state=42
        )

        X_umap = reducer.fit_transform(X)

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            c=y,
            cmap="tab10",
            alpha=0.7
        )

        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title("UMAP 2D — kolorowanie klasą")
        plt.colorbar(scatter, label="Klasa")
        plt.show()

    @staticmethod
    def effective_dimension(X, variance=0.95):
        pca = PCA(n_components=variance)
        pca.fit(X)

        return pca.n_components_

    
