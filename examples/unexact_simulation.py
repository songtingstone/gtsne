import os
from anndata import AnnData, read_h5ad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import decomposition, manifold, preprocessing
import umap
import gtsne

sns.set(context="paper", style="white")

def simulate_data_base(N = 100, D = 30,  nrep=1, resimulate =False, save =True, file_prefix = "./data/"):
    # 3 types
    file_base = f"unexact_simulation_data_N_{N}_D_{D}_nrep_{nrep}.h5ad"
    file = os.path.join(file_prefix, file_base)
    if(not os.path.exists(os.path.abspath(file_prefix))):
        print(f"Create directory {os.path.abspath(file_prefix)}\n")
        os.makedirs(os.path.abspath(file_prefix))
    if os.path.exists(file) and not resimulate:
        adata = read_h5ad(file)
        return adata

    V = np.abs(np.random.randn(*(N * 3, D)))* 6
    x_1 = np.asarray([0,]*D)
    x_2 =  np.asarray([50,]*D)
    x_3 = np.asarray([160, ]*D)
    X = np.zeros_like(V)
    X[0, :] = x_1
    X[N, :] = x_2
    X[N*2, :] = x_3
    for i in np.arange(N-1):
        X[i+1,:] = X[i,:] + V[i,:]
        # X[i + N +1, :] = X[i+ N, :] + V[i+ N, :]
        # X[i + N*2 + 1, :] = X[i + N*2, :] + V[i + N*2, :]
        X[i + N + 1, :] = X[i + N, :] + V[i, :]
        X[i + N * 2 + 1, :] = X[i + N * 2, :] + V[i, :]

    y = ["r",] *N + ["y",]  * N + [ "b",]*N

    adata = AnnData(X=X,  obs={"clusters":y})
    if save:
        adata.write_h5ad(file)
    return adata



N = 700
D = 3
nrep = 6
basis = "tsne"
theta_tsne = 0.5
perplexity=30.0
resimulate = False

# resimulate = True
adata = simulate_data_base(N = N, D = D, nrep=nrep, resimulate =resimulate,
                           save =True, file_prefix ="./data/")
X = adata.X
y = ["r",] *N + ["y",]  * N + [ "b",]*N

test_data = [
    (X,y),
]
dataset_names = ["Three lines (N={}, D={})".format(3*N,D),]



reducers = [
    (decomposition.PCA, {}),
    (gtsne.gtsne, { }),
    (manifold.TSNE, {"perplexity": 30}),
    (manifold.Isomap, {"n_neighbors": 30}),
    (manifold.MDS, {}),
    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3}),

]
n_rows = len(test_data)
n_cols = len(reducers)
ax_index = 1
ax_list = []

plt.figure(figsize=(10, 3.3))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
)
for data, labels in test_data:
    for reducer, args in reducers:
        start_time = time.time()
        if reducer == gtsne.gtsne:
            embedding = gtsne.gtsne(data, d=2, K = None, theta=0.5,alpha=1e-2,beta=5e-2,verbose=True )
        else:
            embedding = reducer(n_components=2, **args).fit_transform(data)
        elapsed_time = time.time() - start_time
        ax = plt.subplot(n_rows, n_cols, ax_index)
        if isinstance(labels[0], tuple):
            ax.scatter(*embedding.T, s=10, c=labels,
                       # alpha=0.5
                       )
        else:
            ax.scatter(*embedding.T, s=10, c=labels,
                       # cmap="Spectral",
                       # cmap="tab10",
                       # alpha=0.5
                       )

        ax.text(
            0.99,
            0.01,
            "{:.2f} s".format(elapsed_time),
            transform=ax.transAxes,
            size=14,
            horizontalalignment="right",
        )
        ax_list.append(ax)
        ax_index += 1
plt.setp(ax_list, xticks=[], yticks=[])

for i in np.arange(n_rows) * n_cols:
    ax_list[i].set_ylabel(dataset_names[i // n_cols], size=16)
for i in range(n_cols):
    if reducers[i][0] != gtsne.gtsne:
        ax_list[i].set_xlabel(repr(reducers[i][0]()).split("(")[0], size=16)
        ax_list[i].xaxis.set_label_position("top")
    else:
        ax_list[i].set_xlabel("GTSNE", size=16)
        ax_list[i].xaxis.set_label_position("top")


plt.tight_layout()
plt.show()

