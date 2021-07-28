import scvelo as scv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import decomposition, manifold, preprocessing
from scipy.sparse import  issparse
import umap
import gtsne

sns.set(context="paper", style="white")

run = True

scv.logging.print_version()
adata = scv.datasets.pancreas()
from scvelo.plotting.utils import set_colors_for_categorical_obs
set_colors_for_categorical_obs(adata,'clusters')

scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

def get_X_y(adata,vkey="velocity",
        xkey="Ms",
        basis=None,
        gene_subset=None,
    ):

    subset = np.ones(adata.n_vars, bool)
    if gene_subset is not None:
        var_names_subset = adata.var_names.isin(gene_subset)
        subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
    elif f"{vkey}_genes" in adata.var.keys():
        subset &= np.array(adata.var[f"{vkey}_genes"].values, dtype=bool)

    xkey = xkey if xkey in adata.layers.keys() else "spliced"
    basis = 'umap' if basis is None else basis
    X = np.array(
        adata.layers[xkey].A[:, subset]
        if issparse(adata.layers[xkey])
        else adata.layers[xkey][:, subset]
    )

    y = adata.obs['clusters']
    catnew = adata.uns['clusters_colors']
    y_new =  y.cat.rename_categories(catnew)
    return X,y_new
X,y =get_X_y(adata)
y = y.values
test_data = [
    (X,y),
    ]
dataset_names = ["Pancreas",]

reducers = [
    (decomposition.PCA, {}),
    (gtsne.gtsne, {"alpha":1e-2, "beta":1e-5, "theta":0.5}),
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
            # embedding = rtsne.rtsne(data,d=2, K=90,K_U_mean=60,)
            reducer_name = "GTSNE"
            embedding = gtsne.gtsne(data, d=2, K = None, theta=0.5,alpha=1e-2,beta=5e-2,verbose=True )

        else:
            reducer_name = repr(reducer()).split("(")[0]
            embedding = reducer(n_components=2, **args).fit_transform(data)

        elapsed_time = time.time() - start_time
        adata.obsm['X2d_'+reducer_name] = embedding
        adata.uns['X2d_' + reducer_name+"_elapsed_time"] = elapsed_time
        ax = plt.subplot(n_rows, n_cols, ax_index)
        if isinstance(labels[0], tuple):
            ax.scatter(*embedding.T, s=10, c=labels,
                       )
        else:
            ax.scatter(*embedding.T, s=10, c=labels,
                       # cmap="Spectral","viridis_r"
                       cmap="tab10",
                       # alpha=0.5
                       )

        legend_loc ='on_data'
        clusters_ = adata.obs['clusters']
        categories = adata.obs['clusters'].cat.categories
        legend_fontsize = None
        legend_fontoutline = None
        legend_fontweight = "bold"

        if legend_loc == 'on_data' and reducer_name != "PCA":

            # identify centroids to put labels
            texts = []
            obs_vals = clusters_.values
            for label in categories:
                x_pos, y_pos = np.nanmedian(embedding[obs_vals == label, :], axis=0)
                if isinstance(label, str):
                    label = label.replace("_", " ")
                kwargs = dict(verticalalignment="center", horizontalalignment="center")
                kwargs.update(dict(weight=legend_fontweight, fontsize=legend_fontsize))
                text = ax.text(x_pos, y_pos, label,  **kwargs)
                texts.append(text)
        elif reducer_name == "PCA":
            legend_loc = "upper right"
            colors = adata.uns['clusters_colors']
            for idx, label in enumerate(categories):
                if isinstance(label, str):
                    label = label.replace("_", " ")
                ax.scatter([], [], c=[colors[idx]], label=label)
            ncol = 1 if len(categories) <= 14 else 2 if len(categories) <= 30 else 3
            kwargs = dict(frameon=False, fontsize=legend_fontsize, ncol=ncol)
            if legend_loc == "upper right":
                ax.legend(loc="upper left", bbox_to_anchor=(0, 1), **kwargs)
            elif legend_loc == "lower right":
                ax.legend(loc="lower left", bbox_to_anchor=(1, 0), **kwargs)
            elif "right" in legend_loc:  # 'right', 'center right', 'right margin'
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), **kwargs)
            elif legend_loc != "none":
                ax.legend(loc=legend_loc, **kwargs)


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

