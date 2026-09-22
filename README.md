
# FracAug: Fractional Augmentation boost Graph-level Anomaly Detection under Limited Supervision

## Tested Environment

- Ubuntu 20.04
- [Python 3.8.18](https://www.anaconda.com/products/individual#Downloads)
- [Sklearn 1.3.2](https://scikit-learn.org/stable/install.html)
- [Pytorch 1.12.1](https://pytorch.org/get-started/locally/#linux-installation)
- [Numpy 1.24.4](https://numpy.org/install/)
- [Torch_geometric 2.4.0](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [Scipy 1.10.1](https://scipy.org/)

## Datasets

First, create directory "datasets". 

Second, use dataset.py to download dataset and generate training/validation/test set. 

**Example**

```
python dataset.py --data MCF-7 --khigh 3 --klow 3
```

**Directory Structure**

```
├── datasets
│   ├── MCF-7
│   │   ├── LM3
│   │   |	├── Es.pt
│   │   |	├── Us.pt
│   │   ├── MCF-7_test.txt
│   │   ├── MCF-7_train.txt
│   │   ├── MCF-7_val.txt
│   │   ├── processed
│   │   |	├── data.pt
│   │   |	├── pre_filter.pt
│   │   |	├── pre_transform.pt
│   │   ├── raw
│   │   |	├── MCF-7_A.txt
│   │   |	├── MCF-7_edge_labels.txt
│   │   |	├── MCF-7_graph_indicator.txt
│   │   |	├── MCF-7_graph_labels.txt
│   │   |	├── MCF-7_label_readme.txt
│   │   |	├── MCF-7_node_labels.txt
│   │   ├── SM3
│   │   |	├── Es.pt
│   │   |	├── Us.pt
├── dataset.py
```
## Experiments

**Parameters**
As described in paper. 

**Example**

```
python main.py --data MCF-7 --khigh 3 --klow 3
```

**Directory Structure**

```
├── datasets
├── configs.py
├── frameworks.py
├── gnns.py
├── main.py
├── model.py
├── utils.py
```