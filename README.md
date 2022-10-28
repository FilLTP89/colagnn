# Colagnn

Project inspired by the original architecture presented in:

[Deng, S. and Wang, S. and Rangwala, H. and Wang, L. and Ning, Y. (2020) _Cola-GNN: Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction_. Proceedings of the 29th ACM International Conference on Information & Knowledge Management](https://dl.acm.org/doi/10.1145/3340531.3411975)

This is the source code for paper [Cola-GNN: Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction](https://yue-ning.github.io/docs/CIKM20-colagnn.pdf) appeared in CIKM2020 (research track)


## Abstract

Forecasting influenza-like illness (ILI) is of prime importance to epidemiologists and health-care providers. Early prediction of epidemic outbreaks plays a pivotal role in disease intervention and control. Most existing work has either limited long-term prediction performance or fails to capture spatio-temporal dependencies in data. In this paper, we design a cross-location attention based graph neural network (Cola-GNN) for learning time series embeddings in long-term ILI predictions. We propose a graph message passing framework to combine graph structures (e.g., geolocations) and time-series features (e.g., temporal sequences) in a dynamic propagation process. We compare the proposed method with stateof-the-art statistical approaches and deep learning models. We conducted a set of extensive experiments on real-world epidemicrelated datasets from the United States and Japan. The proposed method demonstrated strong predictive performance and leads to interpretable results for long-term epidemic predictions.

## Raw Data
The raw dataset are in in the `data` folder. For each dataset, there are two files defined. For example, for the `Japan-prefecture` dataset, we have two files:
- `japan.txt` includes the spatiotemporal data. Columns indicate locations (i.e., prefecture) and rows indicate timestamps (i.e., weeks). Each value is the number of patients in a location at a time point. The data are arranged in chronological order.
- `japan-adj.txt` contains a adjacency matrix.


## Training Data
The training data are processed by the **DataBasicLoader**
class in the `src/data.py` file. We can set different value for historical window size **args.window** and horizon/leadtime **args.horizon**. Setting **args.window=20, args.horizon=1/2** means using data from the previous 20 weeks to predict the *upcoming*/*next* week. There are some functions in this class:
- **_split** splits the data into training/validation/test sets.
- **_batchify** generates data samples. Each sample contains a time series input with length equal to **args.window**, and a value for the output. For the current code, there are overlaps in the inputs of different samples.
- **get_batches** generates random mini-batches for training.
