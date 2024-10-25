# IMDB Review Classifier

This project implements various machine learning models to classify IMDB reviews as positive or negative. The models include Multi-Layer Perceptron (MLP), Random Forest, and AdaBoost.

## Requirements

- Python 3.x
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- gensim

Install the required packages using:

```sh
pip install -r requirements.txt
```

## Usage

### Multi-Layer Perceptron (MLP)

The MLP model is implemented in [`MLPword_embeddings.py`](MLPword_embeddings.py). It uses word embeddings to classify the reviews.

To run the MLP model:

```sh
python 

MLPword_embeddings.py


```

### Random Forest

The Random Forest model is implemented in [`RandomForest/RandomForest.py`](RandomForest/RandomForest.py) and uses the ID3 algorithm defined in [`RandomForest/id3.py`](RandomForest/id3.py).

To run the Random Forest model:

```sh
python 

RandomForest.py


```

### AdaBoost

The AdaBoost model is implemented in [`ΑdaBoost/adaboost.py`](ΑdaBoost/adaboost.py).

To run the AdaBoost model:

```sh
python 

mainAB.py


```

## Data

The training data is stored in the `traindataframes` directory and includes:

- `negative_m500_n21_k25720.csv`: Negative reviews
- `positive_m500_n21_k25720.csv`: Positive reviews

## Evaluation

The evaluation metrics include accuracy, precision, recall, and F1 score. The results are printed and plotted for analysis.
