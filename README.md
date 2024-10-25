# IMDB-Review-Classifier

The project is an IMDB review classifier that uses the machine learning algorithms: MLP (Multi-Layer Perceptron), Random Forest and AdaBoost, to classify movie reviews as positive or negative. The project structure includes scripts for training and evaluating these models, as well as data files for training.

## Project Structure
- **Main.py**: Calculates the total frequency of each word in the vocabulary based on the labeled term frequency data from the input files. Specifically, it reads a labeled term frequency file and a vocabulary file, then processes these files to compute how often each word in the vocabulary appears across the entire dataset.
- **MLPword_embeddings.py**: Contains code for training and evaluating an MLP model using word embeddings.
- **RandomForest/**: Directory containing scripts related to the Random Forest algorithm.
  - **id3.py**
  - **RandomForest.py** custom RandomForest algorithm with ID3 decision trees
  - **SKlearn_RandomForest.py** uses scikit-learn's built-in RandomForest classifier
- **Report.docx**: Project details and results.
- **traindataframes/**: Directory containing training data files.
  - **negative_m500_n21_k25720.csv**
  - **positive_m500_n21_k25720.csv**
- **ΑdaBoost/**: Directory containing scripts related to the AdaBoost algorithm.
  - **adaboost.py**
  - **mainAB.py**  custom AdaBoost implementation
  - **Scikit-learn_mainAB.py** uses scikit-learn's built-in AdaBoost classifier and provides more detailed metric calculations and iterative evaluation.

## Key Scripts
- **ΑdaBoost/Scikit-learn_mainAB.py**: Contains code for plotting learning curves and metrics for the AdaBoost model.
- **MLPword_embeddings.py**: Contains code for training and evaluating an MLP model, including detailed evaluation metrics and plotting.
- **ΑdaBoost/adaboost.py**: Contains the implementation of the AdaBoost algorithm, including training and prediction functions.

## Data
- **Training Data**: Located in the `traindataframes/` directory, containing CSV files for positive and negative reviews.

## Dependencies
- **TensorFlow**: Used for building and training the MLP model.
- **Scikit-learn**: Used for metrics and other machine learning algorithms.
- **NumPy**: Used for numerical operations.
- **Pandas**: Used for data manipulation and analysis.
- **Matplotlib**: Used for plotting learning curves and metrics.

## Usage
To train and evaluate the models, you should run the scripts in the respective directories. For example, to evaluate the MLP model, you might run `MLPword_embeddings.py`. For AdaBoost, you might run `ΑdaBoost/Scikit-learn_mainAB.py` or `mainAB.py`.
