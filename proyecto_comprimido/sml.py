#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Machine Learning Library for Python
Converted from Perl sml.pm module

Uses only native Python commands and specified libraries: MXNet, Plotly, Regex
For plain text file reading, uses native open()
For HTML data manipulation, uses regular expressions (Regex)
Everything else developed with native language commands.
"""

import random
import math
import re
import statistics
from typing import List, Tuple, Optional, Union, Callable, Any, Dict
import numpy as np

# Note: MXNet and Plotly would be imported here when available
# import mxnet as mx
# import plotly.graph_objects as go
# import plotly.express as px

class SimpleMLLibrary:
    """Simple Machine Learning Library with native Python implementations"""
    
    def __init__(self):
        self.methods = {}
    
    def add_to_class(self, name: str, method: Callable):
        """Add method to class dynamically"""
        self.methods[name] = method
        setattr(self, name.replace(' ', '_'), method)
    
    # ===============================
    # Chapter 2: Data Preprocessing
    # ===============================
    
    def dataset_minmax(self, dataset: List[List[float]]) -> List[List[float]]:
        """
        Calculates the minimum and maximum values for each column in the dataset.
        
        Parameters:
            dataset: List of lists containing the dataset
        
        Returns:
            List with shape [2, n_features] where:
            - First row contains minimum values for each feature
            - Second row contains maximum values for each feature
        """
        if not dataset or not dataset[0]:
            return []
        
        n_features = len(dataset[0])
        min_values = [float('inf')] * n_features
        max_values = [float('-inf')] * n_features
        
        for row in dataset:
            for i, value in enumerate(row):
                if i < n_features:
                    min_values[i] = min(min_values[i], value)
                    max_values[i] = max(max_values[i], value)
        
        return [min_values, max_values]
    
    def normalize_dataset(self, dataset: List[List[float]], minmax: List[List[float]]) -> List[List[float]]:
        """
        Performs Min-Max normalization on the dataset, scaling all values to [0,1] range.
        
        Parameters:
            dataset: Dataset to normalize (modified in-place)
            minmax: Min/max values from dataset_minmax()
        
        Formula:
            normalized_value = (original_value - min) / (max - min)
        """
        if not minmax or len(minmax) != 2:
            return dataset
        
        min_values, max_values = minmax[0], minmax[1]
        
        for row in dataset:
            for i in range(len(row)):
                if i < len(min_values) and i < len(max_values):
                    range_val = max_values[i] - min_values[i]
                    if range_val != 0:
                        row[i] = (row[i] - min_values[i]) / range_val
                    else:
                        row[i] = 0.0
        
        return dataset
    
    def column_means(self, dataset: List[List[float]]) -> List[float]:
        """
        Calculates the arithmetic mean for each column (feature) in the dataset.
        
        Parameters:
            dataset: List of lists containing the dataset
        
        Returns:
            List containing mean values for each column
        """
        if not dataset or not dataset[0]:
            return []
        
        n_features = len(dataset[0])
        n_samples = len(dataset)
        means = [0.0] * n_features
        
        for row in dataset:
            for i, value in enumerate(row):
                if i < n_features:
                    means[i] += value
        
        return [mean / n_samples for mean in means]
    
    def column_stdevs(self, dataset: List[List[float]], means: List[float]) -> List[float]:
        """
        Calculates the standard deviation for each column using sample variance.
        
        Parameters:
            dataset: List of lists containing the dataset
            means: List with mean values from column_means()
        
        Returns:
            List containing standard deviation values for each column
        
        Formula:
            stdev = sqrt(sum((x - mean)^2) / (n - 1))
        """
        if not dataset or not dataset[0] or not means:
            return []
        
        n_features = len(dataset[0])
        n_samples = len(dataset)
        variances = [0.0] * n_features
        
        for row in dataset:
            for i, value in enumerate(row):
                if i < n_features and i < len(means):
                    variances[i] += (value - means[i]) ** 2
        
        # Sample standard deviation (n-1 denominator)
        return [math.sqrt(var / (n_samples - 1)) if n_samples > 1 else 0.0 for var in variances]
    
    def standardize_dataset(self, dataset: List[List[float]], means: List[float], stdevs: List[float]) -> List[List[float]]:
        """
        Performs Z-score standardization on the dataset (mean=0, std=1).
        
        Parameters:
            dataset: Dataset to standardize (modified in-place)
            means: List with mean values
            stdevs: List with standard deviation values
        
        Formula:
            standardized_value = (original_value - mean) / standard_deviation
        """
        for row in dataset:
            for i in range(len(row)):
                if (i < len(means) and i < len(stdevs) and stdevs[i] != 0):
                    row[i] = (row[i] - means[i]) / stdevs[i]
                elif i < len(means):
                    row[i] = row[i] - means[i]  # If stdev is 0, just center the data
        
        return dataset
    
    # ===============================
    # Chapter 3: Data Splitting
    # ===============================
    
    def train_test_split(self, dataset: List[List[float]], split: float = 0.6) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Randomly splits a dataset into training and testing portions.
        
        Parameters:
            dataset: Dataset to split
            split: Fraction for training set (default: 0.6 = 60% training)
        
        Returns:
            (train, test): Two lists containing training and testing data
        """
        if not dataset:
            return [], []
        
        dataset_copy = dataset.copy()
        random.shuffle(dataset_copy)
        
        train_size = int(split * len(dataset_copy))
        train = dataset_copy[:train_size]
        test = dataset_copy[train_size:]
        
        return train, test
    
    def cross_validation_split(self, dataset: List[List[float]], n_folds: int = 10) -> List[List[List[float]]]:
        """
        Splits dataset into k-folds for cross-validation evaluation.
        
        Parameters:
            dataset: Dataset to split
            n_folds: Number of folds (default: 10)
        
        Returns:
            List of folds, each containing a portion of the dataset
        """
        if not dataset:
            return []
        
        dataset_copy = dataset.copy()
        random.shuffle(dataset_copy)
        
        fold_size = len(dataset_copy) // n_folds
        folds = []
        
        for i in range(n_folds):
            start_idx = i * fold_size
            if i == n_folds - 1:  # Last fold gets remaining samples
                end_idx = len(dataset_copy)
            else:
                end_idx = (i + 1) * fold_size
            
            folds.append(dataset_copy[start_idx:end_idx])
        
        return folds
    
    def count_labels(self, dataset: List[List[Union[float, int]]]) -> Dict[int, int]:
        """
        Counts the frequency of each class label in a classification dataset.
        
        Parameters:
            dataset: Dataset where last column contains class labels
        
        Returns:
            Dictionary containing count for each class label
        """
        if not dataset or not dataset[0]:
            return {}
        
        label_counts = {}
        for row in dataset:
            label = int(row[-1])  # Last column contains labels
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return label_counts
    
    # ===============================
    # Chapter 4: Evaluation Metrics
    # ===============================
    
    def accuracy_metric(self, actual: List[Union[int, float]], predicted: List[Union[int, float]]) -> str:
        """
        Calculates classification accuracy as percentage of correct predictions.
        
        Parameters:
            actual: List with true class labels
            predicted: List with predicted class labels
        
        Returns:
            String with accuracy percentage formatted to 2 decimal places
        """
        if not actual or not predicted or len(actual) != len(predicted):
            return "0.00"
        
        correct = sum(1 for a, p in zip(actual, predicted) if a == p)
        accuracy = (correct / len(actual)) * 100
        
        return f"{accuracy:.2f}"
    
    def confusion_matrix(self, actual: List[int], predicted: List[int]) -> Tuple[List[int], List[List[int]]]:
        """
        Generates a confusion matrix for multi-class classification problems.
        
        Parameters:
            actual: List with true class labels
            predicted: List with predicted class labels
        
        Returns:
            (classes, matrix): classes list and 2D matrix where matrix[i][j] = 
            count of actual class i predicted as class j
        """
        if not actual or not predicted:
            return [], []
        
        # Get unique classes
        all_labels = set(actual + predicted)
        classes = sorted(list(all_labels))
        n_classes = len(classes)
        
        # Create mapping from label to index
        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        
        # Initialize confusion matrix
        matrix = [[0] * n_classes for _ in range(n_classes)]
        
        # Fill confusion matrix
        for a, p in zip(actual, predicted):
            actual_idx = label_to_idx[a]
            predicted_idx = label_to_idx[p]
            matrix[actual_idx][predicted_idx] += 1
        
        return classes, matrix
    
    def print_confusion_matrix(self, classes: List[int], matrix: List[List[int]]):
        """
        Pretty prints a confusion matrix with proper formatting and labels.
        
        Parameters:
            classes: List with class labels/indices
            matrix: 2D confusion matrix from confusion_matrix()
        """
        print("A/P", end="")
        for cls in classes:
            print(f"\t{cls}", end="")
        print()
        
        for i, cls in enumerate(classes):
            print(f"{cls}", end="")
            for j in range(len(classes)):
                print(f"\t{matrix[i][j]}", end="")
            print()
    
    def mae_metric(self, actual: List[float], predicted: List[float]) -> str:
        """
        Calculates Mean Absolute Error for regression problems.
        
        Parameters:
            actual: List with true continuous values
            predicted: List with predicted continuous values
        
        Returns:
            String with MAE formatted to 2 decimal places
        """
        if not actual or not predicted or len(actual) != len(predicted):
            return "0.00"
        
        mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)
        return f"{mae:.2f}"
    
    def rmse_metric(self, actual: List[float], predicted: List[float]) -> str:
        """
        Calculates Root Mean Square Error for regression problems.
        
        Parameters:
            actual: List with true continuous values
            predicted: List with predicted continuous values
        
        Returns:
            String with RMSE formatted to 3 decimal places
        """
        if not actual or not predicted or len(actual) != len(predicted):
            return "0.000"
        
        mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
        rmse = math.sqrt(mse)
        return f"{rmse:.3f}"
    
    def perf_metrics(self, actual: List[int], predicted_prob: List[float], threshold: float) -> Tuple[str, str]:
        """
        Calculates performance metrics for binary classification: TPR and FPR.
        
        Parameters:
            actual: List with true binary labels (0 or 1)
            predicted_prob: List with predicted probabilities [0,1]
            threshold: Decision threshold for converting probabilities to binary predictions
        
        Returns:
            (fpr, tpr): False Positive Rate and True Positive Rate as formatted strings
        """
        if not actual or not predicted_prob or len(actual) != len(predicted_prob):
            return "0.00", "0.00"
        
        # Apply threshold to create binary predictions
        predicted = [1 if prob >= threshold else 0 for prob in predicted_prob]
        
        # Calculate confusion matrix components
        tp = fp = tn = fn = 0
        
        for a, p in zip(actual, predicted):
            if a == 1 and p == 1:
                tp += 1
            elif a == 0 and p == 1:
                fp += 1
            elif a == 0 and p == 0:
                tn += 1
            else:  # a == 1 and p == 0
                fn += 1
        
        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return f"{fpr:.2f}", f"{tpr:.2f}"
    
    # ===============================
    # Chapter 5: Baseline Algorithms
    # ===============================
    
    def random_algorithm(self, train: List[List[Union[int, float]]], test: List[List[Union[int, float]]]) -> List[Union[int, float]]:
        """
        Baseline algorithm that makes random predictions from training set labels.
        
        Parameters:
            train: Training data (last column contains labels)
            test: Test data (features only)
        
        Returns:
            List with random predictions for test set
        """
        if not train or not test:
            return []
        
        # Extract labels from training set
        labels = [row[-1] for row in train]
        
        # Return random predictions
        return [random.choice(labels) for _ in range(len(test))]
    
    def zero_rule_algorithm_classification(self, train: List[List[Union[int, float]]], test: List[List[Union[int, float]]]) -> List[Union[int, float]]:
        """
        Baseline classification algorithm that always predicts the most frequent class.
        
        Parameters:
            train: Training data (last column contains labels)
            test: Test data
        
        Returns:
            List filled with the most frequent class from training data
        """
        if not train or not test:
            return []
        
        # Count class frequencies
        label_counts = {}
        for row in train:
            label = row[-1]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Find most frequent class
        most_frequent_class = max(label_counts, key=label_counts.get)
        
        # Return predictions filled with most frequent class
        return [most_frequent_class] * len(test)
    
    def zero_rule_algorithm_regression(self, train: List[List[float]], test: List[List[float]]) -> List[float]:
        """
        Baseline regression algorithm that always predicts the mean of training targets.
        
        Parameters:
            train: Training data (last column contains target values)
            test: Test data
        
        Returns:
            List filled with the mean target value from training data
        """
        if not train or not test:
            return []
        
        # Calculate mean of training targets
        targets = [row[-1] for row in train]
        mean_target = sum(targets) / len(targets)
        
        # Return predictions filled with mean
        return [mean_target] * len(test)
    
    # ===============================
    # Chapter 6: Model Evaluation
    # ===============================
    
    def evaluate_algorithm_train_test_split(self, dataset: List[List[Union[int, float]]], 
                                          algorithm: Callable, split: float = 0.6, 
                                          metric: Optional[str] = None) -> Union[str, Tuple]:
        """
        Comprehensive algorithm evaluation using train-test split methodology.
        
        Parameters:
            dataset: Complete dataset
            algorithm: Function reference to algorithm function
            split: Train/test ratio (default: 0.6)
            metric: Evaluation metric ('accuracy', 'rmse', or auto-detect)
        
        Returns:
            Evaluation score (or tuple with detailed results if requested)
        """
        # Split dataset
        train, test = self.train_test_split(dataset, split)
        
        if not train or not test:
            return "0.00"
        
        # Prepare test set (copy without modification for now)
        test_set = [row.copy() for row in test]
        
        # Run algorithm
        try:
            predicted = algorithm(self, train, test_set)
        except Exception as e:
            print(f"Algorithm error: {e}")
            return "0.00"
        
        # Extract actual values
        actual = [row[-1] for row in test]
        
        # Determine metric
        if metric is None:
            # Auto-detect: use RMSE if any float values, else accuracy
            has_floats = any(isinstance(val, float) and val != int(val) for val in actual)
            metric = "rmse" if has_floats else "accuracy"
        
        # Calculate score
        if metric.lower() == "accuracy":
            score = self.accuracy_metric(actual, predicted)
        elif metric.lower() == "rmse":
            score = self.rmse_metric(actual, predicted)
        elif metric.lower() == "mae":
            score = self.mae_metric(actual, predicted)
        else:
            score = self.accuracy_metric(actual, predicted)
        
        return score
    
    # ===============================
    # Utility Functions
    # ===============================
    
    def load_csv(self, file_path: str, has_header: bool = False) -> List[List[Union[str, float]]]:
        """
        Load dataset from CSV file using native Python
        
        Parameters:
            file_path: Path to CSV file
            has_header: Whether file has header row
        
        Returns:
            List of lists containing the dataset
        """
        dataset = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
                start_idx = 1 if has_header else 0
                
                for line in lines[start_idx:]:
                    line = line.strip()
                    if line:
                        values = []
                        for val in line.split(','):
                            val = val.strip()
                            try:
                                # Try to convert to float
                                values.append(float(val))
                            except ValueError:
                                # Keep as string if conversion fails
                                values.append(val)
                        dataset.append(values)
        
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return []
        
        return dataset
    
    def save_csv(self, dataset: List[List[Union[str, float]]], file_path: str, 
                 header: Optional[List[str]] = None):
        """
        Save dataset to CSV file using native Python
        
        Parameters:
            dataset: Dataset to save
            file_path: Output file path
            header: Optional header row
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                if header:
                    file.write(','.join(header) + '\n')
                
                for row in dataset:
                    row_str = ','.join(str(val) for val in row)
                    file.write(row_str + '\n')
        
        except Exception as e:
            print(f"Error saving CSV file: {e}")


# Create global instance
sml = SimpleMLLibrary()

# Add methods to class
sml.add_to_class('dataset_minmax', sml.dataset_minmax)
sml.add_to_class('normalize_dataset', sml.normalize_dataset)
sml.add_to_class('column_means', sml.column_means)
sml.add_to_class('column_stdevs', sml.column_stdevs)
sml.add_to_class('standardize_dataset', sml.standardize_dataset)
sml.add_to_class('train_test_split', sml.train_test_split)
sml.add_to_class('cross_validation_split', sml.cross_validation_split)
sml.add_to_class('count_labels', sml.count_labels)
sml.add_to_class('accuracy_metric', sml.accuracy_metric)
sml.add_to_class('confusion_matrix', sml.confusion_matrix)
sml.add_to_class('print_confusion_matrix', sml.print_confusion_matrix)
sml.add_to_class('mae_metric', sml.mae_metric)
sml.add_to_class('rmse_metric', sml.rmse_metric)
sml.add_to_class('perf_metrics', sml.perf_metrics)
sml.add_to_class('random_algorithm', sml.random_algorithm)
sml.add_to_class('zero_rule_algorithm_classification', sml.zero_rule_algorithm_classification)
sml.add_to_class('zero_rule_algorithm_regression', sml.zero_rule_algorithm_regression)
sml.add_to_class('evaluate_algorithm_train_test_split', sml.evaluate_algorithm_train_test_split)


if __name__ == "__main__":
    # Example usage
    print("Simple Machine Learning Library for Python")
    print("Converted from Perl sml.pm module")
    
    # Example dataset
    dataset = [
        [1.0, 2.0, 0],
        [2.0, 3.0, 1], 
        [3.0, 4.0, 1],
        [4.0, 5.0, 0],
        [5.0, 6.0, 1]
    ]
    
    print(f"\nExample dataset: {dataset}")
    
    # Data preprocessing
    minmax = sml.dataset_minmax(dataset)
    print(f"Min-Max values: {minmax}")
    
    # Evaluation
    score = sml.evaluate_algorithm_train_test_split(
        dataset, 
        sml.zero_rule_algorithm_classification
    )
    print(f"Baseline accuracy: {score}%")