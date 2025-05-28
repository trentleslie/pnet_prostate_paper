import logging
import numpy as np
from data.prostate_paper.data_reader import ProstateDataPaper

class Data():
    def __init__(self, id, type, params, test_size=0.3, stratify=True, include_histology_features=None):
        """
        Initialize Data class for loading and processing prostate cancer data.
        
        Args:
            id: Data identifier
            type: Data type (e.g., 'prostate_paper')
            params: Parameters for data loading
            test_size: Fraction of data to use for testing
            stratify: Whether to stratify splits
            include_histology_features: Whether to include histology data features.
                                      If None, defaults to False (genomic data only).
                                      Currently, histology features are not implemented,
                                      so this parameter ensures explicit behavior.
        """
        self.test_size = test_size
        self.stratify = stratify
        self.data_type = type
        self.data_params = params
        
        # Handle histology features parameter
        if include_histology_features is None:
            include_histology_features = False
        
        self.include_histology_features = include_histology_features
        
        if include_histology_features:
            logging.warning('include_histology_features=True specified, but histology features are not yet '
                           'implemented. Using genomic data only.')
        
        if self.data_type == 'prostate_paper':
            # Pass through all params to ProstateDataPaper - it will ignore unknown parameters
            self.data_reader = ProstateDataPaper(**params)
            logging.info(f'Data initialized with genomic features only (include_histology_features={include_histology_features})')
        else:
            logging.error('unsupported data type')
            raise ValueError('unsupported data type')

    def get_train_validate_test(self):
        return self.data_reader.get_train_validate_test()

    def get_train_test(self):
        x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = self.data_reader.get_train_validate_test()
        # combine training and validation datasets
        x_train = np.concatenate((x_train, x_validate))
        y_train = np.concatenate((y_train, y_validate))
        info_train = list(info_train) + list(info_validate)
        return x_train, x_test, y_train, y_test, info_train, info_test, columns

    def get_data(self):
        x = self.data_reader.x
        y = self.data_reader.y
        info = self.data_reader.info
        columns = self.data_reader.columns
        return x, y, info, columns

    def get_relevant_features(self):
        if hasattr(self.data_reader, 'relevant_features'):
            return self.data_reader.get_relevant_features()
        else:
            return None
