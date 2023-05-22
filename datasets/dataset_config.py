from os.path import join

_BASE_DATA_PATH = "../SplitFSSEIDataset"

dataset_config = {
    'fs_sei': {
        'train_path': {
            'x': join(_BASE_DATA_PATH, 'X_split_train_90Classes.npy'),
            'y': join(_BASE_DATA_PATH, 'Y_split_train_90Classes.npy')
                       },
        'test_path': {
            'x': join(_BASE_DATA_PATH, 'X_split_val_90Classes.npy'),
            'y': join(_BASE_DATA_PATH, 'Y_split_val_90Classes.npy')
                       }
    }

}
