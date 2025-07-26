TRAIN_PATH  = "/mnt/DataNMVE/hilo/train.csv"
TEST_PATH = "/mnt/DataNMVE/hilo/test.csv"
LABELS_PATH = "/mnt/DataNMVE/hilo/train_labels.csv"
SR = 100.0  # Hz

ts_columns = [f"ppg_{i}" for i in range(3000)]
feats_columns = [f"features_{i}" for i in range(5)]