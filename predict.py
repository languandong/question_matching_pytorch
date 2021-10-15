from config import Config
from total_utils.dataIter import DataIterator
from total_utils.predict_utils import predict

if __name__ == '__main__':
    config = Config()
    test_iter = DataIterator(config, data_file=config.source_data_dir + "test_A.tsv", is_test=True)
    predict(config, test_iter)