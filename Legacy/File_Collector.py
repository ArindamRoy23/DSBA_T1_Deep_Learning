import os
import pandas as pd
import numpy as np
class File_Collector:
    def __init__(self,
                 MASK_PATH = r'Hurricane_Harvey\Hurricane_Harvey\vectors\random-split-_2022_11_17-22_35_45\Masks',
                 IMAGE_PATH = r'Hurricane_Harvey\Hurricane_Harvey\rasters\raw'):
        self.MASK_PATH = MASK_PATH
        self.IMAGE_PATH = IMAGE_PATH

    def create_df(self):
        train_set = []
        for dirname, _, filenames in os.walk(self.MASK_PATH):
            for filename in filenames:
                train_set.append(filename.split('.')[0])

        name = []

        name_test_final = []
        for dirname, _, filenames in os.walk(self.IMAGE_PATH):
            for filename in filenames:
                if filename.split('.')[0] in train_set:
                    name.append(filename.split('.')[0])
                else:
                    name_test_final.append(filename.split('.')[0])
                # print(filename.split('.')[0])

        return (pd.DataFrame({'id': name}, index=np.arange(0, len(name))), \
                pd.DataFrame({'id': name_test_final}, index=np.arange(0, len(name_test_final))))

if __name__ == '__main__':
    print(File_Collector().create_df())