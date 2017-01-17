import sys
import logging
import glob
import os
import pandas as pd

SRC_DIR = '../../assets/data/unzipped'
DEST_DIR = '../../build/data/merged'
FILENAME_PREFIXES = ['acc', 'temp']

def detectSep(filename):
    f = open(filename,'r')
    firstLine = f.readline()
    f.close()
    if len(firstLine.split(';')) >= 2:
        return ';'
    else:
        return ','

def main():
    datasets = os.listdir(SRC_DIR)
    for dataset in datasets:
        instances = os.listdir(os.path.join(SRC_DIR, dataset))
        for prefix in FILENAME_PREFIXES:
            for instance in instances:
                filenames = glob.glob(os.path.join(
                    SRC_DIR,
                    dataset,
                    instance,
                    prefix + '_*.csv'
                ))
                print('parsing ' + dataset + '/' + instance + '...')
                for filename in filenames:
                    df = pd.read_csv(
                        filename,
                        sep=detectSep(filename),
                        header=None
                    )
                    destDir = os.path.join(DEST_DIR, dataset)
                    if not os.path.exists(destDir):
                        os.makedirs(destDir)
                    df.to_csv(
                        os.path.join(destDir, instance + '_' + prefix + '.csv'),
                        mode='a',
                        header=False,
                        index=False
                    )

main()
