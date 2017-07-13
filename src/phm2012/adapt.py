import sys
sys.path.insert(0, '..')
import glob
import os
import pandas as pd
from utils.utils import log, prepare_directory

SCOPE = 'phm2012'
SRC_DIR = '../../raw/phm2012'
DATASETS_TO_PARSE = ['Learning_set']
PREFIXES_TO_PARSE = ['acc']
INPUT_CSV_COLUMNS = ['hour', 'minute', 'second', 'us', 'x', 'y']
OUTPUT_CSV_COLUMNS = ['datetime', 'x', 'y']
DATETIME_FIELDS = [
    'year',
    'month',
    'day',
    'hour',
    'minute',
    'second',
    'us',
]

def detectSep(filename):
    f = open(filename,'r')
    firstLine = f.readline()
    f.close()
    if len(firstLine.split(';')) >= 2:
        return ';'
    else:
        return ','

def main():
    dest_dir = prepare_directory(os.path.join(
        '../../build/data', SCOPE, 'initialized'
    ))

    for dataset in DATASETS_TO_PARSE:
        instances = os.listdir(os.path.join(SRC_DIR, dataset))
        for prefix in PREFIXES_TO_PARSE:
            for instance in instances:
                filenames = sorted(glob.glob(os.path.join(
                    SRC_DIR,
                    dataset,
                    instance,
                    prefix + '_*.csv'
                )))

                i = 0
                length = len(filenames)
                for filename in filenames:
                    i = i + 1
                    header =  i is 1
                    mode = 'a' if i > 1 else 'w'
                    log('parsing ' + dataset + '/' + instance + '...' + str(i) + '/' + str(length))

                    df = pd.read_csv(
                        filename,
                        sep=detectSep(filename),
                        header=None,
                        names=INPUT_CSV_COLUMNS
                    )
                    df['year'] = 2017
                    df['month'] = 5
                    df['day'] = 26
                    df['datetime'] = pd.to_datetime(df[DATETIME_FIELDS])
                    df.to_csv(
                        os.path.join(dest_dir, '%s-%s-%s.csv' % (dataset, instance, prefix)),
                        mode=mode,
                        header=header,
                        index=False,
                        columns=OUTPUT_CSV_COLUMNS
                    )

main()
