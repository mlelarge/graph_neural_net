"""
This code is used to download the data used to train and test our model
adapted from https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch
"""

import os
from six.moves import urllib
import zipfile
from pathlib import Path
ROOT_DIR = Path.home()
DATA_DIR = os.path.join(ROOT_DIR,'data/')
#raw_dir = os.path.join(os.getcwd(), 'data')


def download_url(url, folder, filename):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    print('Downloading', url)

    os.makedirs(folder, exist_ok=True)

    data = urllib.request.urlopen(url)
    path = os.path.join(folder, filename)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def download_benchmarks(raw_dir):
    url = 'https://www.dropbox.com/s/vjd6wy5nemg2gh6/benchmark_graphs.zip?dl=1'
    file_path = download_url(url, raw_dir, 'benchmark_graphs.zip')
    zipfile.ZipFile(file_path, 'r').extractall(raw_dir)
    os.unlink(file_path)


def download_QM9(raw_dir):
    urls = [('https://www.dropbox.com/sh/acvh0sqgnvra53d/AAAxhVewejSl7gVMACa1tBUda/QM9_test.p?dl=1', 'QM9_test.p'),
            ('https://www.dropbox.com/sh/acvh0sqgnvra53d/AAAOfEx-jGC6vvi43fh0tOq6a/QM9_val.p?dl=1', 'QM9_val.p'),
            ('https://www.dropbox.com/sh/acvh0sqgnvra53d/AADtx0EMRz5fhUNXaHFipkrza/QM9_train.p?dl=1', 'QM9_train.p')]
    data_dir = os.path.join(raw_dir, 'QM9')
    for url, filename in urls:
        _ = download_url(url, data_dir, filename)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_benchmarks(DATA_DIR)
    #download_QM9()


if __name__ == '__main__':
    main()

