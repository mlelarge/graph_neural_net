import torch
#import pickle
import os
import time

class SuperPixDataset(torch.utils.data.Dataset):

    def __init__(self, name, main_data_dir):
        """
            Loading Superpixels datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = os.path.join(main_data_dir, 'superpixels/', name)
        if name == 'MNIST':
            self.test = torch.load(data_dir+'/mnist_test.pt')
            self.val = torch.load(data_dir+'/mnist_val.pt')
            self.train = torch.load(data_dir+'/mnist_train.pt')
        elif name == 'CIFAR10':
            self.test = torch.load(data_dir+'/cifar_test.pt')
            self.val = torch.load(data_dir+'/cifar_val.pt')
            self.train = torch.load(data_dir+'/cifar_train.pt')
            #self.test = torch.load(data_dir+'/cifar_val.pt')
            #self.val = torch.load(data_dir+'/cifar_val.pt')
            #self.train = torch.load(data_dir+'/cifar_val.pt')
        else:
            print('Only MNIST and CIFAR available')
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))