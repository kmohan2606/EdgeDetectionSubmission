from preprocess import BSDS500_mat
import os
import hashlib
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import cv2
train_set  = BSDS500_mat(root='../extracted_tar/BSR/BSDS500/data', split='train')
test_set = BSDS500_mat(root='../extracted_tar/BSR/BSDS500/data', split='test')
val_set = BSDS500_mat(root='../extracted_tar/BSR/BSDS500/data', split='val')
test_loader   = DataLoader(test_set, batch_size=2, num_workers=4, drop_last=False, shuffle=False)
val_loader = DataLoader(val_set, batch_size=2, num_workers=4, drop_last=False, shuffle=False)
test_list = [os.path.split(i.rstrip())[1] for i in test_set.file_list]

print(len(val_set))
print(len(test_set))
print(len(test_loader))
print(len(test_list))

