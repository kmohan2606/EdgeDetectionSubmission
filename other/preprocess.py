import numpy as np
import cv2
import os.path
import scipy.io 
from torch.utils.data import Dataset


# dataset = os.getcwd() + '/extracted_tar/BSR/BSDS500/data/'
# print(dataset)


class BSDS500_mat(Dataset):
    def __init__(self, root='../extracted_tar/BSR/BSDS500/data/', split='test', transform=False):
        super(BSDS500_mat, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.image_path = os.path.join(self.root, f'images/{split}/')
            self.label_path = os.path.join(self.root, f'groundTruth/{split}/')
            self.image_list = os.listdir(os.path.join(self.root, f'images/{split}/'))
            self.label_list = os.listdir(os.path.join(self.root, f'groundTruth/{split}/'))
        elif self.split == 'test':
            self.image_path = os.path.join(self.root, f'images/{split}/')
            self.label_path = os.path.join(self.root, f'groundTruth/{split}/')
        else:
            raise ValueError('Not a recognized split type.')

        self.mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

        self.images = self.imgloader(self.image_path)
        self.labels = self.lblloader(self.label_path)



        self.file_list = [os.path.basename(f) for f in os.listdir(self.image_path) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if self.split == 'train':
            img, label = self.images[index], self.labels[index]
            label = label[np.newaxis, :, :]
        else:
            img = self.images[index]

        if self.split == 'train':
            return img, label
        else:
            return img
        
    def imgloader(self, path):
        out = []
        files = os.listdir(path)
        files.sort()
        for file in files:
            img_path = os.path.join(path, file)
            image = cv2.imread(img_path)
            if image is None: continue
            if image.shape == (481,321,3):
                image = np.transpose(image, (1,0,2))

            h,w,_ = image.shape
            c_h, c_w = h//2, w//2
            crop_size = 256
            tl_h = c_h - crop_size//2
            tl_w = c_w - crop_size//2

            image = image[tl_h:tl_h + crop_size, tl_w:tl_w + crop_size, :]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = (image - self.mean)
            image = image.transpose((2, 0, 1))  
            out.append(image)
        return out

    def lblloader(self, path):
        out = []
        files = os.listdir(path)
        files.sort()
        for g in files:
            gt = scipy.io.loadmat(os.path.join(path,g))
            gtd = gt['groundTruth']
            num_anno = len(gt['groundTruth'][0])
            avg = np.zeros(gtd[0][0]['Boundaries'][0,0].shape)
            for i in range(num_anno):
                label = gtd[0,i]['Boundaries'][0,0]
                label = np.where(label == 0,-1,1)
                avg += label
            avg = np.where((avg >= -2) & (avg <= -1), 2, avg)  
            avg = np.where((avg >= -1) & (avg != 2), 1, avg)  
            avg = np.where((avg != 1) & (avg != 2), 0, avg)
            if avg.shape==(481,321): avg = avg.T
            h,w = avg.shape
            c_h, c_w = h//2, w//2
            crop_size = 256
            tl_h = c_h - crop_size//2
            tl_w = c_w - crop_size//2

            avg = avg[tl_h:tl_h + crop_size, tl_w:tl_w + crop_size]
            out.append(avg)
        return out
