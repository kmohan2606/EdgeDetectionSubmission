import numpy as np
import cv2
import os.path
import matplotlib.pyplot as plt
import scipy.io
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from skimage.metrics import structural_similarity as ssim

output_path = os.path.join(os.getcwd() + '/RCF/output2/')
gt_path = os.path.join(os.getcwd() + '/extracted_tar/BSR/BSDS500/data/groundTruth/test/')

model_output = []
gt_mat = []

for i in os.listdir(output_path): 
    if '_ms' in i:
        image = cv2.imread(os.path.join(output_path + i))
        model_output.append(image)

for i in os.listdir(gt_path):
    gt_mat.append(i)

gt_mat.sort()

# for i in range(len(gt_mat)):
#     print(gt_mat[i] + " " + model_output[i])

gt_data = []


for g in gt_mat:
    gt = scipy.io.loadmat(os.path.join(gt_path,g))
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
    # if avg.shape==(481,321): avg = avg.T
    gt_data.append(avg.astype(np.uint8))

# for i in range(len(gt_mat)):
#     print(gt_data[i].shape == model_output[i].shape[:2])

model_output = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in model_output]
model_output = [cv2.bitwise_not(img) for img in model_output]

kernel = np.ones((3,3), np.uint8)
gt_data = [cv2.dilate(gt, kernel, iterations=1) for gt in gt_data]

model_output = [255 - img for img in model_output] 
# A couple comparisons
rand_idx = random.randint(0, len(gt_mat))

plt.subplot(1,2,1),plt.imshow(gt_data[rand_idx])
plt.subplot(1,2,2),plt.imshow(model_output[rand_idx])
plt.show()

rand_idx = random.randint(0, len(gt_mat))

plt.subplot(1,2,1),plt.imshow(gt_data[rand_idx])
plt.subplot(1,2,2),plt.imshow(model_output[rand_idx])
plt.show()

rand_idx = random.randint(0, len(gt_mat))

plt.subplot(1,2,1),plt.imshow(gt_data[rand_idx])
plt.subplot(1,2,2),plt.imshow(model_output[rand_idx])
plt.show()
rand_idx = random.randint(0, len(gt_mat))

plt.subplot(1,2,1),plt.imshow(gt_data[rand_idx])
plt.subplot(1,2,2),plt.imshow(model_output[rand_idx])
plt.show()

def binarize(image, threshold=128):
    return (image > threshold).astype(np.uint8)


precision_list, recall_list, f1_list, iou_list, ssim_list = [], [], [], [], []

for i in range(len(model_output)):
    pred_bin = binarize(model_output[i])
    gt_bin = binarize(gt_data[i])

    # Flatten for metric computation
    pred_flat = pred_bin.flatten()
    gt_flat = gt_bin.flatten()

    # Precision, Recall, F1-score
    precision = precision_score(gt_flat, pred_flat, zero_division=1)
    recall = recall_score(gt_flat, pred_flat, zero_division=1)
    f1 = f1_score(gt_flat, pred_flat, zero_division=1)

    # IoU (Jaccard Index)
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    iou = intersection / union if union > 0 else 0

    # Structural Similarity Index (SSIM)
    ssim_value = ssim(gt_bin, pred_bin, data_range=1)

    # Store results
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    iou_list.append(iou)
    ssim_list.append(ssim_value)

# Compute Average Metrics
print(f"Average Precision: {np.mean(precision_list):.4f}")
print(f"Average Recall: {np.mean(recall_list):.4f}")
print(f"Average F1-score: {np.mean(f1_list):.4f}")
print(f"Average IoU: {np.mean(iou_list):.4f}")
print(f"Average SSIM: {np.mean(ssim_list):.4f}")




