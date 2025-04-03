import cv2, tarfile, os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torchvision 


print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

mainpath = os.getcwd()
groundpath = os.getcwd() + '/extracted_tar/BSR/bench/data/groundTruth/'
# os.chdir(mainpath + '/extracted_tar/BSR/bench/data/images')
os.chdir(groundpath)
    


# files = os.listdir(os.getcwd())
# files.sort()
# print(files)
# for img in os.listdir(os.getcwd()):
#     if img is not None and img.endswith(".jpg"):
#         image = cv2.imread(img)
#         if image.shape == (481, 321, 3): 
#             image = np.transpose(image, (1,0,2))
#         image = image[:320, :480, :]
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         images.append(image)

# for i in images: print(i.shape)
# print(len(images))
# plt.imshow(images[0])

    # grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #display original image
    # cv2.imshow("original", image)

    #display grayscale image
    # cv2.imshow("gray scale", grayimg)
    
    #apply laplacian, sobelx, sobely, and canny 
    # laplacian = cv2.Laplacian(grayimg, cv2.CV_64F)
    # sobelX = cv2.Sobel(grayimg, cv2.CV_64F, 1, 0, ksize=3)
    # sobelY = cv2.Sobel(grayimg, cv2.CV_64F, 0, 1, ksize=3)

    # canny = cv2.Canny(grayimg, 100, 200)

    # plt.subplot(folderlen,3,c), plt.imshow(laplacian, cmap="gray"),plt.xticks([]), plt.yticks([]), c=c+1
    # plt.subplot(folderlen,3,c), plt.imshow(sobelX, cmap="gray"),plt.xticks([]), plt.yticks([]), c=c+1
    # plt.subplot(folderlen,3,c), plt.imshow(sobelY, cmap="gray"),plt.xticks([]), plt.yticks([]), c=c+1
    
    # edgeimg = cv2.magnitude(sobelX,sobelY)
    # edgeimg = (edgeimg / edgeimg.max()) * 255
    # edgeimg = edgeimg.astype(np.uint8)

    # c = c+1
    # plt.figure(), plt.imshow(canny, cmap="gray"),plt.xticks([]), plt.yticks([])

    # plt.figure(), plt.imshow(edgeimg, cmap="gray"),plt.xticks([]), plt.yticks([])


# plt.show()

    