import os
from PIL import Image
from tqdm import tqdm

from segformer import SegFormer_Segmentation
from utils.utils_metrics import compute_mIoU, show_results
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from torch.utils.data import DataLoader,Dataset
import torch
import numpy as np
from utils.utils import cvtColor,resize_image,preprocess_input
import cv2
import torch.nn.functional as F
from concurrent import futures
'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''

class test_dataset(Dataset):
    def __init__(self,img_list):
        super(test_dataset, self).__init__()
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_name = self.img_list[index]
        image = Image.open(img_name)
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (512,512))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),axis=0)

        image1 = image_data[:, :, ::-1, :]
        image2 = image_data[:, :, :, ::-1]
        image3 = image_data[:, :, ::-1, ::-1]
        # image_data = np.concatenate((image_data, image1, image2, image3), axis=0)
        image_data = [image_data, image1, image2, image3]

        # images = torch.from_numpy(image_data)
        return img_name,image_data 


def seg_dataset_collate(batch):
    images      = []
    names       = []
    for img_name, image_data in batch:
        images.extend(image_data)
        names.append(img_name)
    images      = torch.from_numpy(np.array(np.concatenate(images,axis=0))).type(torch.FloatTensor)
    return images, names



def inference(dataset,model,pred_dir):
    model = model.cuda()
    images_save = []
    path_save = []
    # for _,(img,name) in enumerate(dataset):
    for _, (img, name) in tqdm(enumerate(dataset), desc="Processing dataset",total=len(dataset)):
        # if _ == 10:
        #     break
        with torch.no_grad():
            img = img.cuda()
            prs, out_connects, out_connect_d1s = model(img)
            for i in range(len(name)):
                pr = prs[i*4:(i+1)*4]
                out_connect = out_connects[i*4:(i+1)*4]
                out_connect_d1 = out_connect_d1s[i*4:(i+1)*4]
                # pr1 = pr1s[i*4:(i+1)*4]
            # ---------------------------------------------------#
                out_connect = torch.sigmoid(out_connect)
                out_connect_d1 = torch.sigmoid(out_connect_d1)
                out_connect_full = []
                out_connect = out_connect.data.cpu().numpy()
                out_connect_full.append(out_connect[0, ...])
                out_connect_full.append(out_connect[1, :, ::-1, :])
                out_connect_full.append(out_connect[2, :, :, ::-1])
                out_connect_full.append(out_connect[3, :, ::-1, ::-1])
                out_connect_full = np.asarray(out_connect_full).mean(axis=0)[np.newaxis, :, :, :]
                pred_connect = np.sum(out_connect_full, axis=1)
                pred_connect[pred_connect < 3] = 0  #3
                pred_connect[pred_connect >= 3] = 1

                out_connect_d1_full = []
                out_connect_d1 = out_connect_d1.data.cpu().numpy()
                out_connect_d1_full.append(out_connect_d1[0, ...])
                out_connect_d1_full.append(out_connect_d1[1, :, ::-1, :])
                out_connect_d1_full.append(out_connect_d1[2, :, :, ::-1])
                out_connect_d1_full.append(out_connect_d1[3, :, ::-1, ::-1])
                out_connect_d1_full = np.asarray(out_connect_d1_full).mean(axis=0)[np.newaxis, :, :, :]
                pred_connect_d1 = np.sum(out_connect_d1_full, axis=1)
                pred_connect_d1[pred_connect_d1 < 1.5] = 0
                pred_connect_d1[pred_connect_d1 >= 1.5] = 1
                
                
                pred = F.softmax(pr, dim=1).cpu().numpy()[:, 1, :, :]
                pred = np.expand_dims(pred, axis=1)
                pred_full = []
                pred_full.append(pred[0, ...])
                pred_full.append(pred[1, :, ::-1, :])
                pred_full.append(pred[2, :, :, ::-1])
                pred_full.append(pred[3, :, ::-1, ::-1])
                pred_full = np.asarray(pred_full).mean(axis=0)

                pred_full[pred_full >= 0.675] = 1
                pred_full[pred_full < 0.675] = 0


                # pred1 = F.softmax(pr1, dim=1).cpu().numpy()[:, 1, :, :]
                # pred1 = np.expand_dims(pred1, axis=1)
                # pred_full1 = []
                # pred_full1.append(pred1[0, ...])
                # pred_full1.append(pred1[1, :, ::-1, :])
                # pred_full1.append(pred1[2, :, :, ::-1])
                # pred_full1.append(pred1[3, :, ::-1, ::-1])
                # pred_full1 = np.asarray(pred_full1).mean(axis=0)
                # pred_full1[pred_full1 >= 0.55] = 1
                # pred_full1[pred_full1 < 0.55] = 0
                # with torch.no_grad():
                #     pred_full1 = torch.max_pool2d(torch.from_numpy(pred_full1).unsqueeze(0),kernel_size=3,stride=1,padding=1).squeeze(0).numpy()

                pr = pred_connect_d1 + pred_connect + pred_full

                pr[pr > 0] = 1

                pr = np.transpose(pr, (1, 2, 0))

                # pr = cv2.resize(pr, (512, 512), interpolation=cv2.INTER_LINEAR)
                # pr = pr.astype(int)
                # image = Image.fromarray(np.uint8(pr))
                images_save.append(pr)
                path_save.append(os.path.join(pred_dir, name[i].split('/')[-1].split('.')[0] + ".png"))
    return images_save,path_save

def save(img_path):
    img,path = img_path
    pr = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    pr = pr.astype(int)
    image = Image.fromarray(np.uint8(pr))
    image.save(path)
    return 


if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 2

    miou_out_path   = "chn6_b1"
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    name_classes = ["_background_","road"]
    # name_classes    = ["_background_","cat","dog"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    # test_txt = ['./model_data/train_val/test_mass.txt','./model_data/train_val/val_chn6.txt','./model_data/train_val/val_deepglobe.txt']
    test_txt = ['./model_data/train_val/val_chn6.txt']
    test_images = []
    test_labels = []
    for i in test_txt:
        with open(i,'r')as f:
            a = f.readlines()
            a = [i.replace('\n','') for i in a]
            for j in a:
                test_images.append(j.split(' ')[0])
            test_labels.extend(a)
    # image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        segformer = SegFormer_Segmentation()
        print("Load model done.")

        print("Get predict result.")
        # for image_path in tqdm(test_images):
        #     image       = Image.open(image_path)
        #     image       = segformer.get_miou_png(image)
        #     image.save(os.path.join(pred_dir, image_path.split('/')[-1].split('.')[0] + ".png"))

        dataset = DataLoader(test_dataset(test_images),num_workers=4,batch_size=4,shuffle=False,drop_last=False,collate_fn=seg_dataset_collate)
        images_save,path_save = inference(dataset,segformer.net,pred_dir)
        with futures.ThreadPoolExecutor(16) as executor:  # 实例化线程池
            res = executor.map(save, zip(images_save,path_save))  
        print("Get predict result done.")


    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, test_labels, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
