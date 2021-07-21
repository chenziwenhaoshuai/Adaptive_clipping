import cv2
import os
import numpy as np
label_path = "G:/dronecar/val/labels/"
fileList = os.listdir(label_path)
img_path = "G:/dronecar/val/images/"
imgList = os.listdir(img_path)
save_path = "G:/dronecar/val/large/"
label_save_path = "G:/dronecar/val/largelabel/"
n = 0
def adaptive_clipping(img,out_shape,save_path,label_save_path,img_name,labels):#自适应裁切
    Iw = img.shape[1]#图像的宽
    Ih = img.shape[0]#图像的高
    Fw = out_shape[0]#输出的宽
    Fh = out_shape[1]#输出的高
    Nw = int(Iw/Fw) if Iw/Fw==1 else int(Iw/Fw)+1#横向裁切的图片个数
    Nh = int(Ih/Fh) if Ih/Fh==1 else int(Ih/Fh)+1#纵向裁切的图片个数
    Sw = int(Fw-((Fw-(Iw%Fw))/(Nw-1))) if Nw!=1 else int(Fw-((Fw-(Iw%Fw))/(Nw)))#横向移动的步长
    Sh = int(Fh-((Fh-(Ih%Fh))/(Nh-1))) if Nh!=1 else int(Fh-((Fh-(Ih%Fh))/(Nh)))#纵向移动的步长
    #output_img = np.ones((Fh, Fw, 3), np.uint8)
    num = 1
    for h in range(Nh):
        for w in range(Nw):
            y3 = h*Sh
            y4 = h*Sh+Fh
            x3 = w*Sw
            x4 = w*Sw+Fw#当前滑动窗口的坐标（x3，y3）（x4，y4）
            output_img = img[y3:y4,x3:x4]


            #cv2.imwrite(save_path+img_name+'_'+str(num)+'.jpg',output_img)
            save(x3,x4,y3,y4,Iw,Ih,Fw,Fh,label_save_path,img_name,labels,num,save_path,output_img)
            #label_mapping(x3, x4, y3, y4, Iw, Ih, Fw, Fh, label_save_path, img_name, labels, num)
            num+=1
            #cv2.imshow("213",output_img)
            #cv2.waitKey(0)
            #print(1)
def save(x3,x4,y3,y4,Iw,Ih,Fw,Fh,label_save_path,img_name,labels,num,save_path,output_img):
    labels = labels.split('\n')
    for label in labels[:-1]:
        label = label.split(' ')
        cls = label[0]
        x_c = float(label[1]) * Iw  # 反归一化xywh
        y_c = float(label[2]) * Ih
        w_1 = float(label[3]) * Iw
        h_1 = float(label[4]) * Ih
        x1 = x_c - 0.5 * w_1  # 将xywh转换为xyxy
        y1 = y_c - 0.5 * h_1
        x2 = x_c + 0.5 * w_1
        y2 = y_c + 0.5 * h_1
        if iou(x1, x2, x3, x4, y1, y2, y3, y4):  # 判断标签框与滑动窗口是否有交集
            cv2.imwrite(save_path + img_name + '_' + str(num) + '.jpg', output_img)
            xtop = max(x1, x3)  # 目标框的坐标
            xbot = min(x2, x4)
            ytop = max(y1, y3)
            ybot = min(y2, y4)
            x_o, y_o, w_o, h_o = xyxy2xywh(xtop, xbot, ytop, ybot, Fw, Fh, x3, y3)  # 转换为输出图的坐标格式并归一化
            with open(label_save_path + img_name + '_' + str(num) + '.txt', 'a') as l:
                l.write(str(cls) + " " + str(x_o) + " " + str(y_o) + " " + str(w_o) + " " + str(h_o) + '\n')
def label_mapping(x3,x4,y3,y4,Iw,Ih,Fw,Fh,label_save_path,img_name,labels,num):

    labels = labels.split('\n')
    for label in labels[:-1]:
        label = label.split(' ')
        cls = label[0]
        x_c = float(label[1])*Iw#反归一化xywh
        y_c = float(label[2])*Ih
        w_1 = float(label[3])*Iw
        h_1 = float(label[4])*Ih
        x1 = x_c - 0.5 * w_1#将xywh转换为xyxy
        y1 = y_c - 0.5 * h_1
        x2 = x_c + 0.5 * w_1
        y2 = y_c + 0.5 * h_1
        if iou(x1,x2,x3,x4,y1,y2,y3,y4):#判断标签框与滑动窗口是否有交集
            xtop = max(x1,x3)#目标框的坐标
            xbot = min(x2,x4)
            ytop = max(y1,y3)
            ybot = min(y2,y4)
            x_o,y_o,w_o,h_o = xyxy2xywh(xtop,xbot,ytop,ybot,Fw,Fh,x3,y3)#转换为输出图的坐标格式并归一化
            with open(label_save_path+img_name+'_'+str(num)+'.txt','a') as l:
                l.write(str(cls)+" "+str(x_o)+" "+str(y_o)+" "+str(w_o)+" "+str(h_o)+'\n')
def xyxy2xywh(xtop,xbot,ytop,ybot,w,h,x3,y3):
    x_o = (((xbot-xtop)/2)+xtop-x3)/w
    y_o = (((ybot-ytop)/2)+ytop-y3)/h
    w_o = (xbot-xtop)/w
    h_o = (ybot - ytop) / h
    return x_o,y_o,w_o,h_o


def iou(x1,x2,x3,x4,y1,y2,y3,y4):
    Xmax = max(x1, x3)
    Ymax = max(y1, y3)
    M = (Xmax, Ymax)
    Xmin = min(x2, x4)
    Ymin = min(y2, y4)
    N = (Xmin, Ymin)
    if M[0] < N[0] and M[1] < N[1]:
        return True
    else:
        return False


if __name__ == '__main__':

     for line in imgList:
         if n <8757:
            img = cv2.imread(img_path+line)
            #cv2.imshow("o",img)
            #img = np.zeros((3000,4000,3),dtype=np.uint8)
            img_name = line.split('.')[0]
            n+=1
            # adaptive_clipping(img,[900,900],save_path,label_save_path,img_name)
            try:
                with open(label_path+img_name+'.txt','r') as labels:
                    labels = labels.read()
                    print("convert {} No:{}".format(img_name,n))
                    adaptive_clipping(img, [640,640], save_path, label_save_path, img_name,labels)
                    #print(labels)
            except:
                print("label {} not find".format(img_name))






















        #s.add(img.shape[0])
#         if img.shape[0] == 1080:
#             n1080+=1
#             cv2.imwrite(save_path+line,img)
#         if img.shape[0] == 2160:
#             n2160+=1
#             cv2.imwrite(save_path + line, img)
#         if img.shape[0] == 480:
#             n480+=1
#         if img.shape[0] == 576:
#             n576+=1
#         if img.shape[0] == 1440:
#             n1440+=1
#             cv2.imwrite(save_path + line, img)
#         if img.shape[0] == 1536:
#             n1536+=1
#             cv2.imwrite(save_path + line, img)
#         if img.shape[0] == 405:
#             n405+=1
#         #print(img.shape)
#         n+=1
#         print(n)
# print("n1080 = {} n2160 = {} n480 = {} n576 = {} n1440 = {} n1536 = {} n405 = {}".format(n1080,n2160,n480,n576,n1440,n1536,n405))
# #n1080 = 604 n2160 = 3558 n480 = 44 n576 = 38 n1440 = 18 n1536 = 9 n405 = 3329
# #n1080 = 596 n2160 = 2831 n480 = 44 n576 = 38 n1440 = 14 n1536 = 6 n405 = 5228
#