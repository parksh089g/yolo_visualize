import cv2
import os
from tqdm import tqdm

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from utils.plots import colors'

def dataload(target_Label_Dir,target_Img_Dir):
    try:
        file_list_label = os.listdir(target_Label_Dir)
        file_list_img = os.listdir(target_Img_Dir)
    except:
        print('Directory is missing...')
    
    txt_list = []
    img_list = []

    print("Collecting Label and Image List...")

    for txt in tqdm(file_list_label):
        if 'classes.txt' in txt:
            continue
        if '.txt' in txt:
            txt_list.append(txt)
    
    for img in tqdm(file_list_img):
        if '.png' or '.jpg' in img:
            img_list.append(img)

    return txt_list, img_list

def box_label(img, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
    if label:
        tf = 1  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=1 / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1 / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    hide_labels=False
    #########################################################
    save_path = r'C:\Users\User\Desktop\visualize\output'           #저장될 곳
    target_Img_Dir = r"C:\Users\User\Desktop\visualize\images"      #이미지가 있는 곳
    target_Label_Dir = r"C:\Users\User\Desktop\visualize\labels"    #라벨이 있는 곳
    #########################################################
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    txt_list, img_list = dataload(target_Label_Dir,target_Img_Dir)
    names=['person','car']      # 클래스 인덱스
    print('start visualize...')
    for image,label in tqdm(zip(img_list,txt_list)):
        image_path=target_Img_Dir+'\\'+image
        label_path=target_Label_Dir+'\\'+label
        im0 = cv2.imread(image_path)
        h, w, channel =im0.shape
        f = open(label_path,'r')
        lines=f.readlines()
        for line in lines:
            label_info=line.split(' ')
            label_info=list(map(float,label_info))
            c = int(label_info[0])  # integer class
            labeling = None if hide_labels else names[c]
            x, y = label_info[1]*w-(label_info[3]*w)/2, label_info[2]*h-(label_info[4]*h)/2
            xyxy=[x,y,x+label_info[3]*w,y+label_info[4]*h]
            box_label(im0, xyxy, labeling, color=colors(c, True))
        # Save results (image with detections)
        cv2.imwrite(save_path+'\\'+image, im0)
    print("작업이 완료되었습니다.")