import dlib
import cv2
import imageio
import torch
from PIL import Image 
from model import AgeGenderModel
from torchvision.transforms import transforms
from tqdm import tqdm
from retinaface.pre_trained_models import get_model


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# Load model age gender
model = AgeGenderModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load("outputs_final/model_epoch_50.pth")
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
model.to(device)

model_face = get_model("resnet50_2020-07-20", max_size=512, device='cuda:1')
model_face.eval()

# load the detector
detector = dlib.get_frontal_face_detector()
FPS = 30
# read the video
out_video = imageio.get_writer("demo_osaka_full.mp4", format='mp4', mode='I', fps=FPS)
video = imageio.get_reader("osaka.mp4")
for img in tqdm(video):
    if img is not None:
        # gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        
        # faces = detector(gray)
        
        annotation = model_face.predict_jsons(img)
        max_thresh = annotation[0]['score']
        bbox = annotation[0]['bbox']
        if max_thresh > 0.3:
            max_head_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
            
        
        # for face in faces:
        #     print(face)
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            
            x1_face = bbox[0]-20
            y1_face = bbox[1]-20
            x2_face = bbox[2]+20
            y2_face = bbox[3]+20
            if x1_face > 0 and y1_face > 0:
                # img_face = img[y1-50:y2+50, x1-50:x2+50]
                img_face = img[y1_face:y2_face, x1_face:x2_face]
                # print(y1-50,y2+50, x1-50,x2+50)
                imageio.imwrite('face.jpg', img_face)
                img_face = Image.fromarray(img_face)
                img_face = transform(img_face)
                # img_face.save('face_pil.jpg')
                img_face = torch.unsqueeze(img_face, 0)
                img_face = img_face.to(device)       

                gen_pred, age_cls_pred, age_reg_pred = model(img_face)
                _, gen_preds = torch.max(gen_pred, 1)
                _, age_cls_pred = torch.max(age_cls_pred, 1)

                if gen_preds.item() == 1:
                    text = f'M:{int(age_reg_pred.item()*100)}'
                    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255,0,0), thickness=4)
                    cv2.putText(img, text, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                elif gen_preds.item() == 0:
                    text = f'F:{int(age_reg_pred.item()*100)}'
                    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0,0,255), thickness=4)
                    cv2.putText(img, text, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        out_video.append_data(img)
out_video.close()
print('Done')
        
    
