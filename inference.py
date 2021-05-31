import dlib
import cv2
import imageio
import torch
from PIL import Image 
from model import AgeGenderModel
from torchvision.transforms import transforms
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# Load model age gender
model = AgeGenderModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load("outputs/model_epoch_5.pth")
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
model.to(device)

# load the detector
detector = dlib.get_frontal_face_detector()
FPS = 30
# read the video
out_video = imageio.get_writer("demo.mp4", format='mp4', mode='I', fps=FPS)
video = imageio.get_reader("/home/cybercore/haimd/test.mp4")
for img in tqdm(video):
    if img is not None:
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            
            img_face = img[y1-80:y2+80, x1-80:x2+80]
            imageio.imwrite('face.jpg', img_face)
            img_face = Image.fromarray(img_face)
            img_face = transform(img_face)
            img_face = torch.unsqueeze(img_face, 0)
            img_face = img_face.to(device)       

            gen_pred, age_cls_pred, age_reg_pred = model(img_face)
            _, gen_preds = torch.max(gen_pred, 1)
            _, age_cls_pred = torch.max(age_cls_pred, 1)

            cv2.rectangle(img=img, pt1=(x1-80, y1-80), pt2=(x2+80, y2+80), color=(0,255,0), thickness=4)
            if gen_preds.item() == 1:
                text = f'Male:{int(age_reg_pred.item()*100)}'
            elif gen_preds.item() == 0:
                text = f'Female:{int(age_reg_pred.item()*100)}'
            cv2.putText(img, text, org=(x1-80, y1-80), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        out_video.append_data(img)
out_video.close()
print('Done')
        
    
