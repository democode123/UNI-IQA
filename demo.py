import torch
from torchvision import transforms
from model import UNI_IQA
from main import parse_config
from Transformers import AdaptiveResize
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    AdaptiveResize(768),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

config = parse_config()

model = UNI_IQA(config)
model = torch.nn.DataParallel(model).cuda()

ckpt = ''

checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint)

model.eval() 

image = './demo/test_ni.jpg'

image = Image.open(image)
image = test_transform(image)
image = torch.unsqueeze(image, dim=0)
image = image.to(device)
with torch.no_grad():
    score1, score2 = model(image)

score1 = score1.cpu().item()
score2 = score2.cpu().item()
print(score1,score2)
