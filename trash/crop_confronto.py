import torch
from torchvision.transforms import v2
import PIL
from utils.transform import GridTransform
import torchvision.transforms.functional as F

torch.manual_seed(42)

image = PIL.Image.open("/work/cvcs2024/VisionWise/train/image-real-00998569.png")
tensorizzatore = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
pacmaner = GridTransform()


# Define transformation parameters
angle = 30            # Rotation angle in degrees (counter-clockwise)
translate = (10, 20)  # Horizontal and vertical translations
scale = 1.0           # Scaling factor
shear = (5, 10)       # Shear angles (x-axis shear, y-axis shear)

x_normal = tensorizzatore(image)
x_pacman = pacmaner(tensorizzatore(image))


x_pacman = F.affine(x_pacman, angle, translate, scale, shear, fill=0)
x_normal = F.affine(x_normal, angle, translate, scale, shear, fill=0)

x_pacman = F.center_crop(x_pacman,[160,160])
x_normal = F.center_crop(x_normal,[160,160])

F.to_pil_image(x_normal.squeeze(0)).save('x_normal.png')
F.to_pil_image(x_pacman.squeeze(0)).save('x_pacman.png')