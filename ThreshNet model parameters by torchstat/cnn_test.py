from torchstat import stat
import torchvision.models as models

model = models.resnet18()
stat(model, (3, 224, 224))