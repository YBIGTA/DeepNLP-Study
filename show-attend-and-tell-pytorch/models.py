from torchvision.models import vgg
from torch import nn

## Encoder

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = vgg.vgg19_bn(pretrained=True)
        self.convs = list(self.original_model.children())[0]
        self.layers = nn.Sequential(*list(self.convs)[:-1])
        
    def forward(self, x):
        x = self.layers(x)
        return x




