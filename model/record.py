from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from Dataload.Chaoyang import CHAOYANG
from torchvision.transforms import transforms
from Dataload.patchCamelyon import Camelyonpatch
from model.models import SimSiam
import torch
from torch.autograd import Variable

dataset = 'chaoyang'
r = 0.0
if dataset == 'chaoyang':
    backbone = 'densenet121'
    num_classes = 4
    batch_size = 32
    r = 0.0
    test_dataset = CHAOYANG(root="/home/hangang/datasets/chaoyang-data",
                            json_name="test.json",
                            train=False,
                            transform1=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
                            )
else:
    backbone = 'densenet121'
    num_classes = 2
    batch_size = 128
    test_dataset = Camelyonpatch(train=False,
                                 transform1=transforms.Compose([transforms.ToTensor()]))

model_net1 = SimSiam(0.99, backbone, num_classes)
# model_net2 = model_net1
model_net1.cuda()
sd = torch.load(f'./log/best{dataset}_{r}_sd.pth')
model_net1.load_state_dict(sd)
test_load = DataLoader(test_dataset, batch_size, shuffle=False)


def evaluate(val_load, model_net1_kd):
    print("Evaluate......")
    model_net1_kd.eval()
    # model_net2_kd.eval()
    sum = 0
    num = 0
    for x, y, index in tqdm(val_load):
        with torch.no_grad():
            x = Variable(x[0]).cuda()
            y = Variable(y).cuda()
            y_hat1 = model_net1_kd.forward_test(x)
            # y_hat2 = model_net2_kd.forward_test(x)
            # y_hat = (y_hat2 + y_hat1) / 2.0
            y_hat = y_hat1
            sum += y_hat.shape[0]
            num += torch.sum(y_hat.argmax(dim=1).type(y.dtype) == y)

    test_acc = float(num / sum)

    return test_acc


test_acc = evaluate(test_load, model_net1,
                    # model_net2,
                    )
print(test_acc)