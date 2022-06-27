from models.PRM import PRM
from models.PRM import BN_ReLU_1x1
from models.Hourglass import Hourglass
from models.PyraNet import Pyranet
import torch
from utils import utils
from dataset.dataloaders import train_dataloader,val_dataloader
from utils.training import train,val
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

def main():
    model = Pyranet(256, 2, 1, 16).cuda()
    model.load_state_dict(torch.load(utils.weight_dir))

    optimizer = torch.optim.RMSprop(model.parameters(), lr=2.5e-4, alpha=0.99,
                                    eps=1e-8,
                                    weight_decay=0,
                                    momentum=0)
    criterion = torch.nn.MSELoss().cuda()

    acc=0
    for i in range(42, 100):
        train(i, train_dataloader, model, criterion, utils.mode, optimizer)
        results = val(i, val_dataloader, model, criterion, utils.mode)

        if results[0]['Acc'] > acc:
            acc = results[0]['Acc']
            torch.save(model.state_dict(), f'./pyrenet_weights{i}.h5')

if __name__=="__main__":
    main()