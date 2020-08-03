from model.TC import TrajGRU10Class as TCModel
from model.EC import ECModel as ECModel
from model.DF import TrajGRUDeform as DFModel
from model.FC import TrajGRU5Class as FCModel
from model import *
from torch.utils.data import DataLoader


if __name__ == '__main__':
    if cfg.singleModel:
        model = TCModel().to(cfg.device)
        checkpoint = torch.load(cfg.checkpointTC, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        loader = DataLoader(TestLoader(), batch_size=1, shuffle=False)
        print('Start generating...')
        model.eval()
        with torch.no_grad():
            for batch, (data, path) in enumerate(loader):
                data = Variable(data).float().to(cfg.device).permute((1, 0, 2, 3, 4))
                output = model(data).permute((1, 0, 2, 3, 4))
                saveOutputT(output.data.cpu().numpy()[0], path)
                progress_bar(batch, len(loader))
    else:
        ECM = ECModel().to(cfg.device)
        ECM_= ECModel().to(cfg.device)
        TCM = TCModel().to(cfg.device)
        DFM = DFModel().to(cfg.device)
        FCM = FCModel().to(cfg.device)
        TCM_= TCModel().to(cfg.device)

        checkpoint = torch.load(cfg.checkpointEC, map_location='cpu')
        ECM.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(cfg.checkpointEC_, map_location='cpu')
        ECM_.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(cfg.checkpointTC, map_location='cpu')
        TCM.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(cfg.checkpointDF, map_location='cpu')
        DFM.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(cfg.checkpointFC, map_location='cpu')
        FCM.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(cfg.checkpointTC_, map_location='cpu')
        TCM_.load_state_dict(checkpoint['state_dict'])

        loader = DataLoader(TestLoader(), batch_size=1, shuffle=False)
        print('Start generating...')
        ECM.eval()
        ECM_.eval()
        TCM.eval()
        DFM.eval()
        FCM.eval()
        TCM_.eval()

        with torch.no_grad():
            for batch, (data, path) in enumerate(loader):
                data = Variable(data).float().to(cfg.device).permute((1, 0, 2, 3, 4))
                ECoutput = ECM(data).permute((1, 0, 2, 3, 4))
                print('model1 done')
                ECoutput_= ECM_(data).permute((1, 0, 2, 3, 4))
                print('model2 done')
                TCoutput = TCM(data).permute((1, 0, 2, 3, 4))
                print('model3 done')
                DFoutput = DFM(data).permute((1, 0, 2, 3, 4))
                print('model4 done')
                FCoutput = FCM(data).permute((1, 0, 2, 3, 4))
                print('model5 done')
                TCoutput_= TCM_(data).permute((1, 0, 2, 3, 4))
                print('model6 done')
                output = [ECoutput.data.cpu().numpy()[0],ECoutput_.data.cpu().numpy()[0],TCoutput.data.cpu().numpy()[0],
                          DFoutput.data.cpu().numpy()[0],FCoutput.data.cpu().numpy()[0],TCoutput_.data.cpu().numpy()[0]]
                saveOutput(output, path)
                progress_bar(batch, len(loader))

