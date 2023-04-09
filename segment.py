import torch

# import dataset
# from evaluate import evaluate
from unet_model import UNet
from medpy.io import load, save
# from torch.utils.data import DataLoader
import numpy as np
import sys


def performSegment(image_path, output_path, model_path, use_bpb):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model = UNet(1, 1, use_BPB=use_bpb)

    print('Start Segmentation ---------------------------------------- ')

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    model.to(device)

    image_data, image_header = load(image_path)

    image_data = np.expand_dims(image_data, axis=0)
    image_data = np.expand_dims(image_data, axis=0)
    data = torch.as_tensor(image_data, dtype=torch.float)

    with torch.no_grad():
        outputs = torch.sigmoid(model(data))
        outputs = (outputs > 0.5).float()
        outputs = outputs[0][0].numpy()

    save(outputs, output_path, image_header)
    # print(outputs.shape)
    print('Done')
    print('The result have been saved in: ', output_path)
    #######################################################

    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # test = torch.rand(1, 1, 240, 240, 155).to(device)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(pytorch_total_params)
    # print(summary(model, torch.zeros((1, 1, 240, 240, 155)), show_input=True))
    # np.set_printoptions(threshold=np.inf)
    # print(np.unique(image_data, return_counts=True))

    # iou = IOU()
    # gt_img, gt_header = load("D:/fyp/BRATS2015_Training/BRATS2015_Training/HGG/brats_2013_pat0001_1/VSD.Brain_3more.XX.O.OT.54517/VSD.Brain_3more.XX.O.OT.54517.mha")
    #
    # gt_img = np.expand_dims(gt_img, axis=0)
    # gt_img = np.clip(gt_img, 0, 1)
    # gt_img = gt_img.astype(np.float32)
    # gt_img_tensor = torch.as_tensor(gt_img, dtype=torch.float)
    # print(np.unique(gt_img, return_counts=True))
    # train_iou = iou(data, gt_img_tensor)
    #
    # print(train_iou)


# def evaluate_testset(model_path):
#     path = 'D:/fyp/fyp2 - selected testing set/dataset_list.csv'
#     brats2015_dataset = dataset.Brats2015ImageDataset(path, 'D:/fyp/fyp2 - selected testing set/')
#     brats2015_testloader = DataLoader(brats2015_dataset, batch_size=1, shuffle=True)  # , num_workers=2)
#
#     device = 'cpu'
#
#     model = UNet(1, 1, use_BPB=False)
#
#     checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#
#     evaluate(model, brats2015_testloader, brats2015_dataset, device=device)

def test():
    print("test")


if __name__ == "__main__":
    # test on U-Net
    # evaluate_testset("C:/Users/lalala/PycharmProjects/Unet/0.00001/unet_model_epoch_00007.pt")

    # test on BPB
    # evaluate_testset("C:/Users/lalala/PycharmProjects/Unet/models/unet_bpb/unet_bpb_model_epoch_00005.pt")

    # argv 1: Input image path
    # argv 2: Output image path
    # argv 3: Model path
    ########

    assert sys.argv[1], "Please enter input image path. Usage: segment.py input.mha output.mha model.pt"
    assert sys.argv[2], "Please enter output image path. Usage: segment.py input.mha output.mha model.pt"
    assert sys.argv[3], "Please enter model path. Usage: segment.py input.mha output.mha model.pt"

    use_BPB = False

    try:
        if sys.argv[4] == 1:
            use_BPB = True
    except:
        use_BPB = False

    performSegment(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        use_BPB)
