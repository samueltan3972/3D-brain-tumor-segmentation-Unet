from medpy.io import save

from metrics import IOU, DICE
import torch


def evaluate(net, testloader, testset, device='cpu'):
    # set to evaluation mode
    net.eval()

    # init performance metrics
    iou = IOU()
    dice = DICE()

    for i, (inputs, targets) in enumerate(testloader):
        # transfer to the GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # perform prediction (no need to compute gradient)
        with torch.no_grad():
            outputs = net(inputs)

            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            test_iou = iou(preds, targets)
            test_dice = dice(preds, targets)

        # Save output file
        # output_path = 'D:/fyp/fyp2-result/unet-bpb/' + str(i) + '.mha'
        # save(preds[0][0].numpy(), output_path, testset.get_img_header(i))

        print(f'Iter {i + 1:5d}/{len(testloader)} IOU = {test_iou :.2f}')
        print(f'Iter {i + 1:5d}/{len(testloader)} DICE = {test_dice :.2f}')

    print('IOU = {:.2f}'.format(test_iou))
    print('DICE = {:.2f}'.format(test_dice))
