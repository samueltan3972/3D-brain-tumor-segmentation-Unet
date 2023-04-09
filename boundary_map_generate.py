from medpy.io import load, save

def generate_boundary_map():
    gt_img, gt_header = load(
        "D:/fyp/BRATS2015_Training/BRATS2015_Training/HGG/brats_2013_pat0001_1/VSD.Brain_3more.XX.O.OT.54517/VSD.Brain_3more.XX.O.OT.54517.mha")



if __name__ == '__main__':
    generate_boundary_map()
