import cv2
import numpy as np
import os
import configargparse

def convert_hdr(args):

    # img_dir = '/home/shindy/projects/hdr-4d/hdr-nsff/logs/_gopro/250913/bag_swing_3_1_i2_tf_hdr_250914_F00-36/render-train_250001/ldr'
    # save_dir = '/home/shindy/projects/hdr-4d/hdr-nsff/logs/_gopro/250913/bag_swing_3_1_i2_tf_hdr_250914_F00-36/render-train_250001/hdr_mertens'
    
    img_dir = args.src_dir
    save_dir = os.path.join(args.src_dir, '..', 'hdr_mertens')
    
    os.makedirs(save_dir, exist_ok=True)

    for i in range(36):
        print('step', i)
        img0_name = '{:05d}_0.png'.format(i)
        img1_name = '{:05d}_1.png'.format(i)
        img2_name = '{:05d}_2.png'.format(i)

        img0_path = os.path.join(img_dir, img0_name)
        img1_path = os.path.join(img_dir, img1_name)
        img2_path = os.path.join(img_dir, img2_name)

        img_list = [img0_path, img1_path, img2_path]
        images = [cv2.imread(f) for f in img_list]

        alignMTB = cv2.createAlignMTB()
        alignMTB.process(images, images)

        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(images)

        res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
        save_name = '{:05d}.png'.format(i)

        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, res_mertens_8bit)
    
    return


if __name__=='__main__':

    parser = configargparse.ArgumentParser()
    parser.add_argument("--src_dir")
    args = parser.parse_args()

    convert_hdr(args)