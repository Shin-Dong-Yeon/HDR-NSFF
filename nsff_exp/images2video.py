from PIL import Image
import os
from glob import glob
import cv2
import argparse



def images2gif():
    source_dir = "/node_data/shindy/projects/hdr-4d/logs/lego-v2-mid-400_F01-31/render-slowmo_bt_path_350001/images"
    img_files = sorted(glob(os.path.join(source_dir, '*.jpg')))

    img_list = [Image.open(img) for img in img_files]

    # import pdb; pdb.set_trace()
    # img_list = []
    # for file in img_files:
    #     img = Image.open(file)
    #     img_list.append(img)
    
    img_list[0].save(source_dir + '/output.gif', save_all=True, append_images=img_list[1:], optimize=False, duration=100, loop = 0)  

def images2video1():
    source_dir = "/node_data/shindy/projects/hdr-4d/logs/lego-v2-mid-400_F01-31/render-slowmo_bt_path_350001/images"
    video_name = "video.avi"
    
    images = [img for img in os.listdir(source_dir) if (img.endswith(".png") or img.endswith(".jpg"))]
    frame = cv2.imread(os.path.join(source_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(source_dir, image)))

    cv2.destroyAllWindows()
    video.release()

def images2video(base_folder, start_frame, end_frame):

    image_folder = base_folder + "/images"
    output_folder = base_folder + "/video"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    splitted_base_folder = base_folder.split('/')

    video_name = output_folder + '/' + splitted_base_folder[-2] + '_' + splitted_base_folder[-1].split('_')[-1] + '.mp4'

    images = [img for img in glob(f"{image_folder}/*g")]
    images.sort()  # Sort the images by name

    # Define the codec and create VideoWriter object
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
    # video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'X264'), 30, (width, height))
    
    if start_frame is not None and end_frame is not None:
        images = images[start_frame:end_frame]

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    
    print("Images to Video transform done!")




if __name__=='__main__':
    # images2gif()
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="image directory")
    parser.add_argument("--start_frame", type=int, default=None, help='start frame')
    parser.add_argument("--end_frame", type=int, default=None, help='end frame')
    args = parser.parse_args()

    images2video(args.src_dir, args.start_frame, args.end_frame)