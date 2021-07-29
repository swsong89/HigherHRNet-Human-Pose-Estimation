import cv2
import os


def save_to_video(input_image_dir, output_video_dir, frame_rate):

    video_type = 'mp4'

    images = os.listdir(input_image_dir)  # '1.jpg'
    # 拿一张图片确认宽高
    img0_name = images[0].split('.')[0]
    img0 = cv2.imread(os.path.join(input_image_dir, images[0]))
    # print(img0)
    height, width, layers = img0.shape

    fourcc = None
    video_save_name = None
    if video_type == 'avi':
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # 用于avi格式的生成
        video_save_name = os.path.join(output_video_dir, img0_name + '.avi')
    elif video_type == 'mp4':
        # 视频保存初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_save_name = os.path.join(output_video_dir, img0_name + '.mp4')

    videowriter = cv2.VideoWriter(video_save_name, fourcc, frame_rate, (width, height))
    # 核心，保存的东西
    for f in images:
        img_name = os.path.join(input_image_dir, f)
        print("saving..." + img_name)
        img = cv2.imread(img_name)
        img = cv2.resize(img, (width, height))
        cv2.waitKey(100)
        videowriter.write(img)
    videowriter.release()
    # cv2.destroyAllWindows()
    print('Success save ' + video_save_name)
    pass


if __name__ == "__main__":
    # 图片变视频
    input_image_dir = 'data/crowd_pose/test_img'
    output_video_dir = 'data/crowd_pose/test_video'  # 输入视频保存位置以及视频名称
    save_to_video(input_image_dir, output_video_dir, 1)
