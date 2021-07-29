import cv2
import os


def save_to_img(input_video_path, output_image_dir):
    vc = cv2.VideoCapture(input_video_path)
    c = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:

        if rval:
            print(os.path.join(output_image_dir, str(c)) + '.jpg')
            cv2.imwrite(os.path.join(output_image_dir, str(c)) + '.jpg', frame)
            c = c + 1
            cv2.waitKey(1)

        rval, frame = vc.read() # 接着读下一张
    vc.release()


if __name__ == "__main__":
    # 视频变图片
    input_video_path = 'data/crowd_pose/test_video/100000.avi'  # 输入视频保存位置以及视频名称
    output_image_dir = 'data/crowd_pose/test1'

    save_to_img(input_video_path, output_image_dir)
