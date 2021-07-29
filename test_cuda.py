import torch
a = torch.cuda.is_available()
print(a)
ngpu= 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())
# import cv2
#
#
# # vid_file = 0 # Or video file path
# vid_file = "data/crowd_pose/test_video/test1.avi"
# print("Opening Camera " + str(vid_file))
# cap = cv2.VideoCapture(vid_file)
# ret, image = cap.read()
# print('cap', image)
