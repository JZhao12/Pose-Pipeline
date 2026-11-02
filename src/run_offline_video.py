import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--testcase", type=str, required=True, help="path to object")
parser.add_argument("--fps", type=int, default=30, help="output video frame rate")
args = parser.parse_args()

# 设置图片文件夹路径（相对项目根目录，不依赖当前工作目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

image_folder = os.path.join(project_root, "results", args.testcase, "pose_visualization")
output_video_path = os.path.join(project_root, "results", args.testcase, "pose_vis.mp4")

# 获取所有png图片并按文件名排序
images = sorted(f for f in os.listdir(image_folder) if f.lower().endswith(".png"))
if not images:
    raise Exception("No PNG images found in the folder.")

# 读取第一张图的尺寸作为视频尺寸
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_frame.shape
fps = args.fps  # 帧率

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以尝试 'XVID'
video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
if not video.isOpened():
    raise RuntimeError(f"无法创建视频 {output_video_path}（缺少 mp4v 编码器？）")

# 逐帧写入
for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    if frame is None:
        raise RuntimeError(f"读取失败：{image}")
    if frame.shape[:2] != (height, width):
        raise RuntimeError(f"{image} 尺寸 {frame.shape[1]}x{frame.shape[0]} 与首帧 {width}x{height} 不一致")
    video.write(frame)

video.release()
print("视频保存成功：", output_video_path)
