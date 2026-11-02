try:
    import pyrealsense2 as rs
except ImportError as e:
    raise SystemExit("缺少 pyrealsense2：pip install pyrealsense2==2.53.1.4623") from e
import numpy as np
import cv2
import os
import time
import argparse

# print("camera reset start")
# ctx = rs.context()
# devices = ctx.query_devices()
# for dev in devices:
#     dev.hardware_reset()
# print("camera reset done")

parser = argparse.ArgumentParser()
parser.add_argument("--obj_path", type=str, required=True, help="path to object")
args = parser.parse_args()

# 项目根目录（src 的上一级），避免依赖当前工作目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# image dir
rgb_dir = os.path.join(project_root, "offline", args.obj_path, "rgb")
depth_dir = os.path.join(project_root, "offline", args.obj_path, "depth")
mask_dir = os.path.join(project_root, "offline", args.obj_path, "masks")
cam_k_file = os.path.join(project_root, "offline", args.obj_path, "cam_K.txt")
depth_scale_file = os.path.join(project_root, "offline", args.obj_path, "depth_scale.txt")

if os.path.isdir(rgb_dir) and os.listdir(rgb_dir):
    raise SystemExit(f"{rgb_dir} 已有 {len(os.listdir(rgb_dir))} 帧；换一个 --obj_path 或先清空，避免新旧帧混在一起")
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# colour depth stream
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

with open(depth_scale_file, 'w') as f:
    f.write(str(depth_scale))
print(f"Depth scale ({depth_scale}) saved to: {depth_scale_file}")

# colour intrinsics
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()

K_matrix = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
])

# save intrinsic matrix
with open(cam_k_file, 'w') as f:
    f.write(f"{K_matrix[0,0]:.8f} {K_matrix[0,1]:.8f} {K_matrix[0,2]:.8f}\n")
    f.write(f"{K_matrix[1,0]:.8f} {K_matrix[1,1]:.8f} {K_matrix[1,2]:.8f}\n")
    f.write(f"{K_matrix[2,0]:.8f} {K_matrix[2,1]:.8f} {K_matrix[2,2]:.8f}")

print(f"Camera K matrix saved to: {cam_k_file}")

# align to colour
align_to = rs.stream.color
align = rs.align(align_to)

try:
    frame_count = 0
    recording = False
    
    while True:
        # wait
        frames = pipeline.wait_for_frames()
        
        # align
        aligned_frames = align.process(frames)
        
        # get data
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # data array
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # vis
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET)
        
        # status
        display_color = color_image.copy()
        status_text = "Recording" if recording else "Not Recording"
        status_color = (0, 0, 255) if recording else (0, 255, 0)
        
        cv2.putText(display_color, f"Status: {status_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(display_color, f"Frame: {frame_count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # show image
        cv2.imshow('RGB Image (Press SPACE to record, Q to quit)', display_color)
        cv2.imshow('Depth Image', depth_colormap)
        
        key = cv2.waitKey(1)
        
        if key == ord(' '): 
            recording = not recording
            print(f"Recording {'started' if recording else 'stopped'}")
            time.sleep(0.3)
        
        elif key == ord('q'):
            break
        
        if recording:
            rgb_filename = os.path.join(rgb_dir, f"{frame_count:06d}.png")
            depth_filename = os.path.join(depth_dir, f"{frame_count:06d}.png")
            
            cv2.imwrite(rgb_filename, color_image)
            cv2.imwrite(depth_filename, depth_image)
            
            print(f"Saved frame {frame_count}")
            frame_count += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Program ended")
