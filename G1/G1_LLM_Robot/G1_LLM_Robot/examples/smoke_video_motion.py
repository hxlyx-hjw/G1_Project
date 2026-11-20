"""下载公开示例图生成静态人物短片，验证真实检测到仿真的链路。"""

# 支持从项目根目录、IDE及脚本所在目录启动。
from pathlib import Path as _ProjectPath
import sys as _project_sys
import os as _project_os
_project_root = _ProjectPath(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_project_root))
_project_os.chdir(_project_root)

from pathlib import Path
import urllib.request
import cv2
from video_motion.pose import extract
from video_motion.retarget import play
from robot_bridge import RobotSession


def main():
    root = Path('motions/smoke')
    root.mkdir(parents=True,exist_ok=True)
    image = root/'full_body.jpg'
    if not image.exists():
        urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-assets/pose.jpg',image)
    frame = cv2.imread(str(image))
    if frame is None:
        raise RuntimeError('测试图片无法解码')
    h,w = frame.shape[:2]
    video = root/'person.avi'
    writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(w,h))
    if not writer.isOpened():
        raise RuntimeError('MJPG 编码器不可用')
    try:
        for _ in range(10):
            writer.write(frame)
    finally:
        writer.release()
    motion = extract(video,root/'person.npz',fps=5,max_seconds=1)
    with RobotSession(verbose=False) as bot:
        result = play(bot,motion)
        print(result)
        if not result['completed']:
            raise RuntimeError('真人骨架回放摔倒')
    print('真实图像短片检测、动作保存、G1回放通过；此用例仅验证链路，不验证动态模仿质量。')


if __name__ == '__main__':
    main()
