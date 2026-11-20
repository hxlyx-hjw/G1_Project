"""每帧记录 G1 骨盆运动学参数（位置/速度/加速度），支持 CSV/JSON 输出与图谱绘制。

记录字段：time, x, y, z（位置）、vx, vy, vz（速度）、ax, ay, az（加速度）。

独立使用：

    from motion_logger import MotionLogger
    logger = MotionLogger("motion.csv")
    logger.record(t, x, y, z, vx, vy, vz, ax, ay, az)
    logger.save()               # 输出 CSV
    logger.plot("motion.png")   # 绘制位置/速度/加速度图谱

通过 RobotSession 集成（每个动作结束自动保存 CSV 并绘制 PNG）：

    with RobotSession(motion_log_path="motion.csv") as bot:
        bot.walk_for(vx=0.5, seconds=3)
    # 动作结束自动保存 motion.csv 与 motion.png

命令行直接跑一段行走并记录 + 绘图：

    python motion_logger.py --output motion.csv --seconds 3 --vx 0.5 --plot
"""

from pathlib import Path
import csv
import json

# 记录字段顺序（CSV 表头 / 绘图都按这个来）
FIELDS = ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']


class MotionLogger:
    """记录骨盆运动学序列，支持 CSV / JSON 输出与图谱绘制。

    参数
        path          输出文件路径；None 表示保存时用默认 motion_log.csv
        fmt           输出格式，'csv' 或 'json'；None 时按 path 后缀推断
        step_interval 每隔多少条记录才真正存一条（1 = 每条都存，对应 500Hz 物理步）
    """

    def __init__(self, path=None, fmt=None, step_interval=1):
        self.path = Path(path) if path else None
        if fmt is None:
            suffix = (self.path.suffix.lstrip('.') if self.path and self.path.suffix else 'csv')
            fmt = suffix.lower()
        self.fmt = 'json' if fmt == 'json' else 'csv'
        self.step_interval = max(1, int(step_interval))
        self._count = 0
        self.records = []

    def record(self, t, x, y, z, vx=0.0, vy=0.0, vz=0.0, ax=0.0, ay=0.0, az=0.0):
        """追加一帧。t 为仿真时间（秒），其余均为 SI 单位（m、m/s、m/s²）。"""
        self._count += 1
        if self._count % self.step_interval:
            return
        self.records.append((float(t), float(x), float(y), float(z),
                             float(vx), float(vy), float(vz),
                             float(ax), float(ay), float(az)))

    def clear(self):
        """清空记录（用于每个动作开始时重置）。"""
        self.records = []
        self._count = 0

    def save(self, path=None):
        """把记录写入文件，返回输出路径。"""
        target = Path(path) if path else self.path
        if target is None:
            target = Path('motion_log.csv')
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.fmt == 'json':
            rows = [dict(zip(FIELDS, r)) for r in self.records]
            target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            with target.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(FIELDS)
                writer.writerows(self.records)
        return str(target)

    def plot(self, path=None, show=False):
        """绘制位置/速度/加速度随时间变化的图谱，返回 PNG 路径。

        path  None 时用「记录文件同名 .png」；show=True 时额外弹出窗口（需图形环境）。
        """
        import numpy as np
        import matplotlib
        matplotlib.use('Agg', force=True)          # 无头安全，先保存 PNG
        import matplotlib.pyplot as plt

        data = np.asarray(self.records, dtype=float)
        if data.ndim != 2 or data.shape[0] < 2:
            return None

        t = data[:, 0]
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

        for ax, cols, ylabel, title in (
            (axes[0], (1, 2, 3), 'Position (m)', '位置'),
            (axes[1], (4, 5, 6), 'Velocity (m/s)', '速度'),
            (axes[2], (7, 8, 9), 'Acceleration (m/s²)', '加速度'),
        ):
            for c, label in zip(cols, ('x', 'y', 'z')):
                ax.plot(t, data[:, c], linewidth=0.9, label=label)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10, loc='left')
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=3, fontsize=8)

        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout()

        target = Path(path) if path else (
            self.path.with_suffix('.png') if self.path else Path('motion_log.png'))
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=120)
        if show:
            plt.show()
        plt.close(fig)
        return str(target)

    def last(self):
        """最后一条记录，无记录返回 None。"""
        return self.records[-1] if self.records else None

    def __len__(self):
        return len(self.records)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='跑一段行走并记录每帧骨盆运动学参数')
    parser.add_argument('--output', default='motion_log.csv', help='输出文件（csv/json）')
    parser.add_argument('--seconds', type=float, default=3.0, help='行走时长（仿真秒）')
    parser.add_argument('--vx', type=float, default=0.5, help='前进速度 m/s')
    parser.add_argument('--interval', type=int, default=1, help='每 N 步记录一次（1=每帧）')
    parser.add_argument('--plot', action='store_true', help='同时绘制图谱 PNG')
    args = parser.parse_args()

    from robot_bridge import RobotSession

    with RobotSession(motion_log_path=args.output, motion_log_interval=args.interval) as bot:
        bot.walk_for(vx=args.vx, seconds=args.seconds)

    print('位置日志已输出:', args.output)
    print('图谱已由 RobotSession 在动作结束时自动绘制（同名 .png）')
