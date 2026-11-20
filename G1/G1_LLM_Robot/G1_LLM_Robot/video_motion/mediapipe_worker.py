"""独立 MediaPipe 检测工作进程。"""
import argparse
import os
import sys
import traceback

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=float, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--space', choices=('world','image'), default='world')
    args = parser.parse_args(argv)
    try:
        from .mediapipe_backend import detect_frames
        from .mediapipe_backend import _write_payload

        last_percent = [-1]
        def report_progress(current, total):
            percent = int(round(100.0 * current / max(1, total)))
            if current == 1 or current == total or percent >= last_percent[0] + 2:
                last_percent[0] = percent
                print('G1_PROGRESS %d %d' % (int(current), int(total)), flush=True)

        def finish(times, points, visibility, source_fps):
            _write_payload(args.output, times, points, visibility, source_fps)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        result = detect_frames(
            args.video, args.fps, args.max_seconds, args.model_path,
            close_detector=False, result_handler=finish,
            exit_after_handler=True, progress=report_progress,
            landmark_space=args.space)
        if result is not None:
            finish(*result)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == '__main__':
    main()
