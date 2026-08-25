import cv2
import time
import argparse
from detector import ShuttlecockDetector
from tracker import TrajectoryTracker
from visualizer import TrajectoryVisualizer

def main():
    parser = argparse.ArgumentParser(description='Badminton Trajectory Tracker')
    parser.add_argument('--input', type=str, default='0',
                       help='Video file path or camera index (0 for webcam)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Display width')
    parser.add_argument('--height', type=int, default=720,
                       help='Display height')
    args = parser.parse_args()

    # Initialize components
    detector = ShuttlecockDetector()
    tracker = TrajectoryTracker(max_trajectory_length=50, hit_detection_threshold=30)
    visualizer = TrajectoryVisualizer()

    # Open video source
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"Error: Cannot open video source '{args.input}'")
        return

    # Set capture properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Get actual frame size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker.set_frame_size(frame_width, frame_height)

    print("Badminton Trajectory Tracker")
    print("=" * 50)
    print(f"Video source: {args.input}")
    print(f"Frame size: {frame_width}x{frame_height}")
    print("\nControls:")
    print("  'c' - Clear trajectory manually")
    print("  'q' - Quit")
    print("=" * 50)

    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame")
            break

        # Detect shuttlecock
        detected_point = detector.detect(frame)

        # Track trajectory and detect hits
        hit_detected = tracker.add_point(detected_point)

        if hit_detected:
            print("Hit detected! Clearing trajectory...")
            tracker.clear_trajectory()

        # Get current trajectory
        trajectory = tracker.get_trajectory()

        # Visualize
        frame = visualizer.draw(frame, trajectory, detected_point)
        frame = visualizer.draw_info(frame, len(trajectory), fps)

        # Calculate FPS
        fps_frame_count += 1
        if fps_frame_count >= 30:
            fps_end_time = time.time()
            fps = fps_frame_count / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            fps_frame_count = 0

        # Display
        cv2.imshow('Badminton Trajectory Tracker', frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Manually clearing trajectory...")
            tracker.clear_trajectory()

    cap.release()
    cv2.destroyAllWindows()
    print("\nTracker stopped.")

if __name__ == "__main__":
    main()
