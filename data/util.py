import numpy as np

def temporal_indices(frame_idx, fps, num_frames, time_span) -> np.ndarray:
    if num_frames == 1:
        return np.array([0])
    max_frames = int(time_span * fps) + 1
    assert num_frames <= max_frames, (
        f"Requested {num_frames} samples over {time_span}s "
        f"but video at {fps} FPS only supports {max_frames}"
    )

    dt = time_span / (num_frames - 1)

    # time of current frame
    t0 = frame_idx / fps

    times = [t0 - i * dt for i in range(num_frames)]
    indices = [int(round(t * fps)) for t in times]

    assert len(set(indices)) == num_frames, "Duplicate frame indices generated. Consider increasing time_span or decreasing num_frames."

    return np.array(indices[::-1])  # chronological order