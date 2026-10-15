"""Videos play at the requested speed."""

import pytest

from evacusim.visualization.video_generator import frame_plan

TIMES = [i * 0.5 for i in range(1801)]  # 15 minutes of frames every 0.5 s


@pytest.mark.parametrize(
    ("speedup", "stride", "fps"),
    [
        (1.0, 1, 2.0),  # real time: every frame, 2 per second
        (10.0, 1, 20.0),  # every frame at the maximum rate
        (30.0, 3, 20.0),  # too fast for 20 fps: every third frame
        (25.0, 2, 25.0),  # a stride of 2 needs 25 fps for exactly 25x
    ],
)
def test_frame_plan_plays_at_the_requested_speed(speedup, stride, fps):
    got_stride, got_fps = frame_plan(TIMES, 20, speedup)
    assert (got_stride, got_fps) == (stride, pytest.approx(fps))
    # Simulated seconds shown per second of video equals the speedup.
    assert got_fps * got_stride * 0.5 == pytest.approx(speedup)


def test_frame_plan_without_times_assumes_half_second_frames():
    assert frame_plan([], 20, 10.0) == (1, 20.0)
