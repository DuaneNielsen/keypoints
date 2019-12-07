import utils as u
import tests.common as com


def test_display_keypoints():
    x = com.bad_monkey()
    d = u.ResultsLogger('model_name', 'run_id')
    k = com.keypoints()
    d.display(x, blocking=True)


def test_color_map():
    cm = u.color_map()
    print(cm)


def test_resize():
    x = com.bad_monkey()
    x = u.resize2D(x[0], (512, 512))
    d = u.ResultsLogger('model_name', 'run_id')
    d.display(x, blocking=True)