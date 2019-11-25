import utils as u
import tests.common as com


def test_display_keypoints():
    x = com.bad_monkey()
    d = u.ResultsLogger(title='debug_test', logfile='test.log')
    k = com.keypoints()
    d.display(x, x, x, k, blocking=True)

def test_color_map():
    cm = u.color_map()
    print(cm)