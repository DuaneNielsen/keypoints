from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='keypoints',
      version='0.1',
      description='Unsupervised keypoint detection',
      author='DNielsen',
      author_email='duane.nielsen.rocks@gmail.com',
      url='https://github.com/duanenielsen/keypoints',
      install_requires=requirements,
     )