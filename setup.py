from setuptools import find_packages, setup
from glob import glob

package_name = 'project'
otherfiles = [
    ('share/' + package_name + '/launch', glob('launch/*')),
    ('share/' + package_name + '/urdf',   glob('urdf/*')),
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]+otherfiles,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuxin-lin',
    maintainer_email='yuxin-lin@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'h1_test = project.h1_test:main',
            'h1_ball = project.h1_ball:main',
            'balldemo = project.balldemo:main',
            'h1_ball_hit = project.h1_ball_hit:main',
            'g1_test = project.g1_test:main',
            'g1_ball_hit = project.g1_ball_hit:main',
            'g1_test_v = project.g1_test_v:main',
            'g1_test_p= project.g1_test_p:main',
            'g1_ball_v = project.g1_ball_v:main',
            'g1_ball_move = project.g1_ball_move:main',
            'g1_ball_rel = project.g1_ball_rel:main',
            'g1_paddle = project.g1_paddle:main',
            'g1_normal= project.g1_normal:main',
            'g1_normal_e=project.g1_normal_e:main',
        ],
    },
)
