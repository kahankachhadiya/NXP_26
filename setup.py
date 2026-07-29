import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'b3rb_ros_line_follower'

standard_data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: ['*.onnx'],
    },
    data_files=standard_data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ishit',
    maintainer_email='ishit.choudhary@nxp.com',
    description='ROS 2 line follower application package for B3RB',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vectors = b3rb_ros_line_follower.b3rb_ros_edge_vectors:main',
            'runner = b3rb_ros_line_follower.b3rb_ros_line_follower:main',
            'detect = b3rb_ros_line_follower.b3rb_ros_object_recog:main',
            'qr_detect = b3rb_ros_line_follower.b3rb_ros_qr_detector:main',
        ],
    },
)
