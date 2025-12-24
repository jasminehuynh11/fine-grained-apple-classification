from setuptools import find_packages, setup

package_name = 'capture_topic'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools',
    'rclpy',
    'sensor_msgs',
    'cv_bridge',
    'python3-opencv'],
    zip_safe=True,
    maintainer='Fine-Grained Fruit Classification Project',
    maintainer_email='maintainer@example.com',
    description='ROS2 package for capturing and saving images from robot camera topics for dataset collection.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
           "capture_subscriber = capture_topic.capture_sub:main",
        ],
    },
)
