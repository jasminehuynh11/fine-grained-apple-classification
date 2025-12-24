from setuptools import find_packages, setup

package_name = 'image_classification'

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
    'python3-opencv',
    'torch',
    'torchvision',
    'numpy',
    'Pillow'
    ],
    zip_safe=True,
    maintainer='Fine-Grained Fruit Classification Project',
    maintainer_email='maintainer@example.com',
    description='ROS2 package for fine-grained fruit classification using deep learning. Implements real-time inference node with robot action triggers.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	"inference = image_classification.inference:main"
        ],
    },
)
