from setuptools import find_packages, setup

package_name = 'hierarchical_qa'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='safaa',
    maintainer_email='sofi29315961@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "vlm = hierarchical_qa.vlm:main",
            "display = hierarchical_qa.display:main"
        ],
    },
)
