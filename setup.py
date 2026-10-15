from setuptools import find_packages, setup

package_name = 'admittance'

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
    maintainer='gustavo',
    maintainer_email='g.hernan.diaz@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'adm_controller = admittance.adm_controller:main',
            'adm_controller_teleop = admittance.adm_controller_teleop:main',
            'tool_calibration = admittance.tool_params_estimation:main',
        ],
    },
)
