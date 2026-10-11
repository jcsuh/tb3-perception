from setuptools import find_packages, setup

package_name = 'tb3p_eval'

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
    maintainer='jcsuh',
    maintainer_email='gitSuh2762@gmail.com',
    description='tb3p 평가 도구: 캘리브레이션 채점, 정답 궤적 기록',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'calib_error = tb3p_eval.calib_error:main',
            'record_tum = tb3p_eval.record_tum:main',
        ],
    },
)
