from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="get-oblique",
    version="0.1.1",
    include_package_data=True,
    packages=find_packages(),
    package_data={
        "GET": ["ancestorTF_File/*.hdf5"],
    },
    url="https://github.com/maoqiangqiang/GET",
    install_requires=[
        "h5py==3.13.0",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scikit_learn==1.5.0",
        "torch>=2.0.0"
    ],
    long_description=description,
    long_description_content_type="text/markdown"
)