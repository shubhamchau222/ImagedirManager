from setuptools import find_packages, setup , find_packages 

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="shubhamchau222",
    description="A small package for handling messed up images in our local directory....",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shubhamchau222/ImagedirManager",
    author_email="shubhamchau78@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        'mtcnn==0.1.0',
        'tensorflow==2.3.1',
        'keras==2.4.3',
        'keras-vggface==0.6',
        'keras_applications==1.0.8',
        'PyYAML',
        'tqdm',
        'scikit-learn'      
    ]
)