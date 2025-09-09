import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


__version__ = "0.0.0"

REPO_NAME = "video-action-recognition"
AUTHOR_USER_NAME = "HSAkash"
SRC_REPO = "src"
AUTHOR_EMAIL = "hemelakash472@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="""
    A small package for video action recognition using deep learning.
    Real-time video action recognition using CNN+RNN.
    """,
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "."},
    packages=setuptools.find_packages(where=".")
)