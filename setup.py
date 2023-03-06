try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name="deep_embeddings",
    version="0.0.1",
    author="Florian P. Mahner",
    author_email="florian.mahner@gmail.com",
    license="LICENSE",
    long_description=open("docs/README.md").read(),
    packages=find_packages(),
    scripts=["deep_embeddings/main.py", "deep_embeddings/tripletize.py"],
    python_requires=">=3.9",
)
