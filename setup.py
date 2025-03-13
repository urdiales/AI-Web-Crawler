from setuptools import setup, find_packages

setup(
    name="ai-web-crawler",
    version="1.0.0",
    author="urdiales",
    author_email="",
    description="A comprehensive web crawling and knowledge base tool for RAG systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/urdiales/AI-Web-Crawler",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
)