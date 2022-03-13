import setuptools

with open("README.md", "r", encoding="utf_8") as file:
    package_description = file.read()

setuptools.setup(
    name="audio_emotion-ApurvTA",
    version="0.1",
    keywords=["audio emotion", "sample package"],
    description="In this package a model is created and implemented to determine emotion of the speaker based on the audio",
    long_description=package_description,
    long_description_content_type="text/markdown",
    author="Apurv Master",
    author_email="apurv.master@tigeranalytics.com",
    url="https://github.com/Apurv-TA/human_emotion",
    project_urls={
        "Bug Tracker": "https://github.com/Apurv-TA/human_emotion/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6"
)
