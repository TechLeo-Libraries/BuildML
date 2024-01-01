import pathlib
from setuptools import setup
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
setup(
  name="leopy",
  version="0.0.1",
  description="A beginners hack to data-preprocessing.",
  long_description=README,
  long_description_content_type="text/markdown",
  author="leopy",
  author_email="workwithtechleo@gmail.com",
  license="MIT",
  packages=["leopy"],
  zip_safe=False
)