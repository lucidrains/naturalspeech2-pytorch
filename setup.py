from setuptools import setup, find_packages

exec(open('naturalspeech2_pytorch/version.py').read())

setup(
  name = 'naturalspeech2-pytorch',
  packages = find_packages(exclude=[]),
  version = __version__,
  license='MIT',
  description = 'Natural Speech 2 - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  include_package_data = True,
  url = 'https://github.com/lucidrains/naturalspeech2-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'latent diffusion',
    'speech synthesis'
  ],
  install_requires=[
    'accelerate',
    'audiolm-pytorch>=0.30.2',
    'beartype',
    'einops>=0.6.1',
    'ema-pytorch',
    'indic-num2words',
    'inflect',
    'local-attention',
    'num2words',
    'pyworld',
    'pydantic<2.0',
    'torch>=1.6',
    'tqdm',
    'vector-quantize-pytorch>=1.4.1'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
