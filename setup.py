from setuptools import setup,find_packages

setup(
    name               = 'chattiori_imagekit'
    , version          = '1.0'
    , license          = 'Apache License'
    , author           = "Chattiori"
    , packages         = find_packages()
    , install_requires = [
        'pillow'
    ]
)
