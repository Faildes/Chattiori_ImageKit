from setuptools import setup,find_packages

setup(
    name               = 'chattiori_imagekit'
    , version          = '1.0'
    , license          = 'Apache License'
    , author           = "Chattiori"
    , packages         = find_packages('src')
    , package_dir      = {'': 'src'}
    , url              = 'https://github.com/Faildes/Chattiori_ImageKit'
    , keywords         = 'diffusers stable-diffusion pil image'
    , install_requires = [
        'pillow'
    ]
)
