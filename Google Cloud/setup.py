# Required to download and install Keras and it required dependencies 

#Setup parameteres for Google Cloud ML Engine
setup(name='trainer',
verion='0.1',
packages=find_packages(),
description='Example to run keras on gcloud ml-engine',
author='Aleksey Konoiko',
author_email='lesha.konoiko@gmail.com',
license='MIT',
install_required=[
    'keras',
    'h5py',
],
zip_safe=False)