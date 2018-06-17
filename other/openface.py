# http://cmusatyalab.github.io/openface/setup/
# sudo apt-get install cmake
# sudo pip3 install numpy pandas scipy scikit-image scikit-learn
# sudo pip3 install opencv-python-headless dlib
# smoke-test -> python3 -c "import cv2, dlib"

# install torch : http://torch.ch/docs/getting-started.html#_
# git clone https://github.com/torch/distro.git torch --recursive

# git clone https://github.com/cmusatyalab/openface.git
# sudo python3 openface/setup.py install
# openface/models/get-models.sh

# http://cmusatyalab.github.io/openface/demo-3-classifier/
# mkdir data-raw data-aligned data-features
# python3 ./openface/util/align-dlib.py data-raw/ align outerEyesAndNose data-aligned --size 96
# openface/batch-represent/main.lua -outDir data-features -data data-aligned
