# UTS_ImageProcessing_Spring_2021

# Setup
1. Clone the repository
2. Move into the directory `cd UTS_ImageProcessing_Spring_2021`
3. Make a virtual environment `python -m venv venv`
4. Activate the virtual environment, Linux: `source venv/bin/activate`
5. Install dependencies `pip install -r requirements.txt`
6. Download and place these pre-processed datasets into the same folder
    - [nist_labels_32x32.npy](https://drive.google.com/file/d/1IShRUWZBsqUVaXeCvRtw-vaygkl-fS4u/view?usp=sharing)
    - [nist_images_32x32.npy](https://drive.google.com/file/d/1G0I4A44psw2PsP_zzQBYikSoFSEmjz4f/view?usp=sharing)
7. Run the file `python main.py` with one argument (model type): `cnn` `ann` `tree` `knn` `lbp` `hog`
    - `cnn` for Convuluted Neural Network
    - `ann` for Artifical Neural Network
    - `tree` for Decision Tree Classifier
    - `knn` for K Nearest Neighbours
    - `lbp` for Local Binary Patterns
    - `hog` for Historgram of Oriented Gradient