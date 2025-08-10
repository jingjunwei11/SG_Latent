# 🤟 A Two-Stage Sign Language Generation Framework with Self-Supervised Latent Representation Learning

This project aims to  generate sign language keypoints/videos from gloss inputs.

## 🚀 Installation & Usage

### 1. Install Dependencies 💻

```bash
pip install -r requirements.txt
```

### 2. Prepare Data 📦 

#### Datasets

We used two sign language datasets:

1. **PHOENIX14T Dataset**  
   You can download it from [RWTH-PHOENIX-2014-T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/).

2. **How2sign Dataset**  
   You can download it from [How2sign](https://how2sign.github.io/#download).

####  Data Preprocessing Steps

First, we need to use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract the 2D skeleton joints from the sign language data. 

Based on the extracted 2D skeleton data, we will use the inverse kinematics code from the [3DposeEstimator](https://github.com/gopeith/SignLanguageProcessing) project to convert it into 3D data.

After the data preprocessing is complete, you will need to convert the data into a format suitable for training and prepare three txt files that match the configuration parameters `src`, `trg`, and `files`.

##### `src` File

The `src` file contains the source sentences, with each line representing a new sentence. Each sentence is the original sign language text representation, stored in `.text` or `.gloss` format.

#### `trg` File

The `trg` file contains the skeleton data for each frame. The data between each frame is separated by spaces, and each sequence is separated by a newline character. If each frame contains 150 joints, make sure to set `trg_size` to 150 in the configuration file. This file should be in `.skels` format.

#### `files` File

The `files` file contains the name of each sequence, with one sequence per line. This file should be in `.files` format.

Place the dataset under the `Data/` directory as follows:

```
Data/
├── train/
├── dev/
└── test/
```

#### Notes

- Ensure that the correct `trg_size` is set in the configuration file.
- Each file must follow the format described above to ensure the order and integrity of the data.
  
### 3. Train the Model 🏋️‍♂️

To run, please start __main__.py with the parameters 'CVT' and the config path.

```bash
python __main__.py CVT {config_path}
```

### 4. Run Inference 🧑‍💻

```bash
python __main__.py --ckpt {checkpoint_path} CVT_test {config_path}
```
### 5. Pre-trained Models 🎯

We provide pre-trained model weights, which you can directly download and use for inference 🔄.

- Pre-trained model link: [Download Link](https://drive.google.com/drive/folders/1X5E0Q64WOpyIMKLmKmaYlSf_wFbEFtV0?usp=sharing)

