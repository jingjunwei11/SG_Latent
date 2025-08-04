# 🤟 A Two-Stage Sign Language Generation Framework with Self-Supervised Latent Representation Learning

This project aims to  generate sign language keypoints/videos from gloss inputs, facilitating communication a for the deaf and hard-of-hearing community.

## 📦 Dataset

This project uses the following two datasets:

- **PHOENIX14T**
- **How2Sign**
> You can download the datasets we have prepared from the links below:
> - PHOENIX14T: [Download Link](https://www.phoenix14t.com)
> - How2Sign: [Download Link](https://www.how2sign.com)

## 🛠️ Project Structure

```
sign-language-generation/
├── Configs/             
├── CVT/                  
├── Data/                
├── data_operate/        
├── German/             
├── __main__.py        
├── BiLSTM.py             
├── builders.py           
├── constants.py          
├── data.py           
├── decoders.py           
├── discriminator_Data.py 
├── dtw.py                
├── embeddings.py         
├── encoders.py          
├── helpers.py            
├── initialization.py   
├── loss.py             
├── model.py             
├── plot_videos.py      
├── pre_encoders.py       
├── requirements.txt     
├── search.py             
├── transformer_layers.py
├── tsn.py                
```

## 🚀 Installation & Usage

### 1. Install Dependencies 💻

```bash
pip install -r requirements.txt
```

### 2. Prepare Data 📁

Place the dataset under the `Data/` directory as follows:

```
Data/
├── train/
├── dev/
└── test/
```

### 3. Train the Model 🏋️‍♂️

```bash
python __main__.py CVT {config_path}
```

### 4. Run Inference 🧑‍💻

```bash
python __main__.py --ckpt {checkpoint_path} CVT_test {config_path}
```
### 5. Pre-trained Models 🎯

We provide pre-trained model weights, which you can directly download and use for inference 🔄.

- Pre-trained model link: [Download Link](https://example.com/model)

