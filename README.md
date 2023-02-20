# OCR Convolutional Recurrent Neural Network

## Installation

```bash
$ pip install -r requirements.txt
```

## Data prepare
```bash
$ trdg --output_dir /content/drive/MyDrive/data -c 2000 -t 4 -w 2 -f 64 -k 5 -rk -do 0 
$ python ocr_crnn/crnn/prepare.py \
--config ocr_crnn/configs/text_recognition.yml \
--dir /content/drive/MyDrive/data
```

## Train
```bash
$ python ocr_crnn/crnn/train.py \
--config ocr_crnn/configs/text_recognition.yml \
--save_dir ocr_exp1
```

## Demo
```bash
$ python captcha/crnn/predict.py  \
--config captcha/configs/text_recognition.yml  \
--weight captcha_exp1/0_best_model.h5 \
--images captcha/captcha_test  \
--post greedy
```