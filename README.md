# OCR Convolutional Recurrent Neural Network

## Environment
```bash
$ conda env create -f ocr_crnn/ocr.yml
$ conda activate ocr
```

## Data prepare
```bash
$ trdg -c 10000 -t 4 -w 2 -f 64 -k 5 -rk -do 0
$ python ocr_crnn/crnn/prepare.py \
--config ocr_crnn/configs/ocr.yml \
--dir out
```

## Train
```bash
$ python ocr_crnn/crnn/train.py \
--config ocr_crnn/configs/ocr.yml \
--save_dir ocr_exp1
```

## Demo
```bash
$ python captcha/crnn/predict.py  \
--config captcha/configs/captcha.yml  \
--weight captcha_exp1/0_best_model.h5 \
--images captcha/captcha_test  \
--post greedy
```