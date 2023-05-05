# OCR Convolutional Recurrent Neural Network

## Installation

```bash
pip install -r requirements.txt
```

## Data prepare
```bash
trdg --output_dir /content/drive/MyDrive/data -c 2000 -t 4 -w 1 -f 64 -k 5 -rk -do 0 

python ocr_crnn/crnn/prepare.py \
--config ocr_crnn/configs/text_recognition.yml \
--dir /content/sample_data
```

## Train
```bash
python ocr_crnn/crnn/train.py \
--config ocr_crnn/configs/text_recognition.yml \
--save_dir /content/drive/MyDrive/ocr_exp1
```

## Demo
```bash
python ocr_crnn/crnn/predict.py  \
--config ocr_crnn/configs/text_recognition.yml  \
--weight /content/drive/MyDrive/ocr_exp1/10_0.2286_0.9502.h5 \
--images /content/ocr_crnn/example/images  \
--post greedy
```

## Results
| Ground truth 	| Prediction 	| Image 	|
|--------------	|------------	|-------	|
| detector 	    | detector 	    | ![1](example/images/detector_2.jpg "1") 	    |
| paraproctitis | paraproctitis | ![2](example/images/paraproctitis_7.jpg "2") 	|
| Lindgren      | Lindgren      | ![3](example/images/Lindgren_5.jpg "3")	    |
| twice-charged | twice-charged | ![4](example/images/twice-charged_9.jpg "4")	|
| Tindale       | Tindale       | ![5](example/images/Tindale_1.jpg "5")	    |
| rhotacistic   | rhotacistic   | ![6](example/images/rhotacistic_4.jpg "6")	|
| encephala     | encephala     | ![7](example/images/encephala_6.jpg "7") 	    |
| fissive       | fissive       | ![8](example/images/fissive_8.jpg "8") 	    |
| microphagous  | microphagous  | ![9](example/images/microphagous_0.jpg "9")	|
| propretorial  | propretorial  | ![10](example/images/propretorial_3.jpg "10") |
