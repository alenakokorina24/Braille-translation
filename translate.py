import numpy as np
import pickle
import argparse
import cv2
import sys
import PIL
import torch
import random
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


input_image = sys.argv[1]


# A file containing information to match Russian letters to network output. 
f_letters = open("letters.pkl", "rb")
letters_dict = pickle.load(f_letters)
f_letters.close()

# This is the information to distinguish numbers that go after a number sign.
numbers_dict = { 'а': 1, 'б': 2, 'ц': 3, 'д': 4, 'е': 5, 'ф': 6, 'г': 7, 'х': 8, 'и': 9, 'ж': 0 }

num_classes = 45


# Setting up model.

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=500, box_score_thresh=0.8)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load('state_dict_model.pt', map_location=torch.device('cpu')))
model.eval()


colors = []

for i in range(num_classes):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    colors.append((r, g, b))

colors_classes = dict(zip([i for i in range(num_classes)], colors))


def get_symbol(value):
    for symbol, class_number in letters_dict.items():
        if value == class_number:
            return symbol
    return 'null'


def draw_boxes(boxes, labels, scores, image):
    """
    Drawing boxes on the original image based on the network predictions. 
    """
    model.rpn_score_thresh = 0.8
    image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_GRAY2BGR)
    for i, box in enumerate(boxes):
        class_number = labels[i]
        color = colors_classes.get(class_number)
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, str(class_number) + ' ' + str(round(scores[i], 2)), (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
        
    return PIL.Image.fromarray((image * 255).astype(np.uint8))


def to_text(pred_boxes, labels):
    """
    Translating text in Braille to text in Russian based on network predictions.
    Args:
        pred_boxes (float array): predicted coordinates for letters,
        labels (int array): predicted letters,
    """
    text = ""
    pred_boxes = pred_boxes.tolist()
    labels = labels.tolist()
    
    is_number = False
    is_capital = False
    
    # Network returns predictions in random order so we need to
    # sort boxes and group them (one group corresponds to one line of text).
    
    # Sorting predicted boxes by y1 (each box consists of [x1, y1, x2, y2]).
    sorted_by_y = sorted(list(zip(pred_boxes, labels)), key = lambda x: x[0][1])
    
    lines = []
    line = []
    is_new_line = False
    prev_box_y1 = sorted_by_y[0][0][1]
    
    for sym in sorted_by_y:
        
        curr_box_y1 = sym[0][1]
        
        # If distance between two symbols (vertically) is greater than approximate 
        # symbol height, this means we have a new line.
        if abs(prev_box_y1 - curr_box_y1) > 50:
            is_new_line = True
            
        prev_box_y1 = curr_box_y1
        
        if is_new_line and len(line) > 0:
            lines.append(line)
            line = []
            is_new_line = False
            
        line.append(sym)
        
    # All symbols have been grouped into "lines".
    lines.append(line)
        
    for l in lines:
        
        # Sorting symbols in each line by x1 to recreate letters order.
        sorted_by_x = sorted(l, key = lambda x: x[0][0])
        
        prev_box_x1 = sorted_by_x[0][0][0]
        
        # Putting the text together.
        for box, label in sorted_by_x:
            
            curr_box_x1 = box[0]
            curr_box_x2 = box[2]
        
            symbol = get_symbol(label)
            
            if symbol == 'цифровой символ':
                is_number = True
            
            if symbol == 'знак заглавной буквы':
                is_capital = True
        
            if abs(prev_box_x1 - curr_box_x1) > 80: 
                if is_number == True:
                    is_number = False
                text = text + " "
            
            if is_capital and symbol != 'знак заглавной буквы':
                symbol = symbol.upper()
            elif is_number and symbol != 'цифровой символ':
                symbol = str(numbers_dict.get(symbol))
            
            is_capital = False
        
            if symbol != 'цифровой символ' and symbol != 'знак заглавной буквы' and symbol != 'зачеркивание':
                text = text + symbol
        
            prev_box_x1 = curr_box_x1
            
        text += '\n'
        
    return text


image = cv2.imread(input_image, 0) / 255
image = cv2.resize(image, (1500, 2000))
original_image = image 

images = [image]
images = list(torch.from_numpy(image).float().reshape((1, 2000, 1500)) for image in images)
output = model(images)[0]
            
pred_boxes = output["boxes"].detach().cpu().numpy()
labels = output["labels"].detach().cpu().numpy()
scores = output["scores"].detach().cpu().numpy()
        
output_image = draw_boxes(pred_boxes, labels, scores, image)
output_text = to_text(pred_boxes, labels)

output_image.save(input_image[:-4] + '_boxes' + '.png')

f = open(input_image[:-4] + '_translated' + '.txt', "w")
f.write(output_text)
f.close()



