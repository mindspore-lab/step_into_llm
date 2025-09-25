# 示例 1: 使用LayoutLMv2处理文档问答
from mindnlp.transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-layoutlmv2")

dqa_pipeline = pipeline("document-question-answering", model="hf-internal-testing/tiny-random-layoutlmv2",
                        tokenizer=tokenizer)

image_url = "https://hf.co/spaces/impira/docquery/resolve/2f6c96314dc84dfda62d40de9da55f2f5165d403/invoice.png"
question = "How many cats are there?"

outputs = dqa_pipeline(image=image_url, question=question, top_k=2)

print(outputs)


# 示例 2: 使用LayoutLM模型和自定义图像处理
from PIL import Image
import pytesseract
from mindnlp.transformers import pipeline, AutoTokenizer


def process_image_and_ocr(image_path):
    image = Image.open(image_path)
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    boxes = [
        [ocr_result["left"][i], ocr_result["top"][i], ocr_result["width"][i], ocr_result["height"][i]]
        for i in range(len(ocr_result["text"]))
        if ocr_result["text"][i].strip() 
    ]

    words = [word for word in ocr_result["text"] if word.strip()]
    
    return words, boxes, image.size

def normalize_boxes(boxes, image_width, image_height):
    normalized_boxes = []
    for box in boxes:
        left, top, width, height = box
        x1 = left / image_width * 1000
        y1 = top / image_height * 1000
        x2 = (left + width) / image_width * 1000
        y2 = (top + height) / image_height * 1000
        normalized_boxes.append((x1, y1, x2, y2))
    return normalized_boxes

image_path = "./path/to/your/invoice/image.png"
words, boxes, (image_width, image_height) = process_image_and_ocr(image_path)
normalized_boxes = normalize_boxes(boxes, image_width, image_height)
word_boxes = list(zip(words, normalized_boxes))

new_word_boxes = []
for item in word_boxes:
    text = item[0]
    box = item[1]   # 四个浮点数
    # 将box的四维坐标转换为整数（四舍五入）
    int_box = [round(x) for x in box]   # 四舍五入取整
    new_word_boxes.append([text, int_box])

tokenizer = AutoTokenizer.from_pretrained("tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa", revision="9977165")

dqa_pipeline = pipeline("document-question-answering", model="tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa",
                        tokenizer=tokenizer)

image = Image.open(image_path)
question = "What is the invoice number?"

outputs = dqa_pipeline(image=image, question=question, word_boxes=new_word_boxes, top_k=2)

print(outputs)
