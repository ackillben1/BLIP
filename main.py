from models.blip_vqa import blip_vqa
from models.blip import blip_decoder
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_feature_extractor
from models.blip_itm import blip_itm
import pika, os, logging, time
from models import ImageSQL

logging.basicConfig()
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))  # Connect to CloudAMQP
channel = connection.channel()  # start a channel
channel.queue_declare(queue='CaptionGen', durable=True)  # Declare a queue

connection3 = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
channel3 = connection3.channel()
channel3.queue_declare(queue='AnswerGen', durable=True)


def captionGen(imageID):
    connection2 = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel2 = connection2.channel()
    channel2.queue_declare(queue='QuestGen', durable=True)

    channel2.basic_publish(
        exchange='',  # This publishes the thing to a default exchange
        routing_key='QuestGen',
        body=imageID,
        properties=pika.BasicProperties(
            delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
        ))
    connection2.close()


# def visualQuestionAnswer(ch, method, properties, body):
#     connection3 = pika.BlockingConnection(
#         pika.ConnectionParameters(host='localhost'))
#     channel3 = connection3.channel()
#     channel3.queue_declare(queue='AnswerGen', durable=True)
#
#     message = body.decode() + ' received'
#     channel3.basic_publish(
#         exchange='',  # This publishes the thing to a default exchange
#         routing_key='AnswerGen',
#         body=message,
#         properties=pika.BasicProperties(
#             delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
#         ))
#     connection3.close()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def callback(ch, method, properties, body):
    image_size = 384
    ID = body.decode()
    image = ImageSQL.findById(ID)
    image_url = image[1]

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # beam search
        caption = model.generate(image_url, sample=False, num_beams=3, max_length=20, min_length=5)
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        image_id = ImageSQL.getImageID(image[0])
        ImageSQL.updatecaption(image_id, caption)
        captionGen(image_id)
        # print('caption: ' + caption[0])

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='CaptionGen', on_message_callback=callback)

def load_demo_image(image_size, device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def image_caption():
    image_size = 384
    image = load_demo_image(image_size=image_size, device=device)

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        print('caption: ' + caption[0])


def VQA():
    image_size = 480
    image = load_demo_image(image_size=image_size, device=device)

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'

    model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    question = 'where is the woman sitting?'

    with torch.no_grad():
        answer = model(image, question, train=False, inference='generate')
        print('answer: ' + answer[0])


def feature_extraction():
    image_size = 224
    image = load_demo_image(image_size=image_size, device=device)

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'

    model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    caption = 'a woman sitting on the beach with a dog'

    multimodal_feature = model(image, caption, mode='multimodal')[0, 0]
    image_feature = model(image, caption, mode='image')[0, 0]
    text_feature = model(image, caption, mode='text')[0, 0]


def image_text_matching():

    image_size = 384
    image = load_demo_image(image_size=image_size, device=device)

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'

    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device='cpu')

    caption = 'a woman sitting on the beach with a dog'

    print('text: %s' % caption)

    itm_output = model(image, caption, match_head='itm')
    itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]
    print('The image and text is matched with a probability of %.4f' % itm_score)

    itc_score = model(image, caption, match_head='itc')
    print('The image feature and text feature has a cosine similarity of %.4f' % itc_score)