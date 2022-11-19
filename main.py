from models.blip_vqa import blip_vqa
from models.blip import blip_decoder
import torch
import pika
from models import ImageSQL, QuestionAnsSQL


connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))  # Connect to CloudAMQP
channel = connection.channel()  # start a channel
channel.queue_declare(queue='CaptionGen', durable=True)  # Declare a queue
channel.queue_declare(queue='AnswerGen', durable=True)


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


# def answerGen(questionID):
#     connection3 = pika.BlockingConnection(
#         pika.ConnectionParameters(host='localhost'))
#     channel3 = connection3.channel()
#     channel3.queue_declare(queue='AnswerGen', durable=True)
#
#     channel3.basic_publish(
#         exchange='',  # This publishes the thing to a default exchange
#         routing_key='AnswerGen',
#         body=questionID,
#         properties=pika.BasicProperties(
#             delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
#         ))
#     connection3.close()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def caption_callback(ch, method, properties, body):
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
        ImageSQL.updatecaption(image_id, caption[0])
        captionGen(image_id)
        # print('caption: ' + caption[0])


def answer_callback(ch, method, properties, body):
    image_size = 480
    message = body.decode()
    image = ImageSQL.findById(message[0])
    QA = QuestionAnsSQL.getDataByID(message[1])
    image_url = image[1]

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'

    model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    question = QA[2]

    with torch.no_grad():
        answer = model(image_url, question, train=False, inference='generate')
        QuestionAnsSQL.updateAnswers(message[1], answer[0])
        # answerGen(message[1])


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='CaptionGen', on_message_callback=caption_callback)
channel.basic_consume(queue='AnswerGen', on_message_callback=answer_callback)
channel.start_consuming()
