import logging

from models import ModelFastNeuralStyle, CycleGan

model_CNN = ModelFastNeuralStyle()
model_GAN = CycleGan()


async def generate_image_first(prompt):
    try:
        model_CNN.result(prompt)

        return prompt
    except Exception as e:
        logging.error(e)
        return e
    else:
        return prompt


async def generate_image_second(prompt):
    try:
        model_GAN.result(prompt)

        return prompt
    except Exception as e:
        logging.error(e)
        return e
    else:
        return prompt
