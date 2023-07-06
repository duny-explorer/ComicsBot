from aiogram.fsm.state import StatesGroup, State


class Gen(StatesGroup):
    CNN_prompt = State()
    GAN_prompt = State()
