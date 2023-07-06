from aiogram import F, Router
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command
from aiogram.types.callback_query import CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram import Bot

import os
import logging

import kb
import text
import utils
from states import Gen


router = Router()


@router.message(Command("start"))
async def start_handler(msg: Message):
    await msg.answer(text.greet, reply_markup=kb.menu)


@router.message(Command("help"))
async def help_handler(msg: Message):
    await msg.answer(text.text_help)


@router.message(F.text == "Меню")
@router.message(F.text == "Выйти в меню")
@router.message(F.text == "◀️ Выйти в меню")
@router.message(Command("menu"))
async def menu(msg: Message):
    await msg.answer(text.menu, reply_markup=kb.menu)


@router.callback_query(F.data == "first")
async def input_image_CNN(callback: CallbackQuery, state: FSMContext):
    await state.set_state(Gen.CNN_prompt)
    await callback.message.edit_text(text.gen_image)
    await callback.message.answer(text.gen_exit, reply_markup=kb.exit_kb)


@router.callback_query(F.data == "second")
async def input_image_GAN(callback: CallbackQuery, state: FSMContext):
    await state.set_state(Gen.GAN_prompt)
    await callback.message.edit_text(text.gen_image)
    await callback.message.answer(text.gen_exit, reply_markup=kb.exit_kb)


@router.message(Gen.CNN_prompt)
async def generate_image_CNN(msg: Message, state: FSMContext, bot: Bot):
    if msg.photo:
        file_info = await bot.get_file(msg.photo[-1].file_id)
        downloaded_file = await bot.download_file(file_info.file_path)

        src = '{}.jpg'.format(msg.photo[-1].file_id)

        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file.getvalue())

        mesg = await msg.answer(text.gen_wait)

        logging.info('CNN')
        image = await utils.generate_image_first(src)

        if not image:
            return await mesg.edit_text(text.gen_error, reply_markup=kb.iexit_kb)

        photo = FSInputFile(image)

        await mesg.delete()
        await mesg.answer_photo(photo=photo, caption='Нравится?')

        os.remove(src)
    else:
        await msg.answer('Это не фото.')


@router.message(Gen.GAN_prompt)
async def generate_image_GAN(msg: Message, state: FSMContext, bot: Bot):
    if msg.photo:
        file_info = await bot.get_file(msg.photo[-1].file_id)
        downloaded_file = await bot.download_file(file_info.file_path)

        src = '{}.jpg'.format(msg.photo[-1].file_id)

        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file.getvalue())

        mesg = await msg.answer(text.gen_wait)

        logging.info('GAN')
        image = await utils.generate_image_second(src)

        if not image:
            return await mesg.edit_text(text.gen_error, reply_markup=kb.iexit_kb)

        photo = FSInputFile(image)

        await mesg.delete()
        await mesg.answer_photo(photo=photo, caption='Нравится?')

        os.remove(src)
    else:
        await msg.answer('Это не фото.')
