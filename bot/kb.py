from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup


algorithms = [
    [InlineKeyboardButton(text="CNN", callback_data="first"),
     InlineKeyboardButton(text="CycleGan", callback_data="second")]]
menu = InlineKeyboardMarkup(inline_keyboard=algorithms)
exit_kb = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="◀️ Выйти в меню")]], resize_keyboard=True)
iexit_kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="◀️ Выйти в меню", callback_data="menu")]])
