from telegram.ext import Updater, CallbackContext, CommandHandler, MessageHandler, Filters
import requests
import re
from chatbot import get_answers
from telegram import Update

chat_id = '1054545517'


def get_url():
    contents = requests.get('https://random.dog/woof.json').json()
    url = contents['url']
    return url


def get_image_url():
    allowed_extension = ['jpg', 'jpeg', 'png']
    file_extension = ''
    while file_extension not in allowed_extension:
        url = get_url()
        file_extension = re.search("([^.]*)$", url).group(1).lower()
    return url


def shibe():
    contents = requests.get("http://shibe.online/api/shibes?count=[1]").json()
    return contents


def bop(update, context):
    url = get_image_url()

    context.bot.send_photo(chat_id=update.effective_chat.id, photo=url)


def bonk(update, context):
    url = shibe()[0]
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=url)


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="hello")


def echo(update, context):
    questions = update.message.text
    fa = get_answers(questions)

    context.bot.send_message(chat_id=update.effective_chat.id, text=fa)


def hello(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(f'Hello {update.effective_user.first_name}')


def help(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text='welcome to skwad_bot!!!\na product of rudimentary efforts by a person with substandard intelligence\n1./bonk:generates a random doge photo\n'
                                  '2./woof generates a cute dog photo(helps with the ladies)\n'
                                  '3./start bot says hello back(practise command)\n'
                                  '4.no commands whatsoever: returns you the answer by the chatbot\n'
                                  'Disclaimer:chatbot is rudimentary trained by 38k lines of conversations between us\n'
                                  'I did not devide the conversations based on mood , timeline etc. so answers maybe random most times \n'
                                  'still a work in progress that. \n'
                                  '5./hello says hello with your name (simple greeting cause manners maketh bot)\n'
                                  'Will add more features\n'
                                  'Made entirely in Python , because Python is besht.')


def main():
    updater = Updater('1430600515:AAH0b3wa6tjfr66mHTYPV7EzkttJNY8YyOY', use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('woof', bop))
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & (~Filters.command), echo))
    dp.add_handler(CommandHandler('hello', hello))
    dp.add_handler(CommandHandler("bonk", bonk))
    dp.add_handler(CommandHandler("help", help))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
