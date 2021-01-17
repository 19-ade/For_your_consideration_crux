from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from whatsapp_skwad import counter_lines_execution
from whatsapp_skwad import manipulations

my_bot = ChatBot(name='PyBot', read_only=True,
                 logic_adapters=['chatterbot.logic.MathematicalEvaluation',
                                 'chatterbot.logic.BestMatch'])

corpus_trainer = ChatterBotCorpusTrainer(my_bot)
corpus_trainer.train('chatterbot.corpus.english')
list_trainer = ListTrainer(my_bot)
small_talk = ['hi there!',
              'hi!',
              'how do you do?',
              'how are you?',
              'i\'m cool.',
              'fine, you?',
              'always cool.',
              'i\'m ok',
              'glad to hear that.',
              'i\'m fine',
              'glad to hear that.',
              'i feel awesome',
              'excellent, glad to hear that.',
              'not so good',
              'sorry to hear that.',
              'what\'s your name?',
              'Jarvis'
              ]
bits_talk = [
    'lite',
    'av pe kitna?',
    "Who's your creator?",
    "Anwesan"
    "Who's Anwesan?",
    "Snek",
    "Ghot",
    "Kaunsa dels lene ka plan hain?",
    "kuch bhi nahi",
    "joke batao bc",
    "Joke: You thought you will get IIT"
    "boomer"
    "anwesan"
]
s =manipulations("/home/anwesan/Downloads/WhatsApp Chat with Skwad.txt")
#print(s)

for i in (small_talk, bits_talk, s):
    list_trainer.train(i)
def get_answers(question):
    return str(my_bot.get_response(question))

'''done=False
print("ask me anything \n to end process type in done/Done")
while not done:

    q=input("question:")
    if (q==("done") or (q==("Done"))):
        done=True
        continue
    else :
        l=get_answers(q)
        

        print(l)'''
