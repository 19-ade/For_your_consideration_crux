# For_your_consideration_crux
This was an attempt to make a home-made chatbot for telegram . 



**chatbot.py**: this is the chatbot i made. I tried making the nueral network and everything on my own , and i did , however the problem was it took too much time to process and took a couple minutes to return hi to hello. It became quite useless. So i use chatterbot here .I trained the bot with the whatsapp chat exported from my whatsapp group i have with my closest homies; so as to mimic the way we chat among ourselves .  Trained with 39K lines of conservations (check whatsapp_skwad.py) , it is a bit arbitrary at times because its a bit unsorted . and i have yet to make more changes to it, but hey! "Tedha hain par mera hain ".
Heres the link to the bot:http://t.me/bitsian_bot


**hello_world.py**: thats the telegram bot . it also runs commands like /bonk, /woof to generate a cheems photo and a dog photo respectively. no / before words invokes the chatbot .


I havent included the exported chat data for privacy purposes; although with a few modifications any conversation data(whatsapp , telegram etc) can be used to train the bot.

**Disclaimer**:due to the absence of the whatsapp data the whatsapp_skwad.py wont work. in the chatbot script one can comment out the part where i cll the whatsapp_skwad function to run the chatbot. It will run ; will have basic greeting commands although it wont echo our whatsapp conversations.


**continous_action_trial**: this is a q-table for continous action space for the gym enviroment (continousMountainCar-v0). I have included q-table for mountain car and cartpole so you can see the changes i made overtime  to achieve this . Im still working on DQN so in the next inductions perhaps you will see the DQN versioon.



