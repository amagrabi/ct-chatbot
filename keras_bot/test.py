import unittest

# from keras_bot.KerasBot import KerasBot
# bot = KerasBot()

from keras_bot.KerasBot import KerasBot
from keras.models import load_model
existing_model = load_model('models/saved_funtalk_model.h5')
bot = KerasBot(model=existing_model)

# python -m unittest /Users/evi/code/ct-chatbot/keras_bot/test.py


class MyBotTest(unittest.TestCase):

    def test_reply(self):

        reply = bot.answer_to_text('is it going to rain?')
        self.assertEqual(reply, ' no , it is not . ')


if __name__ == '__main__':
    unittest.main()
