import unittest

from keras_bot.KerasBot import KerasBot
bot = KerasBot()

# python -m unittest /Users/evi/code/ct-chatbot/keras_bot/test.py


class MyBotTest(unittest.TestCase):

    def test_reply(self):

        reply = bot.answer_to_text('is it going to rain?')
        self.assertEqual(reply, ' no , it is not . ')


if __name__ == '__main__':
    unittest.main()
