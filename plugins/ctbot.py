import random

from will.plugin import WillPlugin
from will.decorators import respond_to, periodic, hear, randomly, route, rendered_template, require_settings


class CTBotPlugin(WillPlugin):

    @respond_to(r'.*')
    def answer(self, message):

        message_str = message.data.content

        if message_str.startswith('decide '):
            options = message_str.split(' ')[1:]
            if len(options) >= 2:
                answer = str(random.choice(options)) + '!'
            else:
                answer = 'If I need to make a choice for you, then there have to be multiple options!'
            self.reply(answer)

        elif message_str == 'test':
            self.reply('test worked')

        elif message_str == 'message':
            self.reply(str(message.data.content))

        elif message_str:
            self.reply('do not get it o_O')

        else:
            pass