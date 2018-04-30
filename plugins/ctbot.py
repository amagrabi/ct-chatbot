from will.plugin import WillPlugin
from will.decorators import respond_to, periodic, hear, randomly, route, rendered_template, require_settings


class CTBotPlugin(WillPlugin):

    @respond_to('.*')
    def test123(self, message):
        if message.data.content == 'test123':
            self.reply(f"{message.data.content}? 456!")
        else:
            pass
