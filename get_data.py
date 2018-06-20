"""

Get and clean HipChat data as a pandas DataFrame.

"""

import pandas as pd
from pathlib import Path
import re

FILEPATH_PKL = Path.cwd() / 'data' / 'data.pkl'


def get_data(filepath_pkl=FILEPATH_PKL):
    """Get data as a pandas DataFrame"""
    df = pd.read_pickle(str(filepath_pkl))

    # Drop useless columns
    df.drop(columns=['data_filename', 'data_filepath', 'file.name', 'file.size', 'file.url'], inplace=True)

    # Drop messages from bots
    bots = ['GitHub', 'Docker Monitor', 'TeamCity', 'Jenkins', 'JIRA', 'Sphere CI', 'Frontend Bot', 'Travis CI',
            'Prometheus · AlertManager', 'Sphere Staging', 'Sphere Production', 'Subversion', 'Grafana', 'grafana',
            'Standup', 'AnomalyDetector', 'PagerDuty', 'UserVoice', 'Confluence', 'MMS',
            'commercetools GmbH · Docker Monitor', 'Frontend Production', 'Mailroom', 'Stackdriver',
            'Prometheus alerts · AlertManager', 'LaunchDarkly', 'Mailroom · system@iron.io', 'Ru Bot',
            'logentries-alerts', 'HipGif', 'commercetools GmbH · GitHub', 'Status IO', 'StatusPage.io', 'ROSHAMBO!',
            'commercetools GmbH · TeamCity', 'appear.in', 'commercetools GmbH · Travis CI', 'Integrations',
            'Sphere Frontend', 'commercetools GmbH · Datadog', 'commercetools GmbH · Jenkins', 'System',
            'commercetools GmbH · Automation', 'commercetools GmbH · Auto Mation', 'commercetools GmbH · akamel',
            'commercetools GmbH · Subversion', 'commercetools GmbH · Heroku', 'Send from CLI',
            'AgileZen', 'Log Entries', 'Link', 'Guggy', 'Automation', 'lunchbot',
            'Prometheus alerts · Your humble Prometheus']
    df = df[~df['from.name'].isin(bots)]

    # Drop messages by user ids (more consistent than names)
    bots_user_id = ['api', 'guest']
    df = df[~df['from.user_id'].isin(bots_user_id)]

    # Also drop automated messages from non-bots
    automated_message_start = (
        '@here The ES Listing Validator',
        'The ES Listing Validator',
        'Listing validator reports',
        '@here Listing ID',
        '@SphereProduction shipit',
        '@SphereStaging shipit'
    )
    df = df[~df.message.str.startswith(automated_message_start)]

    # Deal with multiple and trailing whitespaces, exclude empty messages
    df.message = df.message.apply(lambda x: re.sub(' +',' ', x).strip())
    df = df[~df.message.isin(['', ' '])]

    return df
