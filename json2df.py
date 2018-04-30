"""

Chatbot trained on CTP hipchat data.

"""

import json
import os
import pandas as pd
from pathlib import Path

PATH_DATA = Path.cwd() / 'data'
PATH_ROOMS = PATH_DATA / 'rooms'


def json2df():
    """Convert json files from Hipchat export to a single pandas DataFrame file"""
    rooms = [room for room in PATH_ROOMS.iterdir() if room.is_dir()]
    print(f'Data available for {len(rooms)} rooms')

    df = pd.DataFrame()
    for i, room in enumerate(rooms):
        room_files = list(room.iterdir())
        print(f'Getting data from room {i}: {room.stem} (files: {len(room_files)})')
        for file in room_files:
            try:
                messages = json.load(open(str(file)))
                new_data = pd.io.json.json_normalize(messages)
                new_data['room'] = room.stem
                new_data['data_filepath'] = str(file)
                new_data['data_filename'] = file.name
                df = df.append(new_data)
            # Not sure why that AttributeError below occurrs, but we can handle it by making separate calls to
            # json_normalize() for each message (caveat: makes the conversion very slow)
            except AttributeError as e:
                print(f'Found unusual file ({file}) - converting jsons separately for every message in the file...')
                messages = json.load(open(str(file)))
                for message in messages:
                    new_data = pd.io.json.json_normalize(message)
                    new_data['room'] = room
                    new_data['data_filepath'] = str(file)
                    new_data['data_filename'] = file.name
                    df = df.append(new_data)
                continue
            except Exception as e:
                print(f'Skipping file - Unexpected error occurred: {e}')
                continue

    # Sort by room and date
    df.sort_values(by=['room', 'date'], ascending=[True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # df.to_csv(os.path.join(DIR_DATA, 'data.csv'))
    df.to_pickle(str(PATH_DATA / 'data.pkl'))

if __name__ == '__main__':
    json2df()
