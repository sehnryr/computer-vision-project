#!/bin/env python3

# Sort images by pokemon number and copy them from the 'datasets' directory to
# the 'pokemons' directory

from convert_case import kebab_case
import csv
import os
import shutil
from unidecode import unidecode

# Get all files

files = []
skipped_dirs = ['back', 'Back', 'Back, Shiny', 'shiny', 'Shiny', 'up', 'left', 'right', 'down']

for root, dirs, os_files in os.walk('datasets'):
    if any(f'{d}' in root for d in skipped_dirs):
        continue

    for file in os_files:
        if not file.endswith('.png'):
            continue

        files.append(os.path.join(root, file))

reader = csv.reader(open('datasets/pokedex-short.csv'))

# Create a dictionary with the pokemon's number as key and a list of files as 
# value

pokemons = {}

for row in reader:
    number = row[0]
    name = row[1]
    id_ = row[2]

    if pokemons.get(number) is None:
        pokemons[number] = []

    for i, file in enumerate(files):
        file_name = kebab_case(unidecode(
            os.path.basename(file)
                .rsplit('.', 1)[0]
                .replace('.', '')
                .replace('_', '-')
                .replace('(', '')
                .replace(')', '')
                .replace(' ', '-')
                .replace('%', '')
                .lstrip('0')
                .lower()
        ))

        if file_name == number or (file_name).startswith(id_):
            pokemons[number].append(file)
            files.pop(i)

# Create a directory named 'pokemons' and inside it create a directory for each
# pokemon with the number as name and copy the files to the corresponding
# directory

os.makedirs('pokemons', exist_ok=True)

for number, files in pokemons.items():
    os.makedirs(f'pokemons/{number}', exist_ok=True)

    for i, file in enumerate(files):
        shutil.copy(file, f'pokemons/{number}/{number}-{i}.png')
