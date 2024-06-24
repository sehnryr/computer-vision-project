#!/bin/env python3

import os
import shutil

# Get all directories from the `pokemons` directory
gen_1 = [] # 1-151
gen_2 = [] # 152-251
gen_3 = [] # 252-386
gen_4 = [] # 387-493
gen_5 = [] # 494-649

for directory in os.listdir('pokemons'):
    files = [f'{directory}/{file}' for file in os.listdir(f'pokemons/{directory}')]

    if 1 <= int(directory) < 152:
        gen_1.extend(files)
    elif 152 <= int(directory) < 252:
        gen_2.extend(files)
    elif 252 <= int(directory) < 387:
        gen_3.extend(files)
    elif 387 <= int(directory) < 494:
        gen_4.extend(files)
    elif 494 <= int(directory) < 650:
        gen_5.extend(files)

# Create directories
os.makedirs('generations', exist_ok=True)
os.makedirs('generations/gen-1', exist_ok=True)
os.makedirs('generations/gen-2', exist_ok=True)
os.makedirs('generations/gen-3', exist_ok=True)
os.makedirs('generations/gen-4', exist_ok=True)
os.makedirs('generations/gen-5', exist_ok=True)

# Copy the files to the corresponding directories
for file in gen_1:
    shutil.copy(f'pokemons/{file}', f'generations/gen-1/{file.rsplit('/', 1)[1]}')
for file in gen_2:
    shutil.copy(f'pokemons/{file}', f'generations/gen-2/{file.rsplit('/', 1)[1]}')
for file in gen_3:
    shutil.copy(f'pokemons/{file}', f'generations/gen-3/{file.rsplit('/', 1)[1]}')
for file in gen_4:
    shutil.copy(f'pokemons/{file}', f'generations/gen-4/{file.rsplit('/', 1)[1]}')
for file in gen_5:
    shutil.copy(f'pokemons/{file}', f'generations/gen-5/{file.rsplit('/', 1)[1]}')
