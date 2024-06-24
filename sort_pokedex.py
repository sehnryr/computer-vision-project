#!/bin/env python3

# Only keep the first 3 columns of the `pokedex.csv` file and write the new data
# to a new file called `pokedex-short.csv`.

from convert_case import kebab_case 
import csv
from unidecode import unidecode

seen_pokemon = set()

pokedex_short = open('datasets/pokedex-short.csv', 'w')
pokedex = open('datasets/pokedex.csv')

reader = csv.reader(pokedex)
writer = csv.writer(pokedex_short)

# skip the header
next(reader)

for row in reader:
    if row[1] in seen_pokemon:
        continue
    # Get the name of the pokemon without the alternate form
    raw_name = unidecode(row[2])
    name = ''
    for i, part in enumerate(raw_name.split(' ')):
        if i == 0:
            name += part
            continue
        if not name.endswith(('.', ':')) and not part.endswith(('.', ':')):
            break
        name += ' ' + part

    kebab_name = kebab_case(
        name.replace('\'', ' ')
            .replace('-', ' ')
            .replace('.', '')
            .replace(':', '')
            .strip()
    )

    seen_pokemon.add(row[1])
    writer.writerow([row[1], name, kebab_name])

pokedex_short.close()
pokedex.close()
