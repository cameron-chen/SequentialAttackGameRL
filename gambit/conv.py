import sys
import json

from gameanalysis import gameio

game, serial = gameio.read_game(json.load(sys.stdin))
json.dump(serial.to_game_json(game), sys.stdout)
