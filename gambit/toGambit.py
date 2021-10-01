import json
import sys
from decimal import *
import gambit

def getGame(gameFileName):
    with open(gameFileName) as f:
        data = json.load(f)
        return data

def getPlayerNames(egtaGame):
    playersObj = egtaGame["players"]
    return [key.encode('ascii') for key, value in playersObj.items()]

def getStratsMap(egtaGame):
    playerNames = getPlayerNames(egtaGame)
    result = {}
    stratsObject = egtaGame["strategies"]
    for playerName in playerNames:
        stratsList = stratsObject[playerName]
        for i in range(len(stratsList)):
            stratsList[i] = stratsList[i].encode('ascii')
        result[playerName] = stratsList
    return result
        
def getStratCounts(playerNames, playerToStrats):
    return [len(playerToStrats[x]) for x in playerNames]

def setPlayers(game, playerNames):
    for i in range(len(playerNames)):
        game.players[i].label = playerNames[i]        

def setPlayerStrategies(game, playerToStrats):
    for i in range(len(game.players)):
        playerName = game.players[i].label
        for j in range(len(playerToStrats[playerName])):
            game.players[i].strategies[j].label = playerToStrats[playerName][j]

def payoffInt(payoffFloat):
    result = Decimal(payoffFloat).quantize(Decimal('1.0000'))
    return result

def getProfileObject(player1, stratPlayer1, player2, stratPlayer2, egtaGame):
    for profile in egtaGame["profiles"]:
        profileAscii = {}
        for key in profile:
            profileAscii[key.encode('ascii')] = profile[key]
        strat1 = profile[player1][0][0]
        strat2 = profile[player2][0][0]
        if strat1 == stratPlayer1 and strat2 == stratPlayer2:
            return profile
    raise ValueError("profile not found")

def setPayoffs(gambitGame, playerNamesList, playerToStratsList, egtaGame):
    if len(playerNamesList) != 2:
        raise ValueError("must be a 2-player game")
    player1 = playerNamesList[0]
    player2 = playerNamesList[1]
    stratsPlayer1 = playerToStratsList[player1]
    stratsPlayer2 = playerToStratsList[player2]
    for i in range(len(stratsPlayer1)):
        stratPlayer1 = stratsPlayer1[i]
        for j in range(len(stratsPlayer2)):
            stratPlayer2 = stratsPlayer2[j]
            curProfile = getProfileObject(player1, stratPlayer1, player2, stratPlayer2, egtaGame)
            payoff1 = curProfile[player1][0][2][0]
            payoff2 = curProfile[player2][0][2][0]
            gambitGame[i,j][0] = payoffInt(payoff1)
            gambitGame[i,j][1] = payoffInt(payoff2)

def getGambitGame(egtaGame, gameTitle):
    playerNamesList = getPlayerNames(egtaGame)
    if len(playerNamesList) != 2:
        raise ValueError("must be a 2-player game")

    playerToStratsList = getStratsMap(egtaGame)

    stratCounts = getStratCounts(playerNamesList, playerToStratsList)
    gambitGame = gambit.Game.new_table(stratCounts)

    gambitGame.title = gameTitle

    setPlayers(gambitGame, playerNamesList)
    setPlayerStrategies(gambitGame, playerToStratsList)    
    setPayoffs(gambitGame, playerNamesList, playerToStratsList, egtaGame)
    return gambitGame

def printGambitGame(outFileName, gambitGame):
    f = open(outFileName, 'w')
    f.write(gambitGame.write(format='nfg'))
    f.close()

def main(gameFileName, gameTitle, outFileName):
    egtaGame = getGame(gameFileName)
    gambitGame = getGambitGame(egtaGame, gameTitle)
    printGambitGame(outFileName, gambitGame)
    print "wrote to file: " + outFileName

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise ValueError("need 3 args: gameFileName gameTitle outFileName")
    gameFileName = sys.argv[1]
    gameTitle = sys.argv[2]
    outFileName = sys.argv[3]
    main(gameFileName, gameTitle, outFileName)
