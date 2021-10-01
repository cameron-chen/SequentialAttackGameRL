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
    return [key for key, value in playersObj.items()]

def getStratsMap(egtaGame):
    playerNames = getPlayerNames(egtaGame)
    result = {}
    stratsObject = egtaGame["strategies"]
    for playerName in playerNames:
        stratsList = stratsObject[playerName]
        for i in range(len(stratsList)):
            if type(stratsList[i]) is str:
                stratsList[i] = stratsList[i]
        result[playerName] = stratsList
    return result
        
def getStratCounts(playerNames, playerToStrats):
    return [len(playerToStrats[x]) for x in playerNames]

def setPlayers(game, playerNames):
    for i in range(len(playerNames)):
        # label the player with its name
        game.players[i].label = playerNames[i]

def setPlayerStrategies(game, playerToStrats):
    for i in range(len(game.players)):
        playerName = game.players[i].label
        for j in range(len(playerToStrats[playerName])):
            # label the strategy with its name in playerToStrats
            game.players[i].strategies[j].label = playerToStrats[playerName][j]

def payoffInt(payoffFloat):
    result = Decimal(payoffFloat).quantize(Decimal('1.0000'))
    return result

def getProfileObject(player1, stratPlayer1, player2, stratPlayer2, egtaGame):
    for profile in egtaGame["profiles"]:
        profileAscii = {}
        for key in profile:
            profileAscii[key] = profile[key]
        strat1 = profile[player1][0][0]
        strat2 = profile[player2][0][0]
        if strat1 == stratPlayer1 and strat2 == stratPlayer2:
            return profile
    raise ValueError("profile not found")

def setPayoffs(gambitGame, playerNamesList, playerToStratsList, egtaGame):
    if len(playerNamesList) != 2:
        raise ValueError("must be a 2-player game")
    # get name of player1 and player2
    player1 = playerNamesList[0]
    player2 = playerNamesList[1]
    # get list of strategies for player1 and player2
    stratsPlayer1 = playerToStratsList[player1]
    stratsPlayer2 = playerToStratsList[player2]
    for i in range(len(stratsPlayer1)):
        # iterate over player1 strategies
        stratPlayer1 = stratsPlayer1[i]
        for j in range(len(stratsPlayer2)):
            # iterate over matching player2 strategies
            stratPlayer2 = stratsPlayer2[j]
            curProfile = getProfileObject(player1, stratPlayer1, player2, stratPlayer2, egtaGame)
            payoff1 = curProfile[player1][0][2]
            payoff2 = curProfile[player2][0][2]
            gambitGame[i,j][0] = payoffInt(payoff1)
            gambitGame[i,j][1] = payoffInt(payoff2)

def getGambitGame(egtaGame):
    playerNamesList = getPlayerNames(egtaGame)
    if len(playerNamesList) != 2:
        raise ValueError("must be a 2-player game")

    playerToStratsList = getStratsMap(egtaGame)

    stratCounts = getStratCounts(playerNamesList, playerToStratsList)
    # http://gambit.sourceforge.net/gambit15/pyapi.html
    # create a strategic form game with stratCounts as number of strategies per player
    gambitGame = gambit.Game.new_table(stratCounts)

    # label each player with its name, from playerNamesList
    setPlayers(gambitGame, playerNamesList)
    # label each strategy in the game, with its name from playerToStratsList
    setPlayerStrategies(gambitGame, playerToStratsList)    
    setPayoffs(gambitGame, playerNamesList, playerToStratsList, egtaGame)
    # print(gambitGame.mixed_strategy_profile())
    temp = gambitGame.write()
    print(temp)

    # g = gambit.Game.read_game("/Users/thanhnguyen/Documents/WORKS/SEQUENTIAL_ATTACK/Gambit/gambit-test/e02.nfg")
    # solver = gambit.nash.ExternalEnumPureSolver()
    # solver.solve(g)

    return gambitGame

def pureEq(gambitGame):
    solverPure = gambit.nash.ExternalEnumPureSolver()
    return solverPure.solve(gambitGame)
 
def mixedEq(gambitGame):
    solverMixed = gambit.nash.ExternalEnumMixedSolver()
    return solverMixed.solve(gambitGame)

def printEq(eqList, playerNames, playerToStrats):
    i = 0
    for playerName in playerNames:
        print(playerName)
        stratList = playerToStrats[playerName]
        for strat in stratList:
            prob = eqList[i]
            i += 1
            if prob > 0.0:
                print("\t" + str(prob) + "\t" + strat)

def main(gameFileName):
    egtaGame = getGame(gameFileName)
    gambitGame = getGambitGame(egtaGame)

    playerNames = getPlayerNames(egtaGame)
    playerToStrats = getStratsMap(egtaGame)

    pureEqResult = pureEq(gambitGame)
    print("Pure strategy Nash equilibria:")
    for myPureEq in pureEqResult:
         print("Pure equilibrium:")
         printEq(myPureEq, playerNames, playerToStrats)
    print("")

    print("Mixed strategy Nash equilibria:")
    mixedEqResult = mixedEq(gambitGame)
    for myMixedEq in mixedEqResult:
         print("Mixed equilibrium:")
         printEq(myMixedEq, playerNames, playerToStrats)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("need 1 arg: gameFileName")
    gameFileName = sys.argv[1]
    main(gameFileName)
