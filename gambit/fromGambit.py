import json
import sys
import gambit

def pureEq(gambitGame):
    solverPure = gambit.nash.ExternalEnumPureSolver()
    return solverPure.solve(gambitGame)
 
def mixedEq(gambitGame):
    solverMixed = gambit.nash.ExternalEnumMixedSolver()
    return solverMixed.solve(gambitGame)

def getPlayerNames(gambitGame):
    return [x.label for x in gambitGame.players]

def getPlayerToStrats(gambitGame):
    result = {}
    for player in gambitGame.players:
        playerName = player.label
        strats = [strat.label for strat in player.strategies]
        result[playerName] = strats
    return result

def printEq(eqList, playerNames, playerToStrats):
    i = 0
    for playerName in playerNames:
        print playerName
        stratList = playerToStrats[playerName]
        for strat in stratList:
            prob = eqList[i]
            i += 1
            if prob > 0.0:
                print "\t" + str(prob) + "\t" + strat

def main(gambitFileName):
    gambitGame = gambit.Game.read_game(gambitFileName)

    playerNames = getPlayerNames(gambitGame)
    playerToStrats = getPlayerToStrats(gambitGame)

    pureEqResult = pureEq(gambitGame)
    print "Pure strategy Nash equilibria:"
    for myPureEq in pureEqResult:
         print "Pure equilibrium:"
         printEq(myPureEq, playerNames, playerToStrats)
    print ""

    print "Mixed strategy Nash equilibria:"
    mixedEqResult = mixedEq(gambitGame)
    for myMixedEq in mixedEqResult:
         print "Mixed equilibrium:"
         printEq(myMixedEq, playerNames, playerToStrats)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("need 1 arg: gambitFileName")
    gambitFileName = sys.argv[1]
    main(gambitFileName)
