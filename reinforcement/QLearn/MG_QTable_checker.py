import argparse
from classes import QTable
from myGame import GameField


# 行動名の取得
def get_action_name(action: GameField.Action):
    if action == GameField.Action.PROCEED:
        return 'PROCEED'
    elif action == GameField.Action.DOWN:
        return 'DOWN'
    elif action == GameField.Action.UP:
        return 'UP'
    elif action == GameField.Action.STAY:
        return 'STAY'
    else:
        return 'UNDEFINED'

ORDER = {GameField.Action.UNDEFINED:0, GameField.Action.PROCEED:1, GameField.Action.DOWN:2, GameField.Action.UP:3, GameField.Action.STAY:4}

parser = argparse.ArgumentParser(description='Black Jack Q Table Checker')
parser.add_argument('--file', type=str, default='', help='filename of Q table to be checked')
args = parser.parse_args()

if args.file != '':
    q_table = QTable(action_class=GameField.Action)
    q_table.load(args.file)
    print('State,Action,Q value')
    for k, v in sorted(q_table.table.items(), key=lambda x:(x[0][0], ORDER[x[0][1]])):
        state, action = k
        print('{},{},{}'.format(state, get_action_name(action), v))
