import sys
import collections
import numpy as np
import heapq
import time
import numpy as np

global posWalls, posGoals

class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])

    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)])

    # print("Layout:\n", layout)
    return np.array(layout)

def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2

    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    # Upper Letter: Push Box
    # Lower Letter: Normal Move
    allActions = [[-1, 0, 'u', 'U'],[1, 0, 'd', 'D'],[0, -1, 'l', 'L'],[0, 1, 'r', 'R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""

    # Example Implement
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)
    startState = (beginPlayer, beginBox)

    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 

    temp = []

    while frontier:
        node = frontier.pop()
        node_action = actions.pop()

        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break

        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
                
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""

    # Khởi tạo trạng thái bắt đầu
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)
    startState = (beginPlayer, beginBox) # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    
    # Tập Frontier (Các Node được thêm vào hàng đợi theo dạng Queue) 
    # Tập Closed Set (Các Node đã được duyệt)
    # Tập Actions (Các hành động tương ứng với các Node trong Frontier)
    frontier = collections.deque([[startState]]) # store states
    exploredSet = set() # store explored node
    actions = collections.deque([[0]]) # store actions

    temp = [] # Result

    # # Mở File và ghi lại log chạy của thuật toán DFS
    # f = open("log/bfs_log.txt", "w")

    ### Implement breadthFirstSearch here
    # Lặp lại khi Frontier vẫn còn phần tử
    while frontier:

        # # Ghi lại Frontier
        # f.write("\nFrontier: ")
        # f.write(str(frontier))

        # Mở node có độ sâu nông nhất => Những Node được thêm vào đầu tiên
        # Popleft tương đương với DeQueue
        node = frontier.popleft()
        node_action = actions.popleft()

        # # Ghi lại Node được Expand
        # f.write("\nNode: ")
        # f.write(str(node))

        # node[-1] = Node hiện tại không bao gồm các tập các Node trên đường đi
        # node[-1][-1] = Vị trí các Boxes của Node hiện tại
        # Kiểm tra vị trí của Boxes của Node hiện tại đã được đặt đúng vị trí chưa == EndState?
        if isEndState(node[-1][-1]):
            # Nếu vị trí của Boxes đã được đặt đúng vị trí
            # Trả về tập các action đến Node đó loại trừ phần tử đầu tiên
            # Phần tử đầu tiên là [0] tương ứng với giá trị khởi tạo ban đầu
            temp += node_action[1:]
            
            # Kết thúc quá trình tìm kiếm
            break

        # Nếu vị trí của Boxes của Node hiện tại chưa được đặt đúng vị trí
        # Và Node hiện tại không bao gồm tập các Node trên đường đi chưa được xét trong tập đóng
        if node[-1] not in exploredSet:
            # Thêm Node hiện tại đang xét vào tập đóng
            exploredSet.add(node[-1])

            # Duyệt qua từng action (Action hợp lệ) có thể thực hiện tại Node đó
            for action in legalActions(node[-1][0], node[-1][1]):

                # Với mỗi action sẽ có các newPosPlayer và newPosBox tương ứng khác nhau
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                # Kiểm tra xem vị trí mới đó của Boxes thì có thể dẫn đến không tìm ra kết quả được hay không?
                # Các vị trí Deadlock
                if isFailed(newPosBox):
                    continue

                # Nếu action đó không dẫn đến Failed (Đẩy các Boxes vào các Deadlock) 
                # Thì thêm Node con đó (Successor) của Node hiện tại vào Frontier
                # Tương ứng với việc thêm các actions để đến Node con đó
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
        
        # # Ghi lại tập các Actions ứng với các Node trong tập Frontier
        # f.write("\nActions: ")
        # f.write(str(list(actions)))

        # # Ghi lại tập Actions tới Node hiện tại đang xét
        # f.write("\nNode Actions: ")
        # f.write(str(list(node_action)))
        # f.write("\n\n")
    
    # # Đóng file log
    # f.close()
    return temp

def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""

    # Khởi tạo trạng thái bắt đầu: Start State bao gồm (vị trí boxes) và (vị trí player)
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)
    startState = (beginPlayer, beginBox)

    # Tập Frontier (Các Node được thêm vào hàng đợi theo dạng Priority Queue) 
    # với giá trị ưu tiên và Cost đến Node đó từ Node bắt đầu
    # Tập Closed Set (Các Node đã được duyệt)
    # Tập Actions (Các hành động tương ứng với các Node trong Frontier theo dạng Priority Queue)
    # với giá trị ưu tiên và Cost đến Node đó từ Node bắt đầu (Tương ứng với Frontier)
    frontier = PriorityQueue()
    frontier.push([startState], 0)
    exploredSet = set()
    actions = PriorityQueue()
    actions.push([0], 0)
    
    temp = [] # Result

    # # Mở File và ghi lại log chạy của thuật toán UCS
    # f = open("log/ucs_log.txt", "w")

    ### Implement breadthFirstSearch here
    # Lặp lại khi Frontier vẫn còn phần tử
    while not frontier.isEmpty():

        # # Ghi lại Frontier
        # f.write("\nFrontier: ")
        # f.write(str(frontier))

        # Mở node có chi phí thấp nhất (Pop theo độ ưu tiên Priority Queue)
        # => Những Node có trọng số Priority thấp nhất tương ứng là Cost thấp nhất
        node = frontier.pop()
        node_action = actions.pop()

        # # Ghi lại Node được Expand
        # f.write("\nNode: ")
        # f.write(str(node))

        # node[-1] = Node hiện tại không bao gồm các tập các Node trên đường đi
        # node[-1][-1] = Vị trí các Boxes của Node hiện tại
        # Kiểm tra vị trí của Boxes của Node hiện tại đã được đặt đúng vị trí chưa == EndState?
        if isEndState(node[-1][-1]):
            # Nếu vị trí của Boxes đã được đặt đúng vị trí
            # Trả về tập các action đến Node đó loại trừ phần tử đầu tiên
            # Phần tử đầu tiên là [0] tương ứng với giá trị khởi tạo ban đầu
            temp += node_action[1:]
            
            # Kết thúc quá trình tìm kiếm
            break

        # Nếu vị trí của Boxes của Node hiện tại chưa được đặt đúng vị trí
        # Và Node hiện tại không bao gồm tập các Node trên đường đi chưa được xét trong tập đóng
        if node[-1] not in exploredSet:
            # Thêm Node hiện tại đang xét vào tập đóng
            exploredSet.add(node[-1])

            # Duyệt qua từng action (Action hợp lệ) có thể thực hiện tại Node đó
            for action in legalActions(node[-1][0], node[-1][1]):

                # Với mỗi action sẽ có các newPosPlayer và newPosBox tương ứng với các Successor khác nhau
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                # Kiểm tra xem vị trí mới đó của Boxes thì có thể dẫn đến không tìm ra kết quả được hay không?
                # Các vị trí Deadlock
                if isFailed(newPosBox):
                    continue

                # Nếu action đó không dẫn đến Failed (Đẩy các Boxes vào các Deadlock) 
                # Thì thêm Node con đó (Successor) của Node hiện tại vào Frontier 
                # (Với độ ưu tiên là Cost từ Node bắt đầu đến Node hiện tại)
                # Tương ứng với việc thêm các actions để đến Node con đó
                # (Cũng với độ ưu tiên tương tự như Node vừa được thêm)
                # Hàm cost(node_action[1:] + [action[-1]]) là hàm tính chi phí đến Node hiện tại + Heuristic
                frontier.push(node + [(newPosPlayer, newPosBox)], cost(node_action[1:] + [action[-1]]))
                actions.push(node_action + [action[-1]], cost(node_action[1:] + [action[-1]]))
            
        # # Ghi lại tập các Actions ứng với các Node trong tập Frontier
        # f.write("\nActions: ")
        # f.write(str(list(actions)))

        # # Ghi lại tập Actions tới Node hiện tại đang xét
        # f.write("\nNode Actions: ")
        # f.write(str(list(node_action)))
        # f.write("\n\n")
    
    # # Đóng file log
    # f.close()
    return temp

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method, index_level):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)

    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':
        result = breadthFirstSearch(gameState)    
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
        
    # Time
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)

    # Ghi vào file log kết quả độ dài đường đi và thời gian thực thi (Thống kê)
    f = open("log/log_result.txt", "a")
    f.write("Method: " + str(method))
    f.write(" | Level: " + str(index_level))
    f.write(" | Time: %.2f second" %(time_end-time_start))
    f.write(" | Length of Solutions: " + str(len(result)) + "\n")

    return result