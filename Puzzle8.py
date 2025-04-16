import pygame
import sys
from collections import deque
import copy
import time
import heapq
import queue as Q
import numpy as np
import random
import math


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (160, 160, 160)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (70, 130, 180)
LIGHT_GREEN = (144, 238, 144)
GREEN = (50, 205, 50)
RED = (220, 20, 60)
ORANGE = (255, 165, 0)
LIGHT_ORANGE = (255, 200, 100)  # Add this line
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Khởi tạo pygame
pygame.init()
# Kích thước và vị trí
SIZE = 100  # Kích thước của mỗi ô
MARGIN = 10  # Khoảng cách giữa các bảng
PADDING = 20  # Khoảng cách từ viền cửa sổ
BOARD_SIZE = 3 * SIZE  # Kích thước của mỗi bảng
SIDEBAR_WIDTH = 300  # Chiều rộng của thanh bên
TOTAL_WIDTH = 1000 # Chiều rộng của cửa sổ
TOTAL_HEIGHT = 800 # Chiều cao của cửa sổ

# Khởi tạo cửa sổ
SCREEN = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
pygame.display.set_caption("8-Puzzle Solver")

# Font chữ
FONT = pygame.font.SysFont("Arial", 40)
TITLE_FONT = pygame.font.SysFont("Arial", 28)
INFO_FONT = pygame.font.SysFont("Arial", 20)
BUTTON_FONT = pygame.font.SysFont("Arial", 22)
STAT_FONT = pygame.font.SysFont("Arial", 16)

# Priority Queue cho thuật toán UCS
class PriorityQueue(object):
    def __init__(self):
        self.queue = Q.PriorityQueue()
        
    def empty(self):
        return self.queue.empty()
        
    def put(self, item):
        self.queue.put(item)
        
    def get(self):
        return self.queue.get()

class PuzzleState:
    def __init__(self, board, empty_pos=None):
        self.board = np.asarray(board)  # Chuyển board thành mảng NumPy
        self.size = len(board)
        if self.board.ndim != 2 or self.board.shape[0] != self.board.shape[1]:
            raise ValueError("board phải là mảng 2 chiều vuông.")
        if empty_pos is None:
            for i in range(self.size):
                for j in range(self.size):
                    if self.board[i][j] == 0:
                        self.empty_pos = (i, j)
                        return
        else:
            self.empty_pos = empty_pos

    def __str__(self):
        result = ""
        for row in self.board:
            result += " ".join(str(x) if x != 0 else "_" for x in row) + "\n"
        return result

    def __eq__(self, other):
        if isinstance(other, PuzzleState):
            return np.array_equal(self.board, other.board)
        return False

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.board))

    def get_possible_moves(self):
        moves = []
        i, j = self.empty_pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
        for di, dj in directions:
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < self.size and 0 <= new_j < self.size:
                new_board = copy.deepcopy(self.board)
                new_board[i][j], new_board[new_i][new_j] = new_board[new_i][new_j], new_board[i][j]
                new_state = PuzzleState(new_board, (new_i, new_j))
                moves.append(new_state)
        return moves


# Tìm vị trí ô trống (0)
def find_blank(state):
    """
    Tìm vị trí của ô trống (giá trị 0) trong bảng.
    """
    board = np.asarray(state.board)  # Đảm bảo board là mảng NumPy
    if board.ndim != 2:  # Kiểm tra nếu board không phải là mảng 2 chiều
        raise ValueError("state.board phải là mảng NumPy 2 chiều.")
    return np.where(board == 0)

# Tạo các trạng thái lân cận bằng cách di chuyển ô trống
def get_neighbors(state):
    """
    Tạo các trạng thái lân cận bằng cách di chuyển ô trống.
    """
    neighbors = []
    x, y = find_blank(state)  # Tìm vị trí ô trống
    x, y = x[0], y[0]  # Lấy giá trị từ mảng trả về của np.where
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Lên, xuống, trái, phải

    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < state.size and 0 <= new_y < state.size:  # Kiểm tra trong phạm vi bảng
            new_board = copy.deepcopy(state.board)
            new_board[x][y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[x][y]  # Hoán đổi ô trống
            neighbors.append(PuzzleState(new_board, (new_x, new_y)))  # Tạo trạng thái mới

    return neighbors

def set_initial_state(default_state):
    """
    Cho phép người dùng tự cài đặt trạng thái đầu hoặc sử dụng trạng thái mặc định.
    """
    current_state = default_state
    selected_cell = None
    input_value = ""
    
    # Tạo nút xác nhận và nút mặc định
    confirm_button = Button(
        TOTAL_WIDTH // 2 - 220, 
        TOTAL_HEIGHT - 100,
        200, 50,
        "Xác nhận",
        color=GREEN,
        hover_color=LIGHT_GREEN
    )
    
    default_button = Button(
        TOTAL_WIDTH // 2 + 20,
        TOTAL_HEIGHT - 100,
        200, 50,
        "Mặc định",
        color=ORANGE,
        hover_color=LIGHT_ORANGE
    )

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Xử lý click vào ô trong bảng
                for i in range(3):
                    for j in range(3):
                        cell_rect = pygame.Rect(
                            TOTAL_WIDTH//2 - BOARD_SIZE//2 + j*SIZE,
                            TOTAL_HEIGHT//2 - BOARD_SIZE//2 + i*SIZE,
                            SIZE, SIZE
                        )
                        if cell_rect.collidepoint(mouse_pos):
                            selected_cell = (i, j)
                            input_value = ""
                
                # Xử lý click vào nút
                if confirm_button.is_clicked(mouse_pos, event):
                    if is_valid_state(current_state.board):
                        return current_state
                    else:
                        draw_message_box("Trạng thái không hợp lệ!", RED)
                        pygame.display.flip()
                        pygame.time.wait(2000)
                
                elif default_button.is_clicked(mouse_pos, event):
                    return default_state
            
            elif event.type == pygame.KEYDOWN and selected_cell:
                i, j = selected_cell
                if event.key == pygame.K_BACKSPACE:
                    input_value = ""
                elif event.unicode.isdigit() and len(input_value) < 1:
                    value = int(event.unicode)
                    if 0 <= value <= 8:
                        new_board = copy.deepcopy(current_state.board)
                        new_board[i][j] = value
                        current_state = PuzzleState(new_board)
                        selected_cell = None

        # Vẽ giao diện
        SCREEN.fill(WHITE)
        
        # Vẽ tiêu đề
        title = TITLE_FONT.render("Cài đặt trạng thái ban đầu", True, BLACK)
        SCREEN.blit(title, (TOTAL_WIDTH//2 - title.get_width()//2, 50))
        
        # Vẽ bảng
        draw_board(current_state, 
                  TOTAL_WIDTH//2 - BOARD_SIZE//2,
                  TOTAL_HEIGHT//2 - BOARD_SIZE//2,
                  highlight_cell=selected_cell)
        
        # Vẽ hướng dẫn
        guide_text = INFO_FONT.render("Click vào ô và nhập số từ 0-8", True, BLACK)
        SCREEN.blit(guide_text, (TOTAL_WIDTH//2 - guide_text.get_width()//2, 
                                TOTAL_HEIGHT//2 + BOARD_SIZE//2 + 30))
        
        # Vẽ các nút
        confirm_button.draw()
        default_button.draw()
        
        pygame.display.flip()

def is_valid_state(board):
    """
    Kiểm tra tính hợp lệ của trạng thái.
    - Mỗi số từ 0-8 chỉ xuất hiện một lần
    - Có đủ 9 số từ 0-8
    """
    numbers = [num for row in board for num in row]
    return sorted(numbers) == list(range(9))

def is_solvable(state):
    """
    Kiểm tra trạng thái có thể giải được hay không dựa trên số lần đảo ngược (inversions).
    """
    flat_board = [num for row in state.board for num in row if num != 0]  # Bỏ qua ô trống (0)
    inversions = 0

    for i in range(len(flat_board)):
        for j in range(i + 1, len(flat_board)):
            if flat_board[i] > flat_board[j]:
                inversions += 1
 
    # Trạng thái có thể giải được nếu số lần đảo ngược là chẵn
    return inversions % 2 == 0

def heuristic(state, goal_state):
    """
    Tính khoảng cách Manhattan giữa trạng thái hiện tại và trạng thái đích.
    """
    cost = 0
    for num in range(1, 9):  # Bỏ qua ô trống
        for i in range(state.size):
            for j in range(state.size):
                if state.board[i][j] == num:
                    for x in range(goal_state.size):
                        for y in range(goal_state.size):
                            if goal_state.board[x][y] == num:
                                cost += abs(x - i) + abs(y - j)  # Khoảng cách Manhattan
    return cost

def apply_action(state, action):
    """
    Áp dụng action cho một state và trả về state mới.
    """
    x, y = state.empty_pos
    
    if action == "UP" and x > 0:
        new_board = state.board.copy()
        new_board[x][y], new_board[x-1][y] = new_board[x-1][y], new_board[x][y]
        return PuzzleState(new_board, (x-1, y))
    
    elif action == "DOWN" and x < 2:
        new_board = state.board.copy()
        new_board[x][y], new_board[x+1][y] = new_board[x+1][y], new_board[x][y]
        return PuzzleState(new_board, (x+1, y))
    
    elif action == "LEFT" and y > 0:
        new_board = state.board.copy()
        new_board[x][y], new_board[x][y-1] = new_board[x][y-1], new_board[x][y]
        return PuzzleState(new_board, (x, y-1))
    
    elif action == "RIGHT" and y < 2:
        new_board = state.board.copy()
        new_board[x][y], new_board[x][y+1] = new_board[x][y+1], new_board[x][y]
        return PuzzleState(new_board, (x, y+1))
    
    return None

def bfs(initial_state, goal_state):
    queue = deque([(initial_state, [])])
    visited = set([initial_state])
    steps = 0
    start_time = time.time()

    while queue:
        steps += 1
        current_state, path = queue.popleft()

        if current_state == goal_state:
            end_time = time.time()
            result_info = {
                "steps_checked": steps,
                "path_length": len(path),
                "time": end_time - start_time,
                "states_visited": len(visited)
            }
            return path, result_info

        for next_state in current_state.get_possible_moves():
            if next_state not in visited:
                visited.add(next_state)
                new_path = path + [next_state]
                queue.append((next_state, new_path))

    return None, {"error": "Không tìm thấy lời giải!"}

def dfs(initial_state, goal_state, max_depth=50):
    stack = [(initial_state, [], 0)]  # (trạng thái hiện tại, đường đi, độ sâu hiện tại)
    visited = set()
    steps = 0
    start_time = time.time()

    while stack:
        current_state, path, depth = stack.pop()
        steps += 1

        if current_state == goal_state:
            end_time = time.time()
            result_info = {
                "steps_checked": steps,
                "path_length": len(path),
                "time": end_time - start_time,
                "states_visited": len(visited)
            }
            return path, result_info

        if depth >= max_depth:
            continue  # Giới hạn độ sâu để tránh lặp vô tận

        visited.add(current_state)

        for next_state in current_state.get_possible_moves():
            if next_state not in visited:
                stack.append((next_state, path + [next_state], depth + 1))

    return None, {"error": "Không tìm thấy lời giải!"}

def ucs(initial_state, goal_state):
    pq = PriorityQueue()
    counter = 0  # Biến đếm để phân biệt các phần tử cùng cost
    pq.put((0, counter, initial_state, []))  # (cost, counter, state, path)
    visited = set()
    steps = 0
    start_time = time.time()

    while not pq.empty():
        cost, _, current_state, path = pq.get()
        steps += 1

        if current_state == goal_state:
            end_time = time.time()
            result_info = {
                "steps_checked": steps,
                "path_length": len(path),
                "time": end_time - start_time,
                "states_visited": len(visited)
            }
            return path, result_info

        if current_state in visited:
            continue

        visited.add(current_state)

        for next_state in current_state.get_possible_moves():
            if next_state not in visited:
                counter += 1  # Tăng biến đếm để đảm bảo duy nhất
                pq.put((cost + 1, counter, next_state, path + [next_state]))  # Thêm chi phí 1 mỗi bước

    return None, {"error": "Không tìm thấy lời giải!"}

def greedy_search(initial_state, goal_state):
    pq = []
    counter = 0  # Bộ đếm để đảm bảo tính duy nhất
    heapq.heappush(pq, (heuristic(initial_state, goal_state), counter, initial_state, []))  # (heuristic, counter, state, path)
    visited = set()
    steps = 0
    start_time = time.time()

    while pq:
        steps += 1
        _, _, current_state, path = heapq.heappop(pq)

        # Kiểm tra nếu đã đạt trạng thái đích
        if current_state == goal_state:
            end_time = time.time()
            result_info = {
                "steps_checked": steps,
                "path_length": len(path),
                "time": end_time - start_time,
                "states_visited": len(visited)
            }
            return path, result_info

        # Đánh dấu trạng thái hiện tại là đã thăm
        if current_state in visited:
            continue
        visited.add(current_state)

        # Thêm các trạng thái tiếp theo vào hàng đợi ưu tiên
        for next_state in current_state.get_possible_moves():
            if next_state not in visited:
                counter += 1
                heapq.heappush(pq, (heuristic(next_state, goal_state), counter, next_state, path + [next_state]))

    # Nếu không tìm thấy lời giải
    end_time = time.time()
    return None, {
        "error": "Không tìm thấy lời giải!",
        "steps_checked": steps,
        "time": end_time - start_time,
        "states_visited": len(visited)
    }

def a_star(initial_state, goal_state):
    pq = []
    counter = 0  # Bộ đếm để đảm bảo tính duy nhất
    heapq.heappush(pq, (0 + heuristic(initial_state, goal_state), counter, 0, initial_state, []))  # (f, counter, g, state, path)
    visited = set()
    steps = 0
    start_time = time.time()

    while pq:
        steps += 1
        _, _, g, current_state, path = heapq.heappop(pq)

        # Kiểm tra nếu đã đạt trạng thái đích
        if current_state == goal_state:
            end_time = time.time()
            result_info = {
                "steps_checked": steps,
                "path_length": len(path),
                "time": end_time - start_time,
                "states_visited": len(visited)
            }
            return path, result_info

        # Đánh dấu trạng thái hiện tại là đã thăm
        if current_state in visited:
            continue
        visited.add(current_state)

        # Thêm các trạng thái tiếp theo vào hàng đợi ưu tiên
        for next_state in current_state.get_possible_moves():
            if next_state not in visited:
                counter += 1
                new_g = g + 1
                new_f = new_g + heuristic(next_state, goal_state)
                heapq.heappush(pq, (new_f, counter, new_g, next_state, path + [next_state]))

    # Nếu không tìm thấy lời giải
    end_time = time.time()
    return None, {
        "error": "Không tìm thấy lời giải!",
        "steps_checked": steps,
        "time": end_time - start_time,
        "states_visited": len(visited)
    }

def ida_star(initial_state, goal_state):
    def search(path, g, bound):
        """
        Hàm tìm kiếm đệ quy với giới hạn chi phí (bound).
        """
        current_state = path[-1]
        f = g + heuristic(current_state, goal_state)

        if f > bound:
            return f  # Trả về chi phí mới (bound mới)

        if current_state == goal_state:
            return path  # Tìm thấy lời giải

        min_bound = float('inf')  # Giá trị bound tối thiểu cho lần lặp tiếp theo

        for next_state in current_state.get_possible_moves():
            if next_state not in path:  # Tránh lặp lại trạng thái trong đường đi
                path.append(next_state)
                result = search(path, g + 1, bound)
                if isinstance(result, list):  # Nếu tìm thấy lời giải
                    return result
                min_bound = min(min_bound, result)  # Cập nhật bound tối thiểu
                path.pop()  # Quay lui

        return min_bound

    bound = heuristic(initial_state, goal_state)  # Bắt đầu với chi phí heuristic
    path = [initial_state]
    steps = 0
    start_time = time.time()

    while True:
        result = search(path, 0, bound)
        steps += len(path)
        if isinstance(result, list):  # Nếu tìm thấy lời giải
            end_time = time.time()
            result_info = {
                "steps_checked": steps,
                "path_length": len(result) - 1,
                "time": end_time - start_time,
                "states_visited": steps
            }
            return result, result_info
        if result == float('inf'):  # Không tìm thấy lời giải
            end_time = time.time()
            return None, {
                "error": "Không tìm thấy lời giải!",
                "steps_checked": steps,
                "time": end_time - start_time,
                "states_visited": steps
            }
        bound = result  # Cập nhật bound cho lần lặp tiếp theo

def ids(initial_state, goal_state, max_depth=50):
    """
    Thuật toán Iterative Deepening Search (IDS).
    """
    def dls(state, goal_state, depth, path, visited):
        """
        Depth-Limited Search (DLS): Tìm kiếm theo chiều sâu với giới hạn độ sâu.
        """
        if depth == 0:
            return None  # Đạt giới hạn độ sâu, không tìm thấy lời giải
        if state == goal_state:
            return path  # Tìm thấy lời giải

        visited.add(state)

        for next_state in state.get_possible_moves():
            if next_state not in visited:
                result = dls(next_state, goal_state, depth - 1, path + [next_state], visited)
                if result is not None:
                    return result

        return None

    # Kiểm tra trạng thái có thể giải được
    if not is_solvable(initial_state):
        return None, {
            "error": "Trạng thái không thể giải được!",
            "steps_checked": 0,
            "time": 0,
            "states_visited": 0
        }

    steps = 0
    start_time = time.time()

    for depth in range(1, max_depth + 1):
        visited = set()
        result = dls(initial_state, goal_state, depth, [initial_state], visited)
        steps += len(visited)
        if result is not None:
            end_time = time.time()
            result_info = {
                "steps_checked": steps,
                "path_length": len(result),
                "time": end_time - start_time,
                "states_visited": steps
            }
            return result, result_info

    # Nếu không tìm thấy lời giải
    end_time = time.time()
    return None, {
        "error": "Không tìm thấy lời giải!",
        "steps_checked": steps,
        "time": end_time - start_time,
        "states_visited": steps
    }

def simple_hill_climbing(initial_state, goal_state):
    """
    Thuật toán Simple Hill Climbing cho bài toán 8-Puzzle.
    """
    current_state = initial_state
    path = [current_state]
    visited = set()
    steps = 0
    start_time = time.time()

    while current_state != goal_state:
        visited.add(current_state)
        steps += 1

        # Lấy tất cả các trạng thái lân cận
        neighbors = current_state.get_possible_moves()

        # Tìm trạng thái lân cận tốt nhất dựa trên heuristic
        next_state = None
        min_heuristic = float('inf')
        for neighbor in neighbors:
            if neighbor not in visited:
                h = heuristic(neighbor, goal_state)
                if h < min_heuristic:
                    min_heuristic = h
                    next_state = neighbor

        # Nếu không có trạng thái lân cận nào tốt hơn, dừng thuật toán
        if next_state is None or min_heuristic >= heuristic(current_state, goal_state):
            end_time = time.time()
            return None, {
                "error": "Không tìm thấy lời giải!",
                "steps_checked": steps,
                "time": end_time - start_time,
                "states_visited": len(visited)
            }

        # Chuyển sang trạng thái lân cận tốt nhất
        current_state = next_state
        path.append(current_state)

    # Nếu tìm thấy trạng thái đích
    end_time = time.time()
    result_info = {
        "steps_checked": steps,
        "path_length": len(path),
        "time": end_time - start_time,
        "states_visited": len(visited)
    }
    return path, result_info

def hill_climbing(start_state, goal_state):
    """
    Thuật toán Hill Climbing tìm trạng thái tốt nhất.
    """
    state = start_state
    while True:
        neighbors = get_neighbors(state)
        best_neighbor = min(neighbors, key=lambda neighbor: heuristic(neighbor, goal_state), default=None)  # Chọn trạng thái có heuristic nhỏ nhất
        
        if best_neighbor is None or heuristic(best_neighbor, goal_state) >= heuristic(state, goal_state):
            break  # Không có bước nào tốt hơn, dừng lại
        
        state = best_neighbor  # Cập nhật trạng thái tốt nhất

    return state

def simulated_annealing(initial_state, goal_state, initial_temperature=1000, cooling_rate=0.99, min_temperature=0.1):
    """
    Thuật toán Simulated Annealing cho bài toán 8-Puzzle.
    """
    current_state = initial_state
    current_heuristic = heuristic(current_state, goal_state)
    temperature = initial_temperature
    path = [current_state]
    steps = 0
    visited = set()
    start_time = time.time()

    while temperature > min_temperature:
        steps += 1
        visited.add(current_state)

        # Lấy tất cả các trạng thái lân cận
        neighbors = current_state.get_possible_moves()

        # Chọn ngẫu nhiên một trạng thái lân cận
        next_state = random.choice(neighbors)
        next_heuristic = heuristic(next_state, goal_state)

        # Tính toán sự thay đổi heuristic
        delta_heuristic = next_heuristic - current_heuristic

        # Quyết định chấp nhận trạng thái mới
        if delta_heuristic < 0 or random.uniform(0, 1) < math.exp(-delta_heuristic / temperature):
            current_state = next_state
            current_heuristic = next_heuristic
            path.append(current_state)

        # Kiểm tra nếu đã đạt trạng thái đích
        if current_state == goal_state:
            end_time = time.time()
            result_info = {
                "steps_checked": steps,
                "path_length": len(path),
                "time": end_time - start_time,
                "states_visited": len(visited)
            }
            return path, result_info

        # Giảm nhiệt độ
        temperature *= cooling_rate

    # Nếu không tìm thấy lời giải
    end_time = time.time()
    return None, {
        "error": "Không tìm thấy lời giải!",
        "steps_checked": steps,
        "time": end_time - start_time,
        "states_visited": len(visited)
    }

def stochastic_hill_climbing(initial_state, goal_state):
    """
    Thuật toán Stochastic Hill Climbing cho bài toán 8-Puzzle.
    """
    current_state = initial_state
    path = [current_state]
    visited = set()
    steps = 0
    start_time = time.time()

    while current_state != goal_state:
        visited.add(current_state)
        steps += 1

        # Lấy tất cả các trạng thái lân cận
        neighbors = current_state.get_possible_moves()

        # Lọc các trạng thái lân cận tốt hơn trạng thái hiện tại
        better_neighbors = [
            neighbor for neighbor in neighbors
            if heuristic(neighbor, goal_state) < heuristic(current_state, goal_state)
        ]

        # Nếu không có trạng thái lân cận nào tốt hơn, dừng thuật toán
        if not better_neighbors:
            end_time = time.time()
            return None, {
                "error": "Không tìm thấy lời giải!",
                "steps_checked": steps,
                "time": end_time - start_time,
                "states_visited": len(visited)
            }

        # Chọn ngẫu nhiên một trạng thái lân cận tốt hơn
        next_state = random.choice(better_neighbors)

        # Chuyển sang trạng thái lân cận được chọn
        current_state = next_state
        path.append(current_state)

    # Nếu tìm thấy trạng thái đích
    end_time = time.time()
    result_info = {
        "steps_checked": steps,
        "path_length": len(path),
        "time": end_time - start_time,
        "states_visited": len(visited)
    }
    return path, result_info

def beam_search(initial_state, goal_state, beam_width=2):
    """
    Thuật toán Beam Search cho bài toán 8-Puzzle.
    """
    start_time = time.time()
    steps = 0
    visited = set()

    # Khởi tạo danh sách các trạng thái hiện tại (beam)
    beam = [(initial_state, [])]  # (trạng thái, đường đi)

    while beam:
        steps += 1
        next_beam = []

        for current_state, path in beam:
            # Kiểm tra nếu đã đạt trạng thái đích
            if current_state == goal_state:
                end_time = time.time()
                result_info = {
                    "steps_checked": steps,
                    "path_length": len(path),
                    "time": end_time - start_time,
                    "states_visited": len(visited)
                }
                return path, result_info

            # Đánh dấu trạng thái hiện tại là đã thăm
            visited.add(current_state)

            # Lấy tất cả các trạng thái lân cận
            for neighbor in current_state.get_possible_moves():
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    next_beam.append((neighbor, new_path))

        # Sắp xếp các trạng thái lân cận theo heuristic và chỉ giữ lại `beam_width` trạng thái tốt nhất
        next_beam.sort(key=lambda x: heuristic(x[0], goal_state))
        beam = next_beam[:beam_width]

    # Nếu không tìm thấy lời giải
    end_time = time.time()
    return None, {
        "error": "Không tìm thấy lời giải!",
        "steps_checked": steps,
        "time": end_time - start_time,
        "states_visited": len(visited)
    }

def and_or_graph_search(initial_state, goal_state):
    """
    Thuật toán AND-OR Graph Search cho bài toán 8-Puzzle.
    Trả về (path, result_info) hoặc (None, error_info)
    """
    steps = 0
    start_time = time.time()
    visited = set()

    def or_search(state, depth=0):
        """
        OR node: Tìm một đường đi khả thi đến goal state
        """
        nonlocal steps
        steps += 1

        if state == goal_state:
            return [state]

        if state in visited or depth > 50:  # Giới hạn độ sâu để tránh stack overflow
            return None

        visited.add(state)
        
        # Thử từng action có thể
        for next_state in state.get_possible_moves():
            if next_state not in visited:
                result = and_search(next_state, depth + 1)
                if result is not None:
                    return [state] + result

        return None

    def and_search(state, depth=0):
        """
        AND node: Phải tìm được đường đi cho tất cả trạng thái con
        """
        nonlocal steps
        steps += 1

        if state == goal_state:
            return [state]

        if state in visited or depth > 50:
            return None

        visited.add(state)

        # Lấy tất cả trạng thái kế tiếp có thể
        next_states = state.get_possible_moves()
        if not next_states:
            return None

        # Phải tìm được đường đi cho mọi trạng thái con
        paths = []
        for next_state in next_states:
            result = or_search(next_state, depth + 1)
            if result is None:
                return None
            paths.append(result)

        # Kết hợp các đường đi
        return [state] + [state for path in paths for state in path]

    # Bắt đầu tìm kiếm từ trạng thái ban đầu
    path = or_search(initial_state)
    end_time = time.time()

    if path:
        # Loại bỏ các trạng thái trùng lặp trong path
        unique_path = []
        for state in path:
            if not unique_path or state != unique_path[-1]:
                unique_path.append(state)
                
        return unique_path, {
            "steps_checked": steps,
            "path_length": len(unique_path),
            "time": end_time - start_time,
            "states_visited": len(visited)
        }
    else:
        return None, {
            "error": "Không tìm thấy lời giải!",
            "steps_checked": steps,
            "time": end_time - start_time,
            "states_visited": len(visited)
        }

def sensorless_problem(initial_state, goal_state):
    """
    Thuật toán Sensorless Problem cho bài toán 8-Puzzle.
    """
    from collections import deque
    steps = 0
    visited_beliefs = set()
    start_time = time.time()

    # Khởi tạo belief state ban đầu và hàng đợi
    initial_belief = frozenset([initial_state])
    queue = deque([(initial_belief, [initial_state])])  # (belief_state, path)

    while queue:
        current_belief, path = queue.popleft()
        steps += 1

        # Kiểm tra nếu belief state chỉ chứa goal state
        if len(current_belief) == 1 and next(iter(current_belief)) == goal_state:
            end_time = time.time()
            return path, {
                "steps_checked": steps,
                "time": end_time - start_time,
                "states_visited": len(visited_beliefs),
                "final_belief_size": len(current_belief)
            }

        # Kiểm tra nếu belief state này đã được thăm
        if current_belief in visited_beliefs:
            continue
        visited_beliefs.add(current_belief)

        # Xác định các actions có thể thực hiện
        possible_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        # Thử từng action
        for action in possible_actions:
            next_belief = set()
            next_state = None
            
            # Áp dụng action cho mỗi state trong belief state hiện tại
            for state in current_belief:
                # Lấy trạng thái mới sau khi thực hiện action
                new_state = apply_action(state, action)
                if new_state is not None:
                    next_belief.add(new_state)
                    next_state = new_state  # Lưu một trạng thái để thêm vào path
                else:
                    next_belief.add(state)
            
            # Chuyển next_belief thành frozenset để có thể hash
            next_belief_frozen = frozenset(next_belief)
            
            # Nếu belief state mới chưa được thăm và không rỗng
            if next_belief_frozen not in visited_beliefs and next_belief and next_state:
                new_path = path + [next_state]
                queue.append((next_belief_frozen, new_path))

    # Không tìm thấy lời giải
    end_time = time.time()
    return None, {
        "error": "Không tìm thấy lời giải!",
        "steps_checked": steps,
        "time": end_time - start_time,
        "states_visited": len(visited_beliefs)
    }

def genetic_algorithm(initial_state, goal_state, population_size=100, generations=50):
    """
    Thuật toán Genetic Algorithm cho bài toán 8-Puzzle.
    """
    steps = 0
    start_time = time.time()
    visited = set()

    def create_random_solution():
        """Tạo một chuỗi actions ngẫu nhiên"""
        solution_length = random.randint(20, 50)
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        return [random.choice(actions) for _ in range(solution_length)]

    def create_initial_population():
        """Tạo quần thể ban đầu"""
        return [create_random_solution() for _ in range(population_size)]

    def fitness(solution):
        """Đánh giá độ phù hợp của một solution"""
        current = initial_state
        path = [current]
        
        for action in solution:
            next_state = apply_action(current, action)
            if next_state is None:
                return float('inf'), path
            current = next_state
            path.append(current)
        
        return heuristic(current, goal_state), path

    def crossover(parent1, parent2):
        """Lai ghép hai parents để tạo ra offspring"""
        point = random.randint(0, min(len(parent1), len(parent2)))
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(solution, mutation_rate=0.1):
        """Đột biến một solution"""
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        mutated = solution.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.choice(actions)
        return mutated

    # Khởi tạo quần thể
    population = create_initial_population()
    best_solution = None
    best_fitness = float('inf')
    best_path = None

    # Tiến hóa qua các thế hệ
    for generation in range(generations):
        steps += 1

        # Đánh giá fitness cho toàn bộ quần thể
        fitness_scores = []
        for solution in population:
            score, path = fitness(solution)
            fitness_scores.append((score, solution, path))
            
            # Cập nhật best solution
            if score < best_fitness:
                best_fitness = score
                best_solution = solution
                best_path = path
                
                # Nếu tìm thấy goal state
                if score == 0:
                    end_time = time.time()
                    return best_path, {
                        "steps_checked": steps,
                        "path_length": len(best_path),
                        "time": end_time - start_time,
                        "states_visited": len(visited),
                        "generations": generation + 1
                    }

        # Sắp xếp quần thể theo fitness
        fitness_scores.sort(key=lambda x: x[0])
        
        # Chọn các cá thể tốt nhất làm parents
        parents = [item[1] for item in fitness_scores[:population_size//2]]
        
        # Tạo quần thể mới
        new_population = parents.copy()
        
        # Lai ghép và đột biến để tạo offspring
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]

    # Nếu không tìm thấy lời giải
    end_time = time.time()
    if best_path is not None:
        return best_path, {
            "steps_checked": steps,
            "path_length": len(best_path),
            "time": end_time - start_time,
            "states_visited": len(visited),
            "generations": generations
        }
    else:
        return None, {
            "error": "Không tìm thấy lời giải!",
            "steps_checked": steps,
            "time": end_time - start_time,
            "states_visited": len(visited),
            "generations": generations
        }

# Lớp Button để tạo các nút tương tác
class Button:
    def __init__(self, x, y, width, height, text, color=GRAY, hover_color=LIGHT_BLUE, text_color=BLACK, border_radius=10):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_radius = border_radius
        self.is_hovered = False
        
    def draw(self, selected=False):
        color = GREEN if selected else (self.hover_color if self.is_hovered else self.color)
        pygame.draw.rect(SCREEN, color, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(SCREEN, BLACK, self.rect, 2, border_radius=self.border_radius)
        
        text_surf = BUTTON_FONT.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        SCREEN.blit(text_surf, text_rect)
        
    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered
        
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False

def draw_board(state, x, y, title=None, highlight=False):
    # Vẽ nền cho bảng
    bg_rect = pygame.Rect(x, y, BOARD_SIZE, BOARD_SIZE)
    bg_color = LIGHT_BLUE if highlight else WHITE
    pygame.draw.rect(SCREEN, bg_color, bg_rect)
    pygame.draw.rect(SCREEN, DARK_BLUE, bg_rect, 3)
    
    # Vẽ tiêu đề cho bảng nếu có
    if title:
        title_text = TITLE_FONT.render(title, True, BLACK)
        title_rect = title_text.get_rect(center=(x + BOARD_SIZE // 2, y - 30))
        SCREEN.blit(title_text, title_rect)
    
    # Vẽ các ô trong bảng
    for i in range(3):
        for j in range(3):
            value = state.board[i][j]
            rect = pygame.Rect(x + j * SIZE, y + i * SIZE, SIZE, SIZE)
            pygame.draw.rect(SCREEN, GRAY, rect, 2)
            if value != 0:
                text = FONT.render(str(value), True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                SCREEN.blit(text, text_rect)

def draw_stat_box(x, y, width, title, value, color=LIGHT_BLUE):
    box_height = 60
    # Vẽ hộp
    pygame.draw.rect(SCREEN, color, (x, y, width, box_height), border_radius=5)
    pygame.draw.rect(SCREEN, BLACK, (x, y, width, box_height), 2, border_radius=5)
    
    # Vẽ tiêu đề
    title_text = STAT_FONT.render(title, True, BLACK)
    SCREEN.blit(title_text, (x + 10, y + 10))
    
    # Vẽ giá trị
    value_text = TITLE_FONT.render(str(value), True, BLACK)
    SCREEN.blit(value_text, (x + 10, y + 30))

def draw_message_box(message, color=GREEN):
    """
    Hiển thị khung thông báo ở giữa màn hình.
    """
    box_width = 400
    box_height = 150
    box_x = (TOTAL_WIDTH - box_width) // 2
    box_y = (TOTAL_HEIGHT - box_height) // 2

    # Vẽ khung thông báo
    pygame.draw.rect(SCREEN, color, (box_x, box_y, box_width, box_height), border_radius=10)
    pygame.draw.rect(SCREEN, BLACK, (box_x, box_y, box_width, box_height), 3, border_radius=10)

    # Vẽ nội dung thông báo
    text = TITLE_FONT.render(message, True, BLACK)
    text_rect = text.get_rect(center=(box_x + box_width // 2, box_y + box_height // 2))
    SCREEN.blit(text, text_rect)

def algorithm_menu():
    """
    Hiển thị menu để người dùng chọn thuật toán.
    """
    running = True
    clock = pygame.time.Clock()
    selected_algorithm = None

    # Danh sách các thuật toán
    algorithms = [
        "BFS",
        "DFS",
        "UCS",
        "Greedy",
        "A*",
        "IDA*",
        "IDS",
        "Simple Hill Climbing",
        "Hill Climbing",
        "Stochastic Hill Climbing",
        "Simulated Annealing",
        "Beam Search",
        "AND-OR Graph Search",
        "Sensorless Problem",  # Thêm Sensorless Problem
        "Genetic Algorithm"  # Thêm Genetic Algorithm
    ]

    # Tạo các nút cho từng thuật toán
    buttons = []
    num_columns = 2  # Số cột (chia đều thành 2 hàng)
    button_width = 300
    button_height = 50
    button_spacing_x = 40  # Khoảng cách giữa các nút theo chiều ngang
    button_spacing_y = 20  # Khoảng cách giữa các nút theo chiều dọc
    start_x = (TOTAL_WIDTH - (num_columns * button_width + (num_columns - 1) * button_spacing_x)) // 2
    start_y = 100

    for i, algo in enumerate(algorithms):
        row = i // num_columns  # Xác định hàng
        col = i % num_columns  # Xác định cột
        x = start_x + col * (button_width + button_spacing_x)
        y = start_y + row * (button_height + button_spacing_y)
        buttons.append(Button(x, y, button_width, button_height, algo, color=LIGHT_BLUE, hover_color=GREEN))

    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, button in enumerate(buttons):
                    if button.is_clicked(mouse_pos, event):
                        selected_algorithm = algorithms[i]
                        running = False

        # Xóa màn hình
        SCREEN.fill(WHITE)

        # Tiêu đề menu
        title = TITLE_FONT.render("Chọn thuật toán để giải 8-Puzzle", True, BLACK)
        SCREEN.blit(title, (TOTAL_WIDTH // 2 - title.get_width() // 2, 20))

        # Vẽ các nút
        for button in buttons:
            button.check_hover(mouse_pos)
            button.draw()

        pygame.display.flip()
        clock.tick(60)

    return selected_algorithm

def main_menu():
    while True:  # Thêm vòng lặp để quay lại menu chọn thuật toán
        # Gọi menu chọn thuật toán
        selected_algorithm = algorithm_menu()
        if not selected_algorithm:
            return  # Nếu không chọn thuật toán, thoát khỏi menu chính

        # Thông số mặc định
        initial_board = [
            [2, 6, 5],
            [0, 8, 7],
            [4, 3, 1]
        ]

        goal_board = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ]

        initial_state = PuzzleState(initial_board)
        goal_state = PuzzleState(goal_board)
        
        # Các tùy chọn thuật toán
        algorithms = {
            "BFS": (bfs, "BFS", LIGHT_GREEN),
            "DFS": (dfs, "DFS", LIGHT_BLUE),
            "UCS": (ucs, "UCS", YELLOW),
            "Greedy": (greedy_search, "Greedy", ORANGE),
            "A*": (a_star, "A*", RED),
            "IDA*": (ida_star, "IDA*", DARK_BLUE),
            "IDS": (ids, "IDS", LIGHT_GREEN),
            "Simple Hill Climbing": (simple_hill_climbing, "Simple Hill Climbing", ORANGE),
            "Hill Climbing": (hill_climbing, "Hill Climbing", ORANGE),
            "Stochastic Hill Climbing": (stochastic_hill_climbing, "Stochastic Hill Climbing", LIGHT_BLUE),
            "Simulated Annealing": (simulated_annealing, "Simulated Annealing", YELLOW),  
            "Beam Search": (beam_search, "Beam Search", ORANGE),
            "AND-OR Graph Search": (and_or_graph_search, "AND-OR Graph Search", LIGHT_BLUE),
            "Sensorless Problem": (sensorless_problem, "Sensorless Problem", PURPLE),
            "Genetic Algorithm": (genetic_algorithm, "Genetic Algorithm", PURPLE)
        
            }

        solved = False
        path = None
        result_info = None
        
        # Tạo nút giải
        solve_button = Button(
            TOTAL_WIDTH // 2 - 100,  # Căn giữa theo chiều ngang
            TOTAL_HEIGHT - 160, 
            200, 50, 
            "Solve", 
            color=GREEN, 
            hover_color=LIGHT_GREEN
        )

        # Tạo nút Back
        back_button = Button(
            TOTAL_WIDTH // 2 - 100,  # Căn giữa theo chiều ngang
            TOTAL_HEIGHT - 90, 
            200, 50, 
            "Back", 
            color=RED, 
            hover_color=LIGHT_BLUE
        )
        
        # Vị trí của các bảng
        initial_x = PADDING
        goal_x = TOTAL_WIDTH - BOARD_SIZE - PADDING
        board_y = PADDING + 100  # Đặt bảng trạng thái ở trên cùng
        
        # Tạo nút Setup
        setup_button = Button(
            TOTAL_WIDTH//2 - 100,
            TOTAL_HEIGHT - 230,
            200, 50,
            "Cài đặt",
            color=ORANGE,
            hover_color=LIGHT_GREEN
        )
        
        # Vòng lặp chính
        running = True
        clock = pygame.time.Clock()
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                # Kiểm tra sự kiện click chuột
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if solve_button.is_clicked(mouse_pos, event):
                        # Kiểm tra trạng thái có thể giải được
                        if not is_solvable(initial_state):
                            print("Trạng thái không thể giải được!")
                            draw_message_box("Không thể giải được", RED)
                            pygame.display.flip()
                            pygame.time.wait(2000)  # Hiển thị thông báo trong 2 giây
                            continue

                        # Thực hiện giải thuật toán đã chọn
                        algo_func = algorithms[selected_algorithm][0]
                        print(f"Solving with {algorithms[selected_algorithm][1]}...")

                        # Xử lý riêng cho Sensorless Problem
                        if selected_algorithm == "Sensorless Problem":
                            path, result_info = algo_func(initial_state, goal_state)
                            if path is not None:
                                solved = True
                                print(f"Solution found!")
                                print(f"Steps checked: {result_info['steps_checked']}")
                                print(f"Time: {result_info['time']:.4f} seconds")
                                draw_message_box("Có thể thực hiện", GREEN)
                                pygame.display.flip()
                                pygame.time.wait(2000)
                                visualize_solution(initial_state, goal_state, path, result_info)
                            else:
                                print("No solution found!")
                                print(result_info["error"])
                                draw_message_box("Không thể thực hiện", RED)
                                pygame.display.flip()
                                pygame.time.wait(2000)
                        else:
                            # Xử lý các thuật toán khác
                            path, result_info = algo_func(initial_state, goal_state)
                            if path:
                                solved = True
                                print(f"Solution found with {len(path)} steps")
                                print(f"Steps checked: {result_info['steps_checked']}")
                                print(f"Time: {result_info['time']:.4f} seconds")
                                draw_message_box("Có thể thực hiện", GREEN)
                                pygame.display.flip()
                                pygame.time.wait(2000)
                                visualize_solution(initial_state, goal_state, path, result_info)
                            else:
                                print("No solution found!")
                                draw_message_box("Không thể thực hiện", RED)
                                pygame.display.flip()
                                pygame.time.wait(2000)
                    elif back_button.is_clicked(mouse_pos, event):
                        running = False  # Thoát khỏi vòng lặp hiện tại để quay lại menu chọn thuật toán
                    elif setup_button.is_clicked(mouse_pos, event):
                        initial_state = set_initial_state(PuzzleState(initial_board))
            
            # Kiểm tra hover
            solve_button.check_hover(mouse_pos)
            back_button.check_hover(mouse_pos)
            setup_button.check_hover(mouse_pos)
            
            # Xóa màn hình
            SCREEN.fill(WHITE)
            
            # Thanh công cụ
            toolbar_rect = pygame.Rect(0, 0, TOTAL_WIDTH, 80)
            pygame.draw.rect(SCREEN, LIGHT_BLUE, toolbar_rect)
            pygame.draw.rect(SCREEN, DARK_BLUE, toolbar_rect, 3)
            
            # Tiêu đề chính
            main_title = TITLE_FONT.render("8-Puzzle Solver", True, BLACK)
            SCREEN.blit(main_title, (TOTAL_WIDTH // 2 - main_title.get_width() // 2, 20))
            
            # Vẽ trạng thái ban đầu
            draw_board(initial_state, initial_x, board_y, "Initial State")
            
            # Vẽ trạng thái đích
            draw_board(goal_state, goal_x, board_y, "Goal State")
            
            # Vẽ nút giải
            solve_button.draw()

            # Vẽ nút Back
            back_button.draw()

            # Vẽ nút Setup
            setup_button.draw()
            
            pygame.display.flip()
            clock.tick(60)

def visualize_solution(initial_state, goal_state, path, result_info):
    clock = pygame.time.Clock()
    step_index = 0
    auto_play = False
    delay = 1000  # Milliseconds between steps in auto-play
    last_step_time = 0
    speed_factor = 1.0  # Hệ số tốc độ

    # Vị trí của các bảng
    initial_x = PADDING  # Bảng trạng thái đầu nằm sát vách trái
    goal_x = TOTAL_WIDTH - BOARD_SIZE - PADDING  # Bảng trạng thái đích nằm sát vách phải
    board_y = PADDING + 40  # Căn chỉnh vị trí theo chiều dọc
    current_x = TOTAL_WIDTH // 2 - BOARD_SIZE // 2  # Bảng trạng thái hiện tại ở giữa
    current_y = board_y + BOARD_SIZE + 50  # Bảng trạng thái hiện tại nằm bên dưới hai bảng trạng thái

    # Thông tin hiển thị
    steps_total = len(path)

    # Tạo các nút điều khiển
    prev_btn = Button(PADDING, current_y + BOARD_SIZE + 50, 100, 40, "< Trước")
    play_btn = Button(PADDING + 120, current_y + BOARD_SIZE + 50, 100, 40, "Chạy", GREEN)
    next_btn = Button(PADDING + 240, current_y + BOARD_SIZE + 50, 100, 40, "Sau >")
    speed_btn = Button(PADDING + 360, current_y + BOARD_SIZE + 50, 100, 40, "Tốc độ x1")
    back_btn = Button(PADDING + 480, current_y + BOARD_SIZE + 50, 100, 40, "Quay lại", RED)

    running = True
    while running:
        current_time = pygame.time.get_ticks()
        mouse_pos = pygame.mouse.get_pos()

        # Tự động chuyển bước nếu chế độ auto-play được bật
        if auto_play and current_time - last_step_time > delay / speed_factor and step_index < steps_total:
            step_index += 1
            last_step_time = current_time
            if step_index >= steps_total:
                auto_play = False  # Tự động dừng khi kết thúc

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if prev_btn.is_clicked(mouse_pos, event) and step_index > 0:
                    step_index -= 1
                elif play_btn.is_clicked(mouse_pos, event):
                    auto_play = not auto_play
                    last_step_time = current_time
                elif next_btn.is_clicked(mouse_pos, event) and step_index < steps_total:
                    step_index += 1
                elif speed_btn.is_clicked(mouse_pos, event):
                    # Thay đổi tốc độ: 1x -> 2x -> 3x -> 0.5x -> 1x
                    if speed_factor == 1.0:
                        speed_factor = 2.0
                    elif speed_factor == 2.0:
                        speed_factor = 3.0
                    elif speed_factor == 3.0:
                        speed_factor = 0.5
                    else:
                        speed_factor = 1.0
                    speed_btn.text = f"Tốc độ x{speed_factor}"
                elif back_btn.is_clicked(mouse_pos, event):
                    return  # Quay lại giao diện chọn thuật toán

        # Kiểm tra hover
        prev_btn.check_hover(mouse_pos)
        play_btn.check_hover(mouse_pos)
        next_btn.check_hover(mouse_pos)
        speed_btn.check_hover(mouse_pos)
        back_btn.check_hover(mouse_pos)

        # Xóa màn hình
        SCREEN.fill(WHITE)

        # Vẽ trạng thái ban đầu
        draw_board(initial_state, initial_x, board_y, "Initial State")

        # Vẽ trạng thái đích
        draw_board(goal_state, goal_x, board_y, "Goal State")

        # Vẽ trạng thái hiện tại
        current_state = path[step_index] if step_index < steps_total else goal_state
        draw_board(current_state, current_x, current_y, "Current State", highlight=True)

        # Vẽ thông tin và hướng dẫn
        step_text = INFO_FONT.render(f"Bước: {step_index}/{steps_total}", True, BLACK)
        SCREEN.blit(step_text, (PADDING, current_y + BOARD_SIZE + 20))

        # Hiển thị thêm thông tin về kết quả
        info_x = goal_x + BOARD_SIZE + 20
        info_text = INFO_FONT.render(f"Tổng số bước: {steps_total}", True, BLACK)
        SCREEN.blit(info_text, (info_x, board_y + 20))

        info_text = INFO_FONT.render(f"Trạng thái đã xét: {result_info['steps_checked']}", True, BLACK)
        SCREEN.blit(info_text, (info_x, board_y + 50))

        info_text = INFO_FONT.render(f"Thời gian: {result_info['time']:.4f} giây", True, BLACK)
        SCREEN.blit(info_text, (info_x, board_y + 80))

        # Vẽ các nút điều khiển
        prev_btn.draw()
        play_btn.text = "Dừng" if auto_play else "Chạy"
        play_btn.color = RED if auto_play else GREEN
        play_btn.draw()
        next_btn.draw()
        speed_btn.draw()
        back_btn.draw()

        # Hướng dẫn
        guide_y = current_y + BOARD_SIZE + 100
        guide_text = INFO_FONT.render("Mũi tên trái/phải: Di chuyển từng bước | Space: Chạy/Dừng | Esc: Quay lại", True, BLACK)
        SCREEN.blit(guide_text, (PADDING, guide_y))

        pygame.display.flip()
        clock.tick(60)

def main():
    main_menu()

if __name__ == "__main__":  
    main()

