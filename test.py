import pygame
import sys
from collections import deque
import copy
import time
import heapq
import queue as Q

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
        self.board = board
        self.size = len(board)
        if empty_pos is None:
            for i in range(self.size):
                for j in range(self.size):
                    if board[i][j] == 0:
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
        return self.board == other.board

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
def heuristic(state, goal):
    distance = 0
    for i in range(state.size):
        for j in range(state.size):
            value = state.board[i][j]
            if value != 0:  # Không tính ô trống
                for x in range(goal.size):
                    for y in range(goal.size):
                        if goal.board[x][y] == value:
                            distance += abs(x - i) + abs(y - j)
    return distance

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

# Khởi tạo pygame
pygame.init()

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (160, 160, 160)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (70, 130, 180)
LIGHT_GREEN = (144, 238, 144)
GREEN = (50, 205, 50)
RED = (220, 20, 60)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)

# Kích thước và vị trí
SIZE = 100  # Kích thước của mỗi ô
MARGIN = 10  # Khoảng cách giữa các bảng
PADDING = 20  # Khoảng cách từ viền cửa sổ
BOARD_SIZE = 3 * SIZE  # Kích thước của mỗi bảng
SIDEBAR_WIDTH = 300  # Chiều rộng của thanh bên
# TOTAL_WIDTH = BOARD_SIZE * 2 + MARGIN * 2 + PADDING * 2  # Thu nhỏ chiều rộng
# TOTAL_HEIGHT = BOARD_SIZE * 2 + PADDING * 3 + 300  # Mở rộng chiều dài
TOTAL_WIDTH = 800
TOTAL_HEIGHT = 800

# Khởi tạo cửa sổ
SCREEN = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
pygame.display.set_caption("8-Puzzle Solver")

# Font chữ
FONT = pygame.font.SysFont("Arial", 40)
TITLE_FONT = pygame.font.SysFont("Arial", 28)
INFO_FONT = pygame.font.SysFont("Arial", 20)
BUTTON_FONT = pygame.font.SysFont("Arial", 22)
STAT_FONT = pygame.font.SysFont("Arial", 16)

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

def main_menu():
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
        "A*": (a_star, "A*", RED)
    }
    
    selected_algorithm = "BFS"
    solved = False
    path = None
    result_info = None
    
    # Tạo các nút thuật toán
    algo_buttons = {}
    for i, algo_key in enumerate(algorithms.keys()):
        x = PADDING + i * (SIDEBAR_WIDTH // 3 + 20)  # Căn đều các nút theo chiều ngang
        y = TOTAL_HEIGHT - 100  # Đặt các nút ở phía dưới
        algo_buttons[algo_key] = Button(
            x, y, SIDEBAR_WIDTH // 3, 50, 
            algorithms[algo_key][1], 
            color=algorithms[algo_key][2]
        )
    
    # Tạo nút giải
    solve_button = Button(
        PADDING, 
        TOTAL_HEIGHT - 160, 
        SIDEBAR_WIDTH - 40, 50, 
        "Solve", 
        color=GREEN, 
        hover_color=LIGHT_GREEN
    )
    
    # Vị trí của các bảng
    initial_x = PADDING
    goal_x = TOTAL_WIDTH - BOARD_SIZE - PADDING
    board_y = PADDING + 100  # Đặt bảng trạng thái ở trên cùng
    
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
                for algo_key, button in algo_buttons.items():
                    if button.is_clicked(mouse_pos, event):
                        selected_algorithm = algo_key
                        # Reset kết quả khi chọn thuật toán mới
                        solved = False
                        path = None
                        result_info = None
                
                if solve_button.is_clicked(mouse_pos, event):
                    # Thực hiện giải thuật toán đã chọn
                    algo_func = algorithms[selected_algorithm][0]
                    print(f"Solving with {algorithms[selected_algorithm][1]}...")
                    path, result_info = algo_func(initial_state, goal_state)
                    if path:
                        solved = True
                        print(f"Solution found with {len(path)} steps")
                        print(f"Algorithm checked {result_info['steps_checked']} states")
                        print(f"Execution time: {result_info['time']:.4f} seconds")
                        # Hiển thị giao diện kết quả ngay sau khi giải xong
                        visualize_solution(initial_state, goal_state, path, result_info)
                    else:
                        print("No solution found!")
        
        # Kiểm tra hover
        for button in algo_buttons.values():
            button.check_hover(mouse_pos)
        solve_button.check_hover(mouse_pos)
        
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
        
        # Vẽ các nút thuật toán
        for algo_key, button in algo_buttons.items():
            button.draw(selected=algo_key == selected_algorithm)
            
        # Vẽ nút giải
        solve_button.draw()
        
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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and step_index < steps_total:
                    step_index += 1
                elif event.key == pygame.K_LEFT and step_index > 0:
                    step_index -= 1
                elif event.key == pygame.K_SPACE:
                    auto_play = not auto_play
                    last_step_time = current_time
                elif event.key == pygame.K_ESCAPE:
                    running = False

            # Kiểm tra click chuột lên các nút
            if event.type == pygame.MOUSEBUTTONDOWN:
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
                    running = False

        # Kiểm tra hover
        prev_btn.check_hover(mouse_pos)
        play_btn.check_hover(mouse_pos)
        next_btn.check_hover(mouse_pos)
        speed_btn.check_hover(mouse_pos)
        back_btn.check_hover(mouse_pos)

        # Xóa màn hình
        SCREEN.fill(WHITE)

        # Tiêu đề
        # title = TITLE_FONT.render("Kết quả giải 8-Puzzle", True, BLACK)
        # SCREEN.blit(title, (TOTAL_WIDTH // 2 - title.get_width() // 2, PADDING // 2))

        # Vẽ trạng thái ban đầu
        draw_board(initial_state, initial_x, board_y, "Trạng thái ban đầu")

        # Vẽ trạng thái đích
        draw_board(goal_state, goal_x, board_y, "Trạng thái đích")

        # Vẽ trạng thái hiện tại
        current_state = path[step_index] if step_index < steps_total else goal_state
        draw_board(current_state, current_x, current_y, "Trạng thái hiện tại", highlight=True)

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