import sys
from typing import List, Tuple
from random import Random
import numpy as np
import pygame

from Const import Const
from utils import is_collision, is_passing, spawn_probability


# ----- Simulation Parameters -----
default_C = Const()

RNG_SEED = 42
FPS = 1             # in frames per second
CELL_WIDTH = 48     # in pixels

# ----- class for state representation -----
class State:
    """Lightweight container for the current simulation state."""
    
    __slots__ = ("C", "y", "v", "D", "H")
    
    def __init__(self, 
                 C: Const,
                 y: int,
                 v: int,
                 D: List[int],
                 H: List[int]
                 ) -> None:
        """Initialize a State.

        Args:
            C (Const): Problem constants used for indexing/validation.
            y (int): Vertical position of the bird.
            v (int): Vertical velocity of the bird.
            D (List[int]): Distances to obstacles (d1, d2, ..., dM).
            H (List[int]): Gap centers (h1, h2, ..., hM).

        Raises:
            AssertionError: If D and H lengths are inconsistent with C.M.
        """
        assert (
            len(D) == len(H) and len(D) == C.M
        ), f"D and H should be of length {C.M}"
        self.C = C
        self.y = y
        self.v = v
        self.D = D
        self.H = H
        
    def to_index(self) -> int:
        """Return the index of this state in the enumeration."""
        x = (self.y, self.v, *self.D, *self.H)
        return self.C.state_to_index(x)
        
    
def get_init_state(C: Const, rng: Random) -> State:
    """Generate the initial state for the simulation.

    The bird starts centered vertically with zero velocity, the first
    obstacle at the far right, and remaining obstacle slots inactive.

    Args:
        C (Const): Problem constants.
        rng (Random): Random source used to pick the first gap center.

    Returns:
        State: The initial simulation state.
    """
    y0 = C.Y // 2
    v0 = 0
    D0 = [C.X - 1] + [0] * (C.M - 1)
    h1 = rng.choice(C.S_h)
    H0 = [h1] + [C.S_h[0]] * (C.M - 1)
    return State(C, y0, v0, D0, H0)

# ----- Simulation -----
def step(C: Const, x: State, u: int, rng: Random) -> Tuple[State, bool]:
    """Advance the simulation by one time step.

    Applies the input (with possible flap disturbance), updates obstacle
    positions and spawns new obstacles as needed, and checks for collision.

    Args:
        C (Const): Problem constants.
        x (State): Current state.
        u (int): Applied input (no flap / weak / strong).
        rng (Random): Random source for spawning and flap disturbance.

    Returns:
        Tuple[State, bool]: (next_state, hit_flag) where hit_flag is True
        if a collision occurred and the run ends.
    """
    y, v, D, H = x.y, x.v, x.D, x.H
    
    if is_collision(C, y, D[0], H[0]):
        return x, True   # End simulation run
    
    # Pre-spawn update
    hat_D, hat_H = D.copy(), H.copy()
    if is_passing(C, y, D[0], H[0]):
        hat_D[0] = D[1] - 1
        hat_H[0] = H[1]
        for i in range(1, C.M - 1):
            hat_D[i] = D[i + 1]
            hat_H[i] = H[i + 1]
        hat_D[C.M-1] = 0
        hat_H[C.M-1] = C.S_h[0]
    else:
        hat_D[0] = D[0] - 1
        
    # Check if new obstacle is spawned
    s = (C.X - 1) - sum(hat_D)
    if rng.random() <= spawn_probability(C, s):
        # spawn obstacle
        k = C.M - 1
        for i in range(1, C.M - 1):
            if hat_D[i] == 0:
                k = i
                break
        hat_D[k] = np.clip(s, C.D_min, C.X - 1)
        hat_H[k] = rng.choice(C.S_h)
        
    # Strong-flap uncertainty
    W_flap = range(-C.V_dev, C.V_dev + 1)
    if u == C.U_strong:
        w_flap = rng.choice(W_flap)
    else:
        w_flap = 0
        
    # next-state update
    y_next = int(np.clip(y + v, 0, C.Y - 1))
    v_next = int(np.clip(v + u + w_flap - C.g, -C.V_max, C.V_max))
    next_x = State(C, y_next, v_next, hat_D, hat_H)
    return next_x, False


# ----- pygame renderer -----
class Renderer:
    """Pygame renderer for the grid world, bird, and obstacles."""
    
    def __init__(self, C: Const) -> None:
        """Create a rendering context and window."""
        pygame.init()
        self.C = C
        self.width = C.X * CELL_WIDTH
        self.height = C.Y * CELL_WIDTH
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 12)
        self.bigfont = pygame.font.SysFont("consolas", 20, bold=True)
        pygame.display.set_caption("Endless-Corridor Flight Simulation")

        self.C_BACK = pygame.color.Color(250, 250, 250)
        self.C_GRID = pygame.color.Color(220, 220, 220)
        self.C_BIRD = pygame.color.Color(20, 20, 20)
        self.C_OBST = pygame.color.Color(40, 120, 200)
        self.C_TEXT = pygame.color.Color(127, 127, 127)
        
    def draw_grid(self) -> None:
        """Draw the grid lines."""
        for x in range(self.C.X + 1):
            pygame.draw.line(
                self.screen, self.C_GRID,
                (x * CELL_WIDTH, 0),
                (x * CELL_WIDTH, self.height),
                1,
            )
        for y in range(self.C.Y + 1):
            pygame.draw.line(
                self.screen, self.C_GRID,
                (0, y * CELL_WIDTH),
                (self.width, y * CELL_WIDTH),
                1,
            )
            
    def draw_bird(self, y: int) -> None:
        """Draw the bird at vertical position y (0 at bottom)."""
        xpix = 0
        ypix = (self.C.Y - 1 - y) * CELL_WIDTH
        offset = max(2, CELL_WIDTH // 10)
        bird_width = CELL_WIDTH - 2 * offset
        rect = pygame.Rect(
            xpix + offset,
            ypix + offset,
            bird_width,
            bird_width)
        pygame.draw.rect(
            self.screen,
            self.C_BIRD,
            rect,
            border_radius = offset // 2
        )
        
    def draw_obstacles(self, D: List[int], H: List[int]) -> None:
        """Draw all active obstacles with a vertical gap at each H[i].

        Args:
            D (List[int]): Distances (relative) to obstacles.
            H (List[int]): Gap centers of obstacles.
        """
        # Cumsum relative distances from D to get absolute x positions
        xs = [D[0]]
        for i in range(1, self.C.M):
            if D[i] == 0:
                break
            xs.append(D[i] + xs[-1])

        for idx, xcol in enumerate(xs):
            if xcol < 0 or xcol >= self.C.X:
                continue
            gap_center = H[idx]
            half = (self.C.G - 1) // 2
            gap_lo = np.clip(gap_center - half, 0, self.C.Y - 1)
            gap_hi = np.clip(gap_center + half, 0, self.C.Y - 1)

            # Lower solid (rows 0..gap_lo-1)
            if gap_lo > 0:
                height_pix = gap_lo * CELL_WIDTH
                rect_low = pygame.Rect(
                    xcol * CELL_WIDTH,
                    self.height - height_pix,
                    CELL_WIDTH,
                    height_pix
                )
                pygame.draw.rect(
                    self.screen, self.C_OBST, rect_low, border_radius = 2
                )

            # Upper solid (rows gap_hi+1 .. Y-1)
            if gap_hi < self.C.Y - 1:
                rows_above = (self.C.Y - 1) - gap_hi
                height_pix = rows_above * CELL_WIDTH
                rect_up = pygame.Rect(
                    xcol * CELL_WIDTH, 
                    0, 
                    CELL_WIDTH, 
                    height_pix
                )
                pygame.draw.rect(
                    self.screen, self.C_OBST, rect_up, border_radius = 2
                )
    
    def draw_hud(
        self,
        step_count: int,
        x: State,
        u: int,
        game_over: bool,
        mode: str,
    ) -> None:
        """Draw status text, control help, and a game-over banner."""
        info = (
            f"Step {step_count} | mode={mode} | y={x.y} v={x.v} | "
            f"d1={x.D[0]} h1={x.H[0]} | u={u:.0f}"
        )
        surf = self.font.render(info, True, self.C_TEXT)
        self.screen.blit(surf, (8, 8))

        help1 = "Hold keys: W=weak, S=strong.  R=reset  Esc=quit"
        surf2 = self.font.render(help1, True, self.C_TEXT)
        self.screen.blit(surf2, (8, 8 + 22))

        if game_over:
            banner1 = self.bigfont.render("GAME OVER", True, (200, 40, 40))
            banner2 = self.font.render(
                "(Press R for reset)", True, (200, 40, 40)
            )

            rect1 = banner1.get_rect(center=(self.width // 2, 40 + 36))
            rect2 = banner2.get_rect(
                midtop=(self.width // 2, rect1.bottom + 6)
            )

            self.screen.blit(banner1, rect1)
            self.screen.blit(banner2, rect2)
            
    def tick(self) -> None:
        """Limit the frame rate to FPS."""
        self.clock.tick(FPS)
        
        
# ----- simulation runner -----
def run_simulation(C: Const, policy: np.ndarray | None = None) -> None:
    """Run the main simulation loop.

    If `policy` is None, the simulation runs in manual mode (hold W/S).
    Otherwise, it uses the integer action at `policy[x.to_index()]`.

    Args:
        C (Const): Problem constants.
        policy (np.ndarray | None): Greedy policy over indices, or None
            for manual mode.
    """
    is_manual = policy is None
    rng = Random(RNG_SEED)
    renderer = Renderer(C)
    
    x = get_init_state(C, rng)
    step_count = 0
    is_gameover = False

    while True:
        u = 0  # default input
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)  # end program
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)  # end program
                if event.key == pygame.K_r:
                    # Reset game to initial state
                    x = get_init_state(C, rng)
                    step_count = 0
                    is_gameover = False
            
        if is_manual:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_s]:
                u = C.U_strong
            elif keys[pygame.K_w]:
                u = C.U_weak
        else:
            u = policy[x.to_index()]
            if u not in C.input_space:
                u = 0
                
        # Rendering
        renderer.screen.fill(renderer.C_BACK)
        renderer.draw_grid()
        renderer.draw_obstacles(x.D, x.H)
        renderer.draw_bird(x.y)
        renderer.draw_hud(
            step_count, x, u, is_gameover, "manual" if is_manual  else "policy"
        )
        pygame.display.flip()
        renderer.tick()
                 
        # Step simulation
        if not is_gameover:
            next_x, hit = step(C, x, u, rng)
            if hit:
                is_gameover = True
            else:
                x = next_x
                step_count += 1
        
if __name__ == "__main__":
    run_simulation(default_C)