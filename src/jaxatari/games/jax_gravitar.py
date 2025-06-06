import jax
import jax.numpy as jnp
import pygame
import numpy as np
from jaxatari.core import JaxEnvironment
import jax.random as jrandom
from typing import Tuple, Generic, TypeVar, Dict, NamedTuple
import math
"""
    Group member of the Gravitar: Xusong Yin, Elizaveta Kuznetsova, Li Dai
"""


#action constant
NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
DOWNRIGHT = 8
DOWNLEFT = 9
UPFIRE = 10
RIGHTFIRE = 11
LEFTFIRE = 12
DOWNFIRE = 13
UPRIGHTFIRE = 14
UPLEFTFIRE = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE = 17

#HUD settings
HUD_HEIGHT = 30
MAX_LIVES = 6
HUD_PADDING = 5
HUD_SHIP_WIDTH = 10
HUD_SHIP_HEIGHT = 12
HUD_SHIP_SPACING = 12



# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

def get_action_from_key():
    keys = pygame.key.get_pressed()

    thrust = keys[pygame.K_UP]
    rotate_left = keys[pygame.K_LEFT]
    rotate_right = keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]
    down = keys[pygame.K_DOWN]

    #Map action combinations    映射动作组合  
    if fire:
        return 1
    elif thrust:
        return 2
    elif rotate_right:
        return 3  # RIGHT
    elif rotate_left:
        return 4  # LEFT
    elif down:
        return 5  # DOWN
    elif thrust and rotate_right:
        return 6  # UPRIGHT
    elif thrust and rotate_left:
        return 7
    elif down and rotate_right:
        return 8  # DOWNRIGHT
    elif down and rotate_left:
        return 9  # DOWNLEFT
    elif thrust and fire:
        return 10  # UPFIRE
    elif rotate_right and fire:
        return 11  # RIGHTFIRE
    elif rotate_left and fire:
        return 12  # LEFTFIRE
    elif down and fire:
        return 13  # DOWNFIRE
    elif thrust and rotate_right and fire:
        return 14  # UPRIGHTFIRE
    elif thrust and rotate_left and fire:
        return 15  # UPLEFTFIRE
    elif down and rotate_right and fire:
        return 16  # DOWNRIGHTFIRE
    elif down and rotate_left and fire:
        return 17  # DOWNLEFTFIRE
    
    #default return to action 0     默认返回action 0   
    return 0

def add_bullet(bullets: dict, new_bullet: dict) -> dict:
    #Convert the new bullet to jnp.array and concatenate    单个新子弹转为 jnp.array 并拼接
    new_x  = jnp.array([new_bullet["x"]])
    new_y  = jnp.array([new_bullet["y"]])
    new_vx = jnp.array([new_bullet["vx"]])
    new_vy = jnp.array([new_bullet["vy"]])

    return {
        "x": jnp.concatenate([bullets["x"], new_x]),
        "y": jnp.concatenate([bullets["y"], new_y]),
        "vx": jnp.concatenate([bullets["vx"], new_vx]),
        "vy": jnp.concatenate([bullets["vy"], new_vy])
    }

@jax.jit
def update_bullets(bullets: dict) -> dict:
    #Update bullet positions   更新子弹位置
    x = bullets["x"] + bullets["vx"]
    y = bullets["y"] + bullets["vy"]
    #Keep valid bullets   保留有效子弹
    valid = (y >= HUD_HEIGHT) & (y <= WINDOW_HEIGHT) & (x >= 0) & (x <= WINDOW_WIDTH)

    #Set invalid bullet positions to -1 (off-screen)   将无效子弹位置设置为 -1，不可见区
    x = jnp.where(valid, x, -1.0)
    y = jnp.where(valid, y, -1.0)
    vx = jnp.where(valid, bullets["vx"], 0.0)
    vy = jnp.where(valid, bullets["vy"], 0.0)

    return {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy
    }

@jax.jit
def ship_step(state, action, window_size, hud_height) -> dict:
    #basic parameters   基本参数
    rotation_speed = 0.2
    thrust_power = 0.4
    gravity = 0.08

    rotate_right_actions = jnp.array([3, 6, 8, 11, 14, 16])
    rotate_left_actions  = jnp.array([4, 7, 9, 12, 15, 17])
    thrust_actions       = jnp.array([2, 6, 7, 10, 14, 15])
    down_thrust_actions  = jnp.array([5, 8, 9, 13, 16, 17])

    right       = jnp.isin(action, rotate_right_actions)
    left        = jnp.isin(action, rotate_left_actions)
    thrust      = jnp.isin(action, thrust_actions)
    down_thrust = jnp.isin(action, down_thrust_actions)

    angle = jnp.where(right, state["angle"] + rotation_speed, state["angle"])
    angle = jnp.where(left,  angle - rotation_speed, angle)

    vx = jnp.where(thrust,      state["vx"] + jnp.cos(angle) * thrust_power, state["vx"])
    vy = jnp.where(thrust,      state["vy"] + jnp.sin(angle) * thrust_power, state["vy"])
    vx = jnp.where(down_thrust, vx - jnp.cos(angle) * thrust_power, vx)
    vy = jnp.where(down_thrust, vy - jnp.sin(angle) * thrust_power, vy)

    vy += gravity

    x = state["x"] + vx
    y = state["y"] + vy

    window_width, window_height = window_size
    x = jnp.clip(x, 0, window_width - 5)
    y = jnp.clip(y, hud_height, window_height - 5)

    return {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "angle": angle
    }

@jax.jit
def enemy_step(enemies: dict, window_width: int) -> dict:
    x = enemies["x"] + enemies["vx"]
    
    #Reverse direction at the boundaries   到边界就反向
    left_hit = x <= 0
    right_hit = (x + enemies["w"]) >= window_width
    hit_edge = left_hit | right_hit

    vx = jnp.where(hit_edge, -enemies["vx"], enemies["vx"])

    return {
        "x": x,
        "y": enemies["y"],
        "w": enemies["w"],
        "h": enemies["h"],
        "vx": vx
    }

@jax.jit
def enemy_fire(enemies: dict,
               ship_x: float,
               ship_y: float,
               enemy_bullet_speed: float,
               fire_cooldown: int,
               fire_interval: int,
               key: jax.random.PRNGKey
               ) -> Tuple[dict, int, jax.random.PRNGKey]:

    dx = ship_x - enemies["x"]
    dy = ship_y - enemies["y"]
    dist = jnp.sqrt(dx ** 2 + dy ** 2) + 1e-6
    vx = dx / dist * enemy_bullet_speed
    vy = dy / dist * enemy_bullet_speed

    should_fire = (fire_cooldown == 0)

    #Keep bullet shape unchanged, mark invalid bullets by setting position to -1         保持 bullets 的 shape 不变，只将位置设为 -1 表示无效
    x_out = jnp.where(should_fire, enemies["x"], jnp.full_like(enemies["x"], -1.0))
    y_out = jnp.where(should_fire, enemies["y"], jnp.full_like(enemies["y"], -1.0))
    vx_out = jnp.where(should_fire, vx, jnp.zeros_like(vx))
    vy_out = jnp.where(should_fire, vy, jnp.zeros_like(vy))

    bullets_out = {
        "x": x_out,
        "y": y_out,
        "vx": vx_out,
        "vy": vy_out
    }

    fire_cooldown = jnp.where(should_fire, fire_interval, jnp.maximum(fire_cooldown - 1, 0))

    return bullets_out, fire_cooldown, key

def check_collision(ship_state, bullets, enemy_bullets, enemies, ship_hitbox_size):
    def safe_filter(d, mask):
        indices = jnp.where(mask, size=mask.shape[0])[0]
        return {k: jnp.take(v, indices, axis=0) for k, v in d.items()}

    #Filter dummy bullets (ship and enemies)     过滤 dummy bullets（飞船和敌人）
    if bullets["x"].shape[0] > 0:
        valid_bullet_mask = bullets["x"] >= 0.0
        bullets = safe_filter(bullets, valid_bullet_mask)

    if enemy_bullets["x"].shape[0] > 0:
        valid_enemy_mask = enemy_bullets["x"] >= 0.0
        enemy_bullets = safe_filter(enemy_bullets, valid_enemy_mask)  
    #ship hitbox
    sx1 = ship_state["x"] - ship_hitbox_size
    sx2 = ship_state["x"] + ship_hitbox_size
    sy1 = ship_state["y"] - ship_hitbox_size
    sy2 = ship_state["y"] + ship_hitbox_size

    #bullet hits the enemy
    if (bullets["x"].shape[0] > 0) and (enemies["x"].shape[0] > 0):
        bx = bullets["x"][:, None]
        by = bullets["y"][:, None]

        ex1 = enemies["x"] - enemies["w"] / 2
        ex2 = enemies["x"] + enemies["w"] / 2
        ey1 = enemies["y"] - enemies["h"] / 2
        ey2 = enemies["y"] + enemies["h"] / 2

        cond_bx = (bx >= ex1) & (bx <= ex2)
        cond_by = (by >= ey1) & (by <= ey2)
        bullet_hit_enemy = cond_bx & cond_by

        bullet_hit_any_enemy = jnp.any(bullet_hit_enemy, axis=1)
        hit_any_enemy = jnp.any(bullet_hit_enemy, axis=0)

        keep_bullets = ~bullet_hit_any_enemy
        remaining_bullets = safe_filter(bullets, keep_bullets)

        alive_flags = ~hit_any_enemy
        remaining_enemies = safe_filter(enemies, alive_flags)
    else:
        remaining_bullets = bullets
        remaining_enemies = enemies

    #ship crash with enemy
    if enemies["x"].shape[0] > 0:
        ex1 = enemies["x"] - enemies["w"] / 2
        ex2 = enemies["x"] + enemies["w"] / 2
        ey1 = enemies["y"] - enemies["h"] / 2
        ey2 = enemies["y"] + enemies["h"] / 2

        overlap_x = (sx1 <= ex2) & (sx2 >= ex1)
        overlap_y = (sy1 <= ey2) & (sy2 >= ey1)
        crashed = jnp.any(overlap_x & overlap_y)
    else:
        crashed = False

    #enemy bullet hits the ship
    if enemy_bullets["x"].shape[0] > 0:
        ebx = enemy_bullets["x"]
        eby = enemy_bullets["y"]
        hit_by_bullet_flags = (ebx >= sx1) & (ebx <= sx2) & (eby >= sy1) & (eby <= sy2)
        hit_by_bullet = jnp.any(hit_by_bullet_flags)
        keep_enemy_bullets = ~hit_by_bullet_flags
        remaining_enemy_bullets = safe_filter(enemy_bullets, keep_enemy_bullets)
    else:
        hit_by_bullet = False
        remaining_enemy_bullets = enemy_bullets

    return remaining_bullets, remaining_enemies, remaining_enemy_bullets, crashed, hit_by_bullet

@jax.jit
def step_core_level(state: dict,
              action: int,
              bullets: dict,
              cooldown: int,
              enemies: dict,
              enemy_bullets: dict,
              fire_cooldown: int,
              key: jax.random.PRNGKey,
              window_size: Tuple[int, int],
              hud_height: int,
              cooldown_max: int,
              bullet_speed: float,
              enemy_bullet_speed: float,
              fire_interval: int
              ) -> Tuple[jnp.ndarray, dict, dict, int, dict, dict, int, jax.random.PRNGKey, float, bool, dict]:
    
    #ship moves    飞船移动
    new_state = ship_step(state, action, window_size, hud_height)
    #if fire    判断开火
    fire_actions = jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])
    is_firing = jnp.isin(action, fire_actions)
    #fire the bullets 发射子弹
    bullets, cooldown = fire_bullet(new_state, is_firing, cooldown, cooldown_max, bullet_speed, bullets)
    #update the bullets 更新子弹位置
    bullets = update_bullets(bullets)
    #enemy moves  敌人移动
    enemies = enemy_step(enemies, window_width=window_size[0])
    #enemy fire   敌人发射子弹
    new_enemy_bullets, fire_cooldown, key = enemy_fire(
        enemies,
        new_state["x"],
        new_state["y"],
        enemy_bullet_speed,
        fire_cooldown,
        fire_interval,
        key
    )
    #Merge bullets    合并子弹
    enemy_bullets = {
        "x": jnp.concatenate([enemy_bullets["x"], new_enemy_bullets["x"]]),
        "y": jnp.concatenate([enemy_bullets["y"], new_enemy_bullets["y"]]),
        "vx": jnp.concatenate([enemy_bullets["vx"], new_enemy_bullets["vx"]]),
        "vy": jnp.concatenate([enemy_bullets["vy"], new_enemy_bullets["vy"]])
    }
    #update enemy bullets 更新敌人子弹位置
    enemy_bullets = update_bullets(enemy_bullets)
    #check the collision   碰撞检测
    bullets, enemies, enemy_bullets, crashed, hit_by_bullet = check_collision(
        new_state, bullets, enemy_bullets, enemies, ship_hitbox_size=10.0
    )
    #construct the observation    构建观测
    obs = jnp.array([new_state["x"], new_state["y"], new_state["vx"], new_state["vy"], new_state["angle"]])
    #if dead?
    dead = crashed | hit_by_bullet
    reward = jnp.where(dead, -10.0, -1.0)
    done = dead

    info = {
        "crash": crashed,
        "hit_by_bullet": hit_by_bullet
    }

    return obs, new_state, bullets, cooldown, enemies, enemy_bullets, fire_cooldown, key, reward, done, info

@jax.jit
def step_core_map(state: dict,
                  action: int,
                  window_size: Tuple[int, int],
                  hud_height: int
                  ) -> Tuple[jnp.ndarray, dict, float, bool, dict, bool, int]:
    
    #use the ship_step function          调用 ship_step 得到更新后的状态
    new_state = ship_step(state, action, window_size, hud_height)
    #construct observation   构建观测 obs
    obs = jnp.array([
        new_state["x"],
        new_state["y"],
        new_state["vx"],
        new_state["vy"],
        new_state["angle"]
    ])

    reward = 0.0
    done = False
    info = {}
    reset = False
    level = 0

    return obs, new_state, reward, done, info, reset, level

@jax.jit
def fire_bullet(state: dict,
                is_firing,
                cooldown: int,
                cooldown_max: int,
                bullet_speed: float,
                bullets: dict
                ) -> Tuple[dict, int]:

    tip_offset = 10.0
    should_fire = is_firing & (cooldown == 0)

    tip_x = state["x"] + jnp.cos(state["angle"]) * tip_offset
    tip_y = state["y"] + jnp.sin(state["angle"]) * tip_offset
    bullet_vx = jnp.cos(state["angle"]) * bullet_speed
    bullet_vy = jnp.sin(state["angle"]) * bullet_speed

    tip_x = tip_x[None]
    tip_y = tip_y[None]
    bullet_vx = bullet_vx[None]
    bullet_vy = bullet_vy[None]
    #True 分支
    new_bullets_true = {
        "x": jnp.concatenate([bullets["x"], tip_x]),
        "y": jnp.concatenate([bullets["y"], tip_y]),
        "vx": jnp.concatenate([bullets["vx"], bullet_vx]),
        "vy": jnp.concatenate([bullets["vy"], bullet_vy]),
    }
    #False 分支：补一个假的“空拼接”，保持 shape 一致
    zero_pad = jnp.zeros((1,), dtype=bullets["x"].dtype)
    new_bullets_false = {
        "x": jnp.concatenate([bullets["x"], zero_pad]),
        "y": jnp.concatenate([bullets["y"], zero_pad]),
        "vx": jnp.concatenate([bullets["vx"], zero_pad]),
        "vy": jnp.concatenate([bullets["vy"], zero_pad]),
    }

    new_bullets = jax.lax.cond(
        should_fire,
        lambda _: new_bullets_true,
        lambda _: new_bullets_false,
        operand=None
    )
    
    new_cooldown = jnp.where(should_fire, cooldown_max, jnp.maximum(cooldown - 1, 0))

    return new_bullets, new_cooldown

class JaxGravitar(JaxEnvironment):
    def __init__(self):
        super().__init__()
        self.obs_shape = (5,) # [x, y, vx, vy, angle]
        self.num_actions = 18
        
        #Initialize font library     初始化字体库
        pygame.font.init()
        #Initialize score and health    初始化分数和血条
        self.score = 0
        self.lives = 6
        #Initialize the map    初始化地图
        self.mode = "map"
        self.planets = [
            {"x": 60, "y": 120, "r": 15, "level_id": 0},
            {"x": 120, "y": 200, "r": 15, "level_id": 1},
            {"x": 200, "y": 80, "r": 15, "level_id": 2}
        ]
        self.current_level = None
        self.game_over = False
        self.done = False
        self.key = jrandom.PRNGKey(0)
        #List of bullets     子弹列表
        self.bullets = {        #Store all active bullets: x, y, vx, vy        存放所有存活子弹，x, y, vx, vy,
            "x": jnp.zeros((0,)),
            "y": jnp.zeros((0,)),
            "vx": jnp.zeros((0,)),
            "vy": jnp.zeros((0,))
        }  
        self.bullets_speed = 5.0

        #Enemy bullet list    enemy子弹列表
        self.enemy_bullets = {
            "x": jnp.zeros((0,)),
            "y": jnp.zeros((0,)),
            "vx": jnp.zeros((0,)),
            "vy": jnp.zeros((0,))
        }
        self.enemy_fire_cooldown = 0
        self.enemy_fire_interval = 60
        self.enemy_bullet_speed = 5.0
        
        #Shooting cooldown    射击cd
        self.cooldown = 0
        self.cooldown_max = 5
        
        #Set window size     设置窗口大小
        self.screen_size = (WINDOW_WIDTH, WINDOW_HEIGHT)
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Gravitar")

    def draw_hud(self):
        #Background    背景
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, WINDOW_WIDTH, HUD_HEIGHT))

        #Score   分数
        font = pygame.font.SysFont("Arial", 14, bold=True)
        score_text = font.render(str(self.score), True, (255, 128, 255))
        text_rect = score_text.get_rect(center=(WINDOW_WIDTH //2, HUD_PADDING + 6))
        self.screen.blit(score_text, text_rect)

        #Health bar   血条
        total_width = self.lives * HUD_SHIP_SPACING
        start_x = (WINDOW_WIDTH - total_width) // 2
        for i in range(self.lives):
            x_offset = start_x + i * HUD_SHIP_SPACING
            y_offset = HUD_PADDING + 16
            pygame.draw.rect(self.screen, (135, 206, 250), (x_offset, y_offset, HUD_SHIP_WIDTH, HUD_SHIP_HEIGHT))

    def get_action_space(self) -> Tuple[int]:
        """
        Returns the action space of the environment.
        Returns: The action space of the environment as a tuple.
        """
        # raise NotImplementedError("Abstract method")
        return (18,)  #Supports actions numbered 0 to 17    支持编号0到17的动作
    
    def get_observation_space(self) -> Tuple[int]:
        """
        Returns the observation space of the environment.
        Returns: The observation space of the environment as a tuple.
        """
        #raise NotImplementedError("Abstract method")
        return (5,)
    
    def render(self, state: dict) -> Tuple[jnp.ndarray]:
        """
        Renders the environment state to a single image.
        Args:
            state: The environment state.

        Returns: A single image of the environment state.

        """
        #raise NotImplementedError("Abstract method")

        self.screen.fill((0, 0, 0))
        if self.mode == "map":
            #Draw planets    画星球
            for planet in self.planets:
                color = planet.get("color", (0, 255, 255))
                pygame.draw.circle(self.screen, color, (planet["x"], planet["y"]), planet["r"])
    # Render triangular spaceship      渲染三角形飞船
        # Spaceship center position    飞船中心位置
        cx, cy = float(state["x"]), float(state["y"])
        angle = float(state["angle"])
        size = 10  #Size of the spaceship tip    飞船箭头大小
        #Three triangle vertices, angle indicates direction      三角形三个顶点位置，angle指向朝向
        tip = (cx + math.cos(angle) * size, cy + math.sin(angle) * size)
        left = (cx + math.cos(angle + 2.5) * size, cy + math.sin(angle + 2.5) * size)
        right = (cx + math.cos(angle - 2.5) * size, cy + math.sin(angle - 2.5) * size)
        # Clear background   清空背景
        self.draw_hud()
        #Draw spaceship  画飞船
        pygame.draw.polygon(self.screen, (0, 255, 0), [tip, left, right])

        #Draw enemy   画enemy
        if self.mode == "level":
            for i in range(len(self.enemies["x"])):
                ex = self.enemies["x"][i]
                ey = self.enemies["y"][i]
                ew = self.enemies["w"][i]
                eh = self.enemies["h"][i]
                pygame.draw.rect(self.screen, (255, 0, 0), (int(ex), int(ey), int(ew), int(eh)))
        
        #Score  得分
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10)) 
            
        # Render bullets   渲染子弹
        for bx, by in zip(self.bullets["x"], self.bullets["y"]):
            if bx >= 0 and by >= 0:  # 排除 dummy bullets
                pygame.draw.circle(self.screen, (255, 255, 0), (int(bx), int(by)), 2)#yellow
        #Render enemy bullets    渲染enemy子弹
        for bx, by in zip(self.enemy_bullets["x"], self.enemy_bullets["y"]):
            if bx >= 0 and by >= 0:
                pygame.draw.circle(self.screen, (255, 100, 0), (int(bx), int(by)), 3)
        #gameover screen
        if hasattr(self, "done") and self.done:
            font = pygame.font.SysFont(None, 72)
            text = font.render("game over", True, (255, 0, 0))
            text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT // 2))
            self.screen.blit(text, text_rect)


        pygame.display.flip()
        pygame.time.delay(100)

        dummy_image = jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3))
        return (dummy_image,)

    def reset(self, key: jrandom.PRNGKey=None) -> Tuple[jnp.ndarray, dict]:
        self.lives = 6
        self.done = False
        self.game_over = False
        self.key = key if key is not None else jrandom.PRNGKey(0)
        return self.reset_map(self.key)

    def reset_map(self, key: jrandom.PRNGKey) -> Tuple[jnp.ndarray, dict]:
        self.mode = "map"
        self.current_level = None
        self.score = 0
        self.cooldown = 0
        self.enemy_fire_cooldown = 0
        self.enemies = []
        #添加星球
        self.planets = [
            {"x": 80, "y": 100, "r": 15, "level_id": 0, "color": (255, 100, 100)},
            {"x": 150, "y": 180, "r": 15, "level_id": 1, "color": (100, 255, 100)},
            {"x": 230, "y": 60, "r": 15, "level_id": 2, "color": (100, 100, 255)}
        ]

        state = {
            "x": jnp.array(WINDOW_WIDTH / 2),
            "y": jnp.array((WINDOW_HEIGHT + HUD_HEIGHT) / 2),
            "vx": jnp.array(0.0),
            "vy": jnp.array(0.0),
            "angle": jnp.array(-jnp.pi / 2)
        } 
        # 初始化 bullets
        self.bullets = {
            "x": jnp.zeros((0,)),
            "y": jnp.zeros((0,)),
            "vx": jnp.zeros((0,)),
            "vy": jnp.zeros((0,))
        }

        self.enemy_bullets = {
            "x": jnp.zeros((0,)),
            "y": jnp.zeros((0,)),
            "vx": jnp.zeros((0,)),
            "vy": jnp.zeros((0,))
        }

        obs = jnp.array([state["x"], state["y"], state["vx"], state["vy"], state["angle"]])

        return obs, state

    def reset_level(self, key: jrandom.PRNGKey, level_id: int) -> Tuple[jnp.ndarray, dict]:
        """
        Resets the environment to the initial state.
        Returns: The initial observation and the initial environment state.

        """
        self.mode = "level"
        self.current_level = level_id
        self.cooldown = 0
        self.enemy_fire_cooldown = 0
        self.score = 0
        self.done = False
        #Initialize spaceship state     初始化飞船状态
        state = {
            "x": jnp.array(WINDOW_WIDTH / 2),
            "y": jnp.array((WINDOW_HEIGHT + HUD_HEIGHT) / 2),
            "vx": jnp.array(0.0),
            "vy": jnp.array(0.0),
            "angle": jnp.array(-jnp.pi / 2)
        }

        self.bullets = {
            "x": jnp.zeros((0,)),
            "y": jnp.zeros((0,)),
            "vx": jnp.zeros((0,)),
            "vy": jnp.zeros((0,))
        }

        self.enemy_bullets = {
            "x": jnp.zeros((0,)),
            "y": jnp.zeros((0,)),
            "vx": jnp.zeros((0,)),
            "vy": jnp.zeros((0,))
        }

        #Enemy parameters    enemy参数
        num_enemies = 5
        enemy_w, enemy_h = 10, 10

        keys = jax.random.split(key, num_enemies * 3)

        enemy_x = []
        enemy_y = []
        enemy_w_arr = []
        enemy_h_arr = []
        enemy_vx = []

        for i in range(num_enemies):
            x = jrandom.randint(keys[i * 3 + 0], (), 60, WINDOW_WIDTH - 80)
            y = jrandom.randint(keys[i * 3 + 1], (), 60, WINDOW_HEIGHT - 100)
            move_flag = jrandom.uniform(keys[i * 3 + 2]) < 0.5
            vx = jnp.where(jrandom.bernoulli(keys[i * 3 + 2]), 1.5, -1.5) if move_flag else 0.0

            enemy_x.append(float(x))
            enemy_y.append(float(y))
            enemy_w_arr.append(float(enemy_w))
            enemy_h_arr.append(float(enemy_h))
            enemy_vx.append(float(vx))

        self.enemies = {
            "x": jnp.array(enemy_x),
            "y": jnp.array(enemy_y),
            "w": jnp.array(enemy_w_arr),
            "h": jnp.array(enemy_h_arr),
            "vx": jnp.array(enemy_vx)
        }
        
        #Output observation    输出observation
        obs = jnp.array([state["x"], state["y"],state["vx"],state["vy"],state["angle"]])

        return obs, state
        #raise NotImplementedError("Abstract method")
    
    def step(self, state: dict, action: int) -> Tuple[jnp.ndarray, dict, float, bool, dict]:
        """
        执行一个step：输入：当前state+哪个动作
        输出：环境观测，环境状态，reward，是否结束，环境信息
        dictionary用来存放x,y,vx,vy,angle      
        Takes a step in the environment.
        Args:
            state: The current environment state.
            action: The action to take.

        Returns: The observation, the new environment state, the reward, whether the state is terminal, and additional info.

        """
        #raise NotImplementedError("Abstract method")
        #Select map    选地图
        if self.mode == "map":
            # Unpack state  解包状态
            obs, new_state, reward, done, info, reset, level = step_core_map(
                state,
                action,
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                HUD_HEIGHT
            )

            x, y = float(new_state["x"]), float(new_state["y"])
            entered = False
            for planet in self.planets:
                dx = x - planet["x"]
                dy = y - planet["y"]
                dist2 = dx * dx + dy * dy
                if dist2 <= planet["r"] * planet["r"]:
                    entered = True
                    entered_level = planet["level_id"]
                    break

            if entered:
                self.key, subkey = jrandom.split(self.key)
                obs, new_state = self.reset_level(subkey, entered_level)
                reward = 0.0
                done = False
                info = {"entered_level": True}
            else:
                reward = -1.0
                done = False
                info = {}

            return obs, new_state, reward, done, info
        #In level mode    在level里
        obs, new_state, self.bullets, self.cooldown, self.enemies, self.enemy_bullets, self.enemy_fire_cooldown, self.key, reward, done, info = step_core_level(
            state,
            action,
            self.bullets,
            self.cooldown,
            self.enemies,
            self.enemy_bullets,
            self.enemy_fire_cooldown,
            self.key,
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            HUD_HEIGHT,
            self.cooldown_max,
            self.bullets_speed,
            self.enemy_bullet_speed,
            self.enemy_fire_interval
        )

        #when the ship is dead?   死亡逻辑: 飞船被打或撞
        if done:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else:
                self.key, subkey = jrandom.split(self.key)
                obs, new_state = self.reset_level(subkey, self.current_level)
                info["respawn"] = True
                done = False

        #how to success?    通关逻辑: 敌人全灭
        if (not done) and (len(self.enemies["x"]) == 0):
            self.key, subkey = jrandom.split(self.key)
            obs, new_state = self.reset_map(subkey)
            reward = 10.0
            info["level_cleared"] = True
            done = False

        self.done = done
        return obs, new_state, reward, done, info

if __name__ == "__main__":
    env = JaxGravitar()
    #Initialize random seed    初始化随机种子
    key = jax.random.PRNGKey(0)
    #Get initial observation and state            得到初始观测和状态obs, state
    obs, state = env.reset(key)
    #loop 循环
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if env.done:
            env.render(state)
            if keys[pygame.K_r]:
                key, subkey = jrandom.split(key)
                obs, state = env.reset(subkey)
                env.done = False
            else:
                pygame.time.wait(100)
            continue

        #Read keyboard        读键盘
        action = get_action_from_key()
        #Environment interaction              环境交互
        obs, state, reward, done, info = env.step(state, action)
        env.render(state)

        clock.tick(30)