import os
import jax
import jax.numpy as jnp
import pygame
import chex
from functools import partial
from jax.scipy import ndimage
import numpy as np
import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.core import JaxEnvironment
import jax.random as jrandom
from typing import Tuple, NamedTuple
import math
from typing import NamedTuple, Tuple, Dict, Any, Optional
import glob
from enum import IntEnum
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr
import jax.debug


"""
    Group member of the Gravitar: Xusong Yin, Elizaveta Kuznetsova, Li Dai
"""
FORCE_SPRITES = True
WORLD_SCALE = 3.0
# ========== Constants ==========
SPRITE_DIR = os.path.join(os.path.dirname(__file__), "sprites", "gravitar")
SCALE = 1
MAX_BULLETS = 64
MAX_ENEMIES = 16
# 18 discrete spaceship action constants
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

# HUD settings
HUD_HEIGHT = 24
MAX_LIVES = 3
HUD_PADDING = 5
HUD_SHIP_WIDTH = 10
HUD_SHIP_HEIGHT = 12
HUD_SHIP_SPACING = 12
# Pygame window dimensions
WINDOW_WIDTH = 160
WINDOW_HEIGHT = 210

SAUCER_SPAWN_DELAY_FRAMES = 60 * 3
SAUCER_RESPAWN_DELAY_FRAMES = 180 * 3
SAUCER_SPEED_MAP = jnp.float32(0.4) / WORLD_SCALE
SAUCER_SPEED_ARENA = jnp.float32(0.4) / WORLD_SCALE
SAUCER_RADIUS = jnp.float32(3.0)
SHIP_RADIUS = jnp.float32(2.0)
SAUCER_INIT_HP = jnp.int32(1)

PLAYER_BULLET_SPEED = jnp.float32(4) / WORLD_SCALE
SAUCER_EXPLOSION_FRAMES = jnp.int32(60)
SAUCER_FIRE_INTERVAL_FRAMES = jnp.int32(24)
SAUCER_BULLET_SPEED = jnp.float32(2) / WORLD_SCALE
ENEMY_EXPLOSION_FRAMES = jnp.int32(60)
UFO_HIT_RADIUS = jnp.float32(3.0)

SHIP_ANCHOR_X = None
SHIP_ANCHOR_Y = None
DEBUG_DRAW_SHIP_ORIGIN = True
PLAYER_FIRE_COOLDOWN_FRAMES = 30


def _jax_rotate(image, angle_deg, reshape=False, order=1, mode='constant', cval=0):
    angle_rad = jnp.deg2rad(angle_deg)
    height, width = image.shape[:2]
    center_y, center_x = jnp.float32(height) / jnp.float32(2), jnp.float32(width) / jnp.float32(2)
    y_coords, x_coords = jnp.mgrid[0:height, 0:width]
    y_centered, x_centered = y_coords - center_y, x_coords - center_x
    cos_angle, sin_angle = jnp.cos(-angle_rad), jnp.sin(-angle_rad)
    source_x = center_x + x_centered * cos_angle - y_centered * sin_angle
    source_y = center_y + x_centered * sin_angle + y_centered * cos_angle
    source_coords = jnp.stack([source_y, source_x])
    rotated_channels = []
    for i in range(image.shape[2]):
        rotated_channel = ndimage.map_coordinates(
            image[..., i], source_coords, order=order, mode=mode, cval=cval
        )
        rotated_channels.append(rotated_channel)
    return jnp.stack(rotated_channels, axis=-1).astype(image.dtype)


class SpriteIdx(IntEnum):
    SHIP_IDLE = 0
    SHIP_THRUST = 1
    SHIP_BULLET = 2
    ENEMY_BULLET = 3
    ENEMY_GREEN_BULLET = 4
    ENEMY_ORANGE = 5
    ENEMY_GREEN = 6
    ENEMY_SAUCER = 7
    ENEMY_UFO = 8
    ENEMY_CRASH = 9
    SAUCER_CRASH = 10
    SHIP_CRASH = 11
    FUEL_TANK = 12
    OBSTACLE = 13
    SPAWN_LOC = 14
    REACTOR = 15
    REACTOR_TERR = 16
    TERRANT1 = 17
    TERRANT2 = 18
    TERRANT3 = 19
    TERRANT4 = 20
    PLANET1 = 21
    PLANET2 = 22
    PLANET3 = 23
    PLANET4 = 24
    REACTOR_DEST = 25
    SCORE_UI = 26
    HP_UI = 27
    SHIP_THRUST_BACK = 28
    DIGIT_0 = 29
    DIGIT_1 = 30
    DIGIT_2 = 31
    DIGIT_3 = 32
    DIGIT_4 = 33
    DIGIT_5 = 34
    DIGIT_6 = 35
    DIGIT_7 = 36
    DIGIT_8 = 37
    DIGIT_9 = 38
    ENEMY_ORANGE_FLIPPED = 39

TERRANT_SCALE_OVERRIDES = {
    SpriteIdx.TERRANT2: 0.80,
}


LEVEL_LAYOUTS = {
    0: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (37, 44)},
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (82, 32)},
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (152, -3)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (22, 71)},
        {'type': SpriteIdx.FUEL_TANK, 'coords': (104, 60)},
    ],
    1: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (93, 19)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (52, 77)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (8, 36)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (11, 60)},
        {'type': SpriteIdx.FUEL_TANK, 'coords': (29, 0)},
    ],
    2: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (24, 38)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (43, 82)},
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (60, -2)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (108, 22)},
        {'type': SpriteIdx.FUEL_TANK, 'coords': (135, 68)},
    ],
    3: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (88, 93 - 114 + 48)},
        {'type': SpriteIdx.ENEMY_ORANGE_FLIPPED, 'coords': (116, 73 - 114 + 51)},
        {'type': SpriteIdx.ENEMY_ORANGE, 'coords': (122, 180 - 114 + 47)},
        {'type': SpriteIdx.ENEMY_GREEN, 'coords': (76, 126 - 114 + 47)},
        {'type': SpriteIdx.FUEL_TANK, 'coords': (19, 162 - 114 + 47)},
    ],
    4: [],
}

LEVEL_OFFSETS = {
    0: (0, 30),
    1: (0, 30),
    2: (0, 30),
    3: (0, 30),
    4: (0, 0),
}

SPRITE_TO_LEVEL_ID = {
    int(SpriteIdx.PLANET1): 0,
    int(SpriteIdx.PLANET2): 1,
    int(SpriteIdx.PLANET3): 2,
    int(SpriteIdx.PLANET4): 3,
    int(SpriteIdx.REACTOR): 4,
}

LEVEL_ID_TO_TERRAIN_SPRITE = {
    0: SpriteIdx.TERRANT1,
    1: SpriteIdx.TERRANT2,
    2: SpriteIdx.TERRANT3,
    3: SpriteIdx.TERRANT4,
    4: SpriteIdx.REACTOR_TERR,
}

LEVEL_ID_TO_BANK_IDX = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
}

class Bullets(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    alive: jnp.ndarray

class Enemies(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    w: jnp.ndarray
    h: jnp.ndarray
    vx: jnp.ndarray
    sprite_idx: jnp.ndarray
    death_timer: jnp.ndarray
    hp: jnp.ndarray

class ShipState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    angle: jnp.ndarray

class SaucerState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    hp: jnp.ndarray
    alive: jnp.ndarray
    death_timer: jnp.ndarray

class UFOState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    hp: jnp.ndarray
    alive: jnp.ndarray
    death_timer: jnp.ndarray

class FuelTanks(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    w: jnp.ndarray
    h: jnp.ndarray
    sprite_idx: jnp.ndarray
    active: jnp.ndarray

class EnvState(NamedTuple):
    mode: jnp.ndarray
    state: ShipState
    bullets: Bullets
    cooldown: jnp.ndarray
    enemies: Enemies
    fuel_tanks: FuelTanks
    enemy_bullets: Bullets
    fire_cooldown: jnp.ndarray
    key: jnp.ndarray
    key_alt: jnp.ndarray
    score: jnp.ndarray
    done: jnp.ndarray
    lives: jnp.ndarray
    crash_timer: jnp.ndarray
    planets_pi: jnp.ndarray
    planets_px: jnp.ndarray
    planets_py: jnp.ndarray
    planets_pr: jnp.ndarray
    planets_id: jnp.ndarray
    current_level: jnp.ndarray
    terrain_sprite_idx: jnp.ndarray
    terrain_mask: jnp.ndarray
    terrain_scale: jnp.ndarray
    terrain_offset: jnp.ndarray
    terrain_bank: jnp.ndarray
    terrain_bank_idx: jnp.ndarray
    respawn_shift_x: jnp.ndarray
    reactor_dest_active: jnp.ndarray
    reactor_dest_x: jnp.ndarray
    reactor_dest_y: jnp.ndarray
    reactor_dest_radius: jnp.ndarray
    mode_timer: jnp.ndarray
    saucer: SaucerState
    map_return_x: jnp.ndarray
    map_return_y: jnp.ndarray
    saucer_spawn_timer: jnp.ndarray
    ufo: UFOState
    ufo_used: jnp.ndarray
    ufo_home_x: jnp.ndarray
    ufo_home_y: jnp.ndarray
    ufo_bullets: Bullets
    level_offset: jnp.ndarray
    reactor_destroyed: jnp.ndarray
    planets_cleared_mask: jnp.ndarray


def make_empty_ufo() -> UFOState:
    f32 = jnp.float32; i32 = jnp.int32
    return UFOState(
        x=f32(0.0), y=f32(0.0),
        vx=f32(0.0), vy=f32(0.0),
        hp=i32(0),
        alive=jnp.array(False),
        death_timer=i32(0)
    )

def make_default_saucer() -> SaucerState:
    return SaucerState(
        x=jnp.float32(-999.0), y=jnp.float32(-999.0),
        vx=jnp.float32(0.0), vy=jnp.float32(0.0),
        hp=jnp.int32(0),
        alive=jnp.array(False),
        death_timer=jnp.int32(0),
    )

@jax.jit
def planet_to_bank_idx(psi: jnp.ndarray) -> jnp.ndarray:
    b = jnp.int32(0)
    b = jnp.where(psi == jnp.int32(int(SpriteIdx.PLANET1)), jnp.int32(1), b)
    b = jnp.where(psi == jnp.int32(int(SpriteIdx.PLANET2)), jnp.int32(2), b)
    b = jnp.where(psi == jnp.int32(int(SpriteIdx.PLANET3)), jnp.int32(3), b)
    b = jnp.where(psi == jnp.int32(int(SpriteIdx.PLANET4)), jnp.int32(4), b)
    b = jnp.where(psi == jnp.int32(int(SpriteIdx.REACTOR)), jnp.int32(5), b)
    return b

@jax.jit
def map_planet_to_terrant(planet_sprite_idx: jnp.ndarray) -> jnp.ndarray:
    P1 = jnp.int32(int(SpriteIdx.PLANET1))
    P2 = jnp.int32(int(SpriteIdx.PLANET2))
    P3 = jnp.int32(int(SpriteIdx.PLANET3))
    P4 = jnp.int32(int(SpriteIdx.PLANET4))
    PR = jnp.int32(int(SpriteIdx.REACTOR))

    T1 = jnp.int32(int(SpriteIdx.TERRANT1))
    T2 = jnp.int32(int(SpriteIdx.TERRANT2))
    T3 = jnp.int32(int(SpriteIdx.TERRANT3))
    T4 = jnp.int32(int(SpriteIdx.TERRANT4))
    TR = jnp.int32(int(SpriteIdx.REACTOR_TERR))

    invalid = jnp.int32(-1)
    out = invalid
    out = jnp.where(planet_sprite_idx == P1, T1, out)
    out = jnp.where(planet_sprite_idx == P2, T2, out)
    out = jnp.where(planet_sprite_idx == P3, T3, out)
    out = jnp.where(planet_sprite_idx == P4, T4, out)
    out = jnp.where(planet_sprite_idx == PR, TR, out)
    return out


def _opt(name_wo_ext: str):
    path = os.path.join(SPRITE_DIR, f"{name_wo_ext}.npy")
    if not os.path.exists(path):
        print(f"[sprite miss] {name_wo_ext}.npy")
        return None
    try:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if np.issubdtype(arr.dtype, np.floating):
            if 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0:
                arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype == np.uint8 and arr.max() <= 1:
            arr = (arr * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        rgb = arr[..., :3]
        alpha = (rgb.max(axis=-1) >= 1).astype(np.uint8) * 255
        rgba = np.dstack([rgb, alpha])
        surf = pygame.image.frombuffer(rgba.tobytes(), (rgba.shape[1], rgba.shape[0]), "RGBA").convert_alpha()
        if SCALE != 1:
            surf = pygame.transform.scale(surf, (surf.get_width() * SCALE, surf.get_height() * SCALE))
        return surf
    except Exception as e:
        print(f"[sprite error] {name_wo_ext}: {e}")
        return None

def _load_and_convert_sprites():
    pygame.init()
    pygame.display.set_mode((1, 1), pygame.NOFRAME)
    pygame_sprites = load_sprites_tuple()

    def surface_to_jax(surf):
        if surf is None: return None
        rgb = jnp.array(pygame.surfarray.pixels3d(surf)).transpose((1, 0, 2))
        alpha = jnp.array(pygame.surfarray.pixels_alpha(surf)).transpose((1, 0))
        return jnp.concatenate([rgb, alpha[..., None]], axis=-1).astype(jnp.uint8)

    jax_sprites = {}
    for i, surf in enumerate(pygame_sprites):
        if surf is not None:
            jax_sprites[i] = surface_to_jax(surf)
    return jax_sprites

def load_sprites_tuple() -> tuple:
    num_sprites = max(int(e) for e in SpriteIdx) + 1
    sprites = [None] * num_sprites
    sprite_map = {
        SpriteIdx.SHIP_IDLE: "spaceship",
        SpriteIdx.SHIP_THRUST: "ship_thrust",
        SpriteIdx.SHIP_BULLET: "ship_bullet",
        SpriteIdx.ENEMY_BULLET: "enemy_bullet",
        SpriteIdx.ENEMY_GREEN_BULLET: "enemy_green_bullet",
        SpriteIdx.ENEMY_ORANGE: "enemy_orange",
        SpriteIdx.ENEMY_GREEN: "enemy_green",
        SpriteIdx.ENEMY_SAUCER: "saucer",
        SpriteIdx.ENEMY_UFO: "UFO",
        SpriteIdx.ENEMY_CRASH: "enemy_crash",
        SpriteIdx.SAUCER_CRASH: "saucer_crash",
        SpriteIdx.SHIP_CRASH: "ship_crash",
        SpriteIdx.FUEL_TANK: "fuel_tank",
        SpriteIdx.OBSTACLE: "obstacle",
        SpriteIdx.SPAWN_LOC: "spawn_location",
        SpriteIdx.REACTOR: "reactor",
        SpriteIdx.REACTOR_TERR: "reactor_terrant",
        SpriteIdx.TERRANT1: "terrant1",
        SpriteIdx.TERRANT2: "terrant2",
        SpriteIdx.TERRANT3: "terrant_3",
        SpriteIdx.TERRANT4: "terrant_4",
        SpriteIdx.PLANET1: "planet1",
        SpriteIdx.PLANET2: "planet2",
        SpriteIdx.PLANET3: "planet3",
        SpriteIdx.PLANET4: "planet4",
        SpriteIdx.REACTOR_DEST: "reactor_destination",
        SpriteIdx.SCORE_UI: "score",
        SpriteIdx.HP_UI: "HP",
        SpriteIdx.SHIP_THRUST_BACK: "ship_thrust_back",
        SpriteIdx.DIGIT_0: "score_0",
        SpriteIdx.DIGIT_1: "score_1",
        SpriteIdx.DIGIT_2: "score_2",
        SpriteIdx.DIGIT_3: "score_3",
        SpriteIdx.DIGIT_4: "score_4",
        SpriteIdx.DIGIT_5: "score_5",
        SpriteIdx.DIGIT_6: "score_6",
        SpriteIdx.DIGIT_7: "score_7",
        SpriteIdx.DIGIT_8: "score_8",
        SpriteIdx.DIGIT_9: "score_9",
    }
    for idx_enum, name in sprite_map.items():
        sprites[int(idx_enum)] = _opt(name)
    orange_surf = sprites[int(SpriteIdx.ENEMY_ORANGE)]
    if orange_surf:
        sprites[int(SpriteIdx.ENEMY_ORANGE_FLIPPED)] = pygame.transform.flip(orange_surf, False, True)
    return tuple(sprites)

def create_empty_bullets_fixed(size: int) -> Bullets:
    return Bullets(
        x=jnp.zeros((size,), dtype=jnp.float32),
        y=jnp.zeros((size,), dtype=jnp.float32),
        vx=jnp.zeros((size,), dtype=jnp.float32),
        vy=jnp.zeros((size,), dtype=jnp.float32),
        alive=jnp.zeros((size,), dtype=bool)
    )

def create_empty_bullets_64():
    return create_empty_bullets_fixed(64)

def create_empty_bullets_16():
    return create_empty_bullets_fixed(16)

@jax.jit
def create_empty_enemies():
    return Enemies(
        x=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        y=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        w=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        h=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        vx=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        sprite_idx=jnp.full((MAX_ENEMIES,), -1, dtype=jnp.int32),
        death_timer=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),
        hp=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32)
    )

@jax.jit
def make_level_start_state(level_id: int) -> ShipState:
    START_Y = jnp.float32(30.0)
    x = jnp.array(WINDOW_WIDTH / 2, dtype=jnp.float32)
    y = jnp.array(START_Y, dtype=jnp.float32)
    angle = jnp.array(-jnp.pi / 2, dtype=jnp.float32)
    is_reactor = (jnp.asarray(level_id, dtype=jnp.int32) == 4)
    x = jnp.where(is_reactor, x - 60.0, x)
    return ShipState(x=x, y=y, vx=jnp.float32(0.0), vy=jnp.float32(0.0), angle=angle)

@jax.jit
def update_bullets(bullets: Bullets) -> Bullets:
    new_x = bullets.x + bullets.vx
    new_y = bullets.y + bullets.vy
    valid_x = (new_x >= 0) & (new_x <= WINDOW_WIDTH)
    valid_y = (new_y >= HUD_HEIGHT) & (new_y <= WINDOW_HEIGHT)
    valid = valid_x & valid_y & bullets.alive
    return Bullets(x=new_x, y=new_y, vx=bullets.vx, vy=bullets.vy, alive=valid)

@jax.jit
def fire_bullet(bullets: Bullets, ship_x, ship_y, ship_angle, bullet_speed):
    def add_bullet(_):
        idx = jnp.argmax(bullets.alive == False)
        new_vx = jnp.cos(ship_angle) * bullet_speed
        new_vy = jnp.sin(ship_angle) * bullet_speed
        return Bullets(
            x=bullets.x.at[idx].set(ship_x),
            y=bullets.y.at[idx].set(ship_y),
            vx=bullets.vx.at[idx].set(new_vx),
            vy=bullets.vy.at[idx].set(new_vy),
            alive=bullets.alive.at[idx].set(True)
        )
    can_fire = jnp.any(bullets.alive == False)
    return jax.lax.cond(can_fire, add_bullet, lambda b: b, bullets)

@jax.jit
def _fire_single_from_to(bullets: Bullets, sx, sy, tx, ty, speed=jnp.float32(0.7)) -> Bullets:
    dx = tx - sx
    dy = ty - sy
    d = jnp.maximum(jnp.sqrt(dx * dx + dy * dy), 1e-3)
    vx = speed * dx / d
    vy = speed * dy / d
    idx = jnp.argmax(bullets.alive == False)
    can_fire = jnp.any(bullets.alive == False)

    def add_bullet(b):
        return b._replace(
            x=b.x.at[idx].set(sx),
            y=b.y.at[idx].set(sy),
            vx=b.vx.at[idx].set(vx),
            vy=b.vy.at[idx].set(vy),
            alive=b.alive.at[idx].set(True)
        )
    return jax.lax.cond(can_fire, add_bullet, lambda b: b, bullets)


@jax.jit
def check_ship_enemy_collisions(ship: ShipState, enemies: Enemies, ship_radius: float) -> jnp.ndarray:
    enemy_half_w = enemies.w / 2
    enemy_half_h = enemies.h / 2
    delta_x = ship.x - enemies.x
    delta_y = ship.y - enemies.y
    clamped_x = jnp.clip(delta_x, -enemy_half_w, enemy_half_w)
    clamped_y = jnp.clip(delta_y, -enemy_half_h, enemy_half_h)
    closest_point_dx = delta_x - clamped_x
    closest_point_dy = delta_y - clamped_y
    distance_sq = closest_point_dx ** 2 + closest_point_dy ** 2
    collided_mask = (distance_sq < ship_radius ** 2) & (enemies.w > 0.0)
    return collided_mask

@jax.jit
def check_enemy_hit(bullets: Bullets, enemies: Enemies) -> Tuple[Bullets, Enemies, jnp.ndarray]:
    padding = 0.2
    ex1 = enemies.x - enemies.w / 2 - padding
    ex2 = enemies.x + enemies.w / 2 + padding
    ey1 = enemies.y - enemies.h / 2 - padding
    ey2 = enemies.y + enemies.h / 2
    bx = bullets.x[:, None]
    by = bullets.y[:, None]
    cond_x = (bx >= ex1) & (bx <= ex2)
    cond_y = (by >= ey1) & (by <= ey2)
    hit_matrix = cond_x & cond_y & bullets.alive[:, None] & (enemies.hp > 0)[:, None].T
    bullet_hit_mask = jnp.any(hit_matrix, axis=1)
    enemy_hit_mask = jnp.any(hit_matrix, axis=0)
    new_bullets = bullets._replace(alive=bullets.alive & ~bullet_hit_mask)
    hp_after_hit = enemies.hp - enemy_hit_mask.astype(jnp.int32)
    was_alive = enemies.hp > 0
    is_dead_now = hp_after_hit <= 0
    just_died_mask = was_alive & is_dead_now
    new_enemies = enemies._replace(
        hp=hp_after_hit,
        death_timer=jnp.where(just_died_mask, ENEMY_EXPLOSION_FRAMES, enemies.death_timer)
    )
    return new_bullets, new_enemies, just_died_mask

@jax.jit
def terrain_hit(env_state: EnvState, x: jnp.ndarray, y: jnp.ndarray, radius=jnp.float32(0.3)) -> jnp.ndarray:
    H, W = env_state.terrain_bank.shape[1], env_state.terrain_bank.shape[2]
    xi = jnp.clip(jnp.round(x).astype(jnp.int32), 0, W - 1)
    yi = jnp.clip(jnp.round(y).astype(jnp.int32), 0, H - 1)
    RMAX = 16
    dx = jnp.arange(-RMAX, RMAX + 1, dtype=jnp.int32)
    dy = jnp.arange(-RMAX, RMAX + 1, dtype=jnp.int32)
    xs = jnp.clip(xi + dx, 0, W - 1)
    ys = jnp.clip(yi + dy, 0, H - 1)
    bi = jnp.clip(env_state.terrain_bank_idx, 0, env_state.terrain_bank.shape[0] - 1)
    page = env_state.terrain_bank[bi]
    patch = page[ys[:, None], xs[None, :]]
    dxf, dyf = dx.astype(jnp.float32), dy.astype(jnp.float32)
    dist2 = dyf[:, None] ** 2 + dxf[None, :] ** 2
    r_eff = jnp.minimum(jnp.float32(radius), jnp.float32(RMAX))
    mask = dist2 <= (r_eff ** 2)
    is_not_black = jnp.sum(patch, axis=-1) > 0
    return jnp.any(is_not_black & mask)

@jax.jit
def consume_ship_hits(state, bullets, hitbox_size):
    hs = jnp.asarray(hitbox_size, dtype=jnp.float32)
    eff_r = hs + jnp.float32(0.04)
    px0 = bullets.x - bullets.vx
    py0 = bullets.y - bullets.vy
    dx = bullets.vx
    dy = bullets.vy
    a = dx * dx + dy * dy + 1e-6
    t = jnp.clip(-(((px0 - state.x) * dx + (py0 - state.y) * dy) / a), 0.0, 1.0)
    qx = px0 + t * dx
    qy = py0 + t * dy
    d2 = (qx - state.x) ** 2 + (qy - state.y) ** 2
    hit_mask = bullets.alive & (d2 <= (eff_r * eff_r))
    any_hit = jnp.any(hit_mask)
    new_bullets = bullets._replace(alive=bullets.alive & ~hit_mask)
    return new_bullets, any_hit

@jax.jit
def ship_step(state: ShipState, action: int, window_size: tuple[int, int], hud_height: int) -> ShipState:
    rotation_speed = 0.2 / WORLD_SCALE
    thrust_power = 0.03 / WORLD_SCALE
    gravity = 0.008 / WORLD_SCALE
    max_speed = 1.0 / WORLD_SCALE
    bounce_damping = 0.5
    vx, vy = state.vx, state.vy
    rotate_right_actions = jnp.array([3, 6, 8, 11, 14, 16])
    rotate_left_actions = jnp.array([4, 7, 9, 12, 15, 17])
    right = jnp.isin(action, rotate_right_actions)
    left = jnp.isin(action, rotate_left_actions)
    angle = jnp.where(left, state.angle - rotation_speed, state.angle)
    angle = jnp.where(right, angle + rotation_speed, angle)
    thrust_actions = jnp.array([2, 6, 7, 10, 14, 15])
    down_thrust_actions = jnp.array([5, 8, 9, 13, 16, 17])
    thrust_pressed = jnp.isin(action, thrust_actions)
    down_pressed = jnp.isin(action, down_thrust_actions)
    vx = jnp.where(thrust_pressed, vx + jnp.cos(angle) * thrust_power, vx)
    vy = jnp.where(thrust_pressed, vy + jnp.sin(angle) * thrust_power, vy)
    vx = jnp.where(down_pressed, vx - jnp.cos(angle) * thrust_power, vx)
    vy = jnp.where(down_pressed, vy - jnp.sin(angle) * thrust_power, vy)
    vy += gravity
    speed_sq = vx ** 2 + vy ** 2
    def cap_velocity(v_tuple):
        v_x, v_y, spd_sq = v_tuple
        speed = jnp.sqrt(spd_sq)
        scale = max_speed / speed
        return v_x * scale, v_y * scale
    vx, vy = jax.lax.cond(speed_sq > max_speed ** 2, cap_velocity, lambda v: (v[0], v[1]), (vx, vy, speed_sq))
    next_x, next_y = state.x + vx, state.y + vy
    ship_half_size = SHIP_RADIUS
    window_width, window_height = window_size
    hit_left, hit_right = next_x < ship_half_size, next_x > window_width - ship_half_size
    hit_top, hit_bottom = next_y < hud_height + ship_half_size, next_y > window_height - ship_half_size
    final_vx = jnp.where(hit_left | hit_right, -vx * bounce_damping, vx)
    final_vy = jnp.where(hit_top | hit_bottom, -vy * bounce_damping, vy)
    final_x = jnp.clip(next_x, ship_half_size, window_width - ship_half_size)
    final_y = jnp.clip(next_y, hud_height + ship_half_size, window_height - ship_half_size)
    normalized_angle = (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return ShipState(x=final_x, y=final_y, vx=final_vx, vy=final_vy, angle=normalized_angle)

@jax.jit
def _get_reactor_center(px, py, pi) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    REACTOR = jnp.int32(int(SpriteIdx.REACTOR))
    mask = (pi == REACTOR)
    any_reactor = jnp.any(mask)
    idx = jnp.argmax(mask.astype(jnp.int32))
    rx = jax.lax.cond(any_reactor, lambda _: px[idx], lambda _: jnp.float32(WINDOW_WIDTH * 0.18), operand=None)
    ry = jax.lax.cond(any_reactor, lambda _: py[idx], lambda _: jnp.float32(WINDOW_HEIGHT * 0.43), operand=None)
    return rx, ry, any_reactor

@jax.jit
def _spawn_saucer_at(x, y, towards_x, towards_y, speed=jnp.float32(0.8)) -> SaucerState:
    dx = towards_x - x
    dy = towards_y - y
    d = jnp.maximum(jnp.sqrt(dx * dx + dy * dy), 1e-3)
    vx = speed * dx / d
    vy = speed * dy / d
    return SaucerState(
        x=jnp.float32(x), y=jnp.float32(y),
        vx=vx, vy=vy,
        hp=SAUCER_INIT_HP, alive=jnp.array(True),
        death_timer=jnp.int32(0),
    )

@jax.jit
def _update_saucer_seek(s: SaucerState, target_x, target_y, speed) -> SaucerState:
    dx = target_x - s.x
    dy = target_y - s.y
    d = jnp.maximum(jnp.sqrt(dx * dx + dy * dy), 1e-3)
    vx = speed * dx / d
    vy = speed * dy / d
    return s._replace(x=s.x + vx, y=s.y + vy, vx=vx, vy=vy)

@jax.jit
def _bullets_hit_saucer(bullets: Bullets, sauc: SaucerState) -> Tuple[Bullets, jnp.ndarray]:
    eff_r = SAUCER_RADIUS
    px0 = bullets.x - bullets.vx
    py0 = bullets.y - bullets.vy
    dx = bullets.vx
    dy = bullets.vy
    a = dx * dx + dy * dy + 1e-6
    t = jnp.clip(-(((px0 - sauc.x) * dx + (py0 - sauc.y) * dy) / a), 0.0, 1.0)
    qx = px0 + t * dx
    qy = py0 + t * dy
    d2 = (qx - sauc.x) ** 2 + (qy - sauc.y) ** 2
    hit_mask = bullets.alive & sauc.alive & (d2 <= (eff_r * eff_r))
    any_hit = jnp.any(hit_mask)
    new_bullets = bullets._replace(alive=bullets.alive & ~hit_mask)
    return new_bullets, any_hit

@jax.jit
def _circle_hit(ax, ay, ar, bx, by, br) -> jnp.ndarray:
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) <= (ar + br) * (ar + br)

@jax.jit
def step_map(env_state: EnvState, action: int) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any], jnp.ndarray, jnp.ndarray]:
    was_crashing = env_state.crash_timer > 0
    ship_state_before_move = env_state.state._replace(
        vx=jnp.where(was_crashing, 0.0, env_state.state.vx),
        vy=jnp.where(was_crashing, 0.0, env_state.state.vy)
    )
    actual_action = jnp.where(was_crashing, NOOP, action)
    ship_after_move = ship_step(ship_state_before_move, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT)
    can_fire = jnp.isin(action, jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])) & (env_state.cooldown == 0)
    bullets = jax.lax.cond(
        can_fire,
        lambda b: fire_bullet(b, ship_after_move.x, ship_after_move.y, ship_after_move.angle, PLAYER_BULLET_SPEED),
        lambda b: b,
        env_state.bullets
    )
    bullets = update_bullets(bullets)
    cooldown = jnp.where(can_fire, PLAYER_FIRE_COOLDOWN_FRAMES, jnp.maximum(env_state.cooldown - 1, 0))
    new_env = env_state._replace(state=ship_after_move, bullets=bullets, cooldown=cooldown)
    saucer = new_env.saucer
    timer = new_env.saucer_spawn_timer
    should_tick_timer = (new_env.mode == 0) & (~saucer.alive) & (saucer.death_timer == 0)
    timer = jnp.where(should_tick_timer, jnp.maximum(timer - 1, 0), timer)
    rx, ry, has_reactor = _get_reactor_center(new_env.planets_px, new_env.planets_py, new_env.planets_pi)
    should_spawn = (timer == 0) & (~saucer.alive) & has_reactor
    saucer = jax.lax.cond(should_spawn, lambda: _spawn_saucer_at(rx, ry, new_env.state.x, new_env.state.y, SAUCER_SPEED_MAP), lambda: saucer)
    timer = jnp.where(should_spawn, jnp.int32(99999), timer)
    saucer_after_move = jax.lax.cond(saucer.alive, lambda s: _update_saucer_seek(s, new_env.state.x, new_env.state.y, SAUCER_SPEED_MAP), lambda s: s, saucer)
    bullets_after_hit, hit_any_bullet = _bullets_hit_saucer(new_env.bullets, saucer_after_move)
    sauc_after_hp = saucer_after_move._replace(hp=saucer_after_move.hp - jnp.where(hit_any_bullet, 1, 0))
    just_died = (saucer_after_move.hp > 0) & (sauc_after_hp.hp <= 0) & saucer_after_move.alive
    timer = jnp.where(just_died, SAUCER_RESPAWN_DELAY_FRAMES, timer)
    sauc_final = sauc_after_hp._replace(
        alive=sauc_after_hp.hp > 0,
        death_timer=jnp.where(just_died, SAUCER_EXPLOSION_FRAMES, jnp.maximum(saucer_after_move.death_timer - 1, 0))
    )
    mode_timer = jnp.where(new_env.mode == 0, new_env.mode_timer + 1, jnp.int32(0))
    can_shoot_saucer = sauc_final.alive & (jnp.sum(new_env.enemy_bullets.alive) < 1) & ((mode_timer % SAUCER_FIRE_INTERVAL_FRAMES) == 0)
    enemy_bullets = jax.lax.cond(can_shoot_saucer, lambda eb: _fire_single_from_to(eb, sauc_final.x, sauc_final.y, new_env.state.x, new_env.state.y, SAUCER_BULLET_SPEED), lambda eb: eb, new_env.enemy_bullets)
    enemy_bullets = update_bullets(enemy_bullets)
    new_env = new_env._replace(bullets=bullets_after_hit, saucer=sauc_final, saucer_spawn_timer=timer, enemy_bullets=enemy_bullets, mode_timer=mode_timer)
    enemy_bullets_after_hit, hit_ship_by_bullet = consume_ship_hits(new_env.state, new_env.enemy_bullets, SHIP_RADIUS)
    new_env = new_env._replace(enemy_bullets=enemy_bullets_after_hit)
    px, py, pr, pi, pid = new_env.planets_px, new_env.planets_py, new_env.planets_pr, new_env.planets_pi, new_env.planets_id
    dx, dy = px - new_env.state.x, py - new_env.state.y
    dist2 = dx * dx + dy * dy
    hit_obstacle = jnp.any((pi == SpriteIdx.OBSTACLE) & (dist2 <= (pr + SHIP_RADIUS) ** 2))
    ship_should_crash = hit_ship_by_bullet | hit_obstacle
    start_crash = ship_should_crash & ~was_crashing
    crash_timer_next = jnp.where(start_crash, 30, jnp.maximum(new_env.crash_timer - 1, 0))
    new_env = new_env._replace(crash_timer=crash_timer_next)
    is_crashing_now = new_env.crash_timer > 0
    allowed = jnp.any(jnp.stack([pi == SpriteIdx.PLANET1, pi == SpriteIdx.PLANET2, pi == SpriteIdx.PLANET3, pi == SpriteIdx.PLANET4, pi == SpriteIdx.REACTOR], 0), axis=0)
    allowed &= ~new_env.planets_cleared_mask
    is_reactor_and_destroyed = (pi == jnp.int32(int(SpriteIdx.REACTOR))) & new_env.reactor_destroyed
    allowed &= ~is_reactor_and_destroyed
    hit_planet = allowed & (dist2 <= (pr * 0.85 + SHIP_RADIUS) ** 2)
    can_enter_planet = jnp.any(hit_planet) & ~is_crashing_now
    hit_to_arena = sauc_final.alive & _circle_hit(new_env.state.x, new_env.state.y, SHIP_RADIUS, sauc_final.x, sauc_final.y, SAUCER_RADIUS) & ~is_crashing_now
    def _enter_arena(env):
        W, H = jnp.float32(WINDOW_WIDTH), jnp.float32(WINDOW_HEIGHT)
        return env._replace(
            mode=jnp.int32(2), mode_timer=jnp.int32(0),
            state=env.state._replace(x=W * jnp.float32(0.20), y=H * jnp.float32(0.50), vx=jnp.float32(0.0), vy=jnp.float32(0.0)),
            saucer=sauc_final._replace(x=W * jnp.float32(0.80), y=H * jnp.float32(0.50), vx=-SAUCER_SPEED_ARENA, vy=jnp.float32(0.0), hp=SAUCER_INIT_HP, alive=jnp.array(True, dtype=jnp.bool_), death_timer=jnp.int32(0)),
            map_return_x=env.state.x, map_return_y=env.state.y
        )
    new_env = jax.lax.cond(hit_to_arena, _enter_arena, lambda e: e, new_env)
    reset_signal_from_crash = (env_state.crash_timer > 0) & (crash_timer_next == 0)
    hit_idx = jnp.argmax(hit_planet.astype(jnp.int32))
    level_id = jax.lax.cond(can_enter_planet, lambda: pid[hit_idx], lambda: -1)
    should_reset = can_enter_planet | reset_signal_from_crash
    final_level_id = jnp.where(reset_signal_from_crash, jnp.int32(-2), level_id)
    obs = jnp.array([new_env.state.x, new_env.state.y, new_env.state.vx, new_env.state.vy, new_env.state.angle], dtype=jnp.float32)
    reward_saucer = jnp.where(just_died, jnp.float32(300.0), jnp.float32(0.0))
    reward_penalty = jnp.where(start_crash & ~hit_obstacle, jnp.float32(-10.0), jnp.float32(0.0))
    reward = reward_saucer + reward_penalty
    info = {
        "crash": start_crash,
        "hit_by_bullet": hit_ship_by_bullet,
        "reactor_crash_exit": jnp.array(False, dtype=jnp.bool_),
        "all_rewards": {
            "reward_enemies": jnp.float32(0.0), "reward_reactor": jnp.float32(0.0),
            "reward_ufo": jnp.float32(0.0), "reward_tanks": jnp.float32(0.0),
            "reward_saucer_kill": reward_saucer, "reward_penalty": reward_penalty,
        }
    }
    new_env = new_env._replace(score=new_env.score + reward)
    return obs, new_env, jnp.float32(reward), jnp.array(False, dtype=jnp.bool_), info, should_reset, final_level_id

@jax.jit
def _bullets_hit_ufo(bullets: Bullets, ufo) -> Tuple[Bullets, jnp.ndarray]:
    eff_r = SAUCER_RADIUS
    px0 = bullets.x - bullets.vx
    py0 = bullets.y - bullets.vy
    dx, dy = bullets.vx, bullets.vy
    a = dx * dx + dy * dy + 1e-6
    t = jnp.clip(-(((px0 - ufo.x) * dx + (py0 - ufo.y) * dy) / a), 0.0, 1.0)
    qx, qy = px0 + t * dx, py0 + t * dy
    d2 = (qx - ufo.x) ** 2 + (qy - ufo.y) ** 2
    hit_mask = bullets.alive & ufo.alive & (d2 <= (eff_r * eff_r))
    any_hit = jnp.any(hit_mask)
    new_bullets = bullets._replace(alive=bullets.alive & ~hit_mask)
    return new_bullets, any_hit

@jax.jit
def _bullets_hit_terrain(env_state: EnvState, bullets: Bullets) -> Bullets:
    H, W = env_state.terrain_bank.shape[1], env_state.terrain_bank.shape[2]
    bank_idx = jnp.clip(env_state.terrain_bank_idx, jnp.int32(0), env_state.terrain_bank.shape[0] - jnp.int32(1))
    terrain_map = env_state.terrain_bank[bank_idx]
    xi = jnp.clip(jnp.round(bullets.x).astype(jnp.int32), jnp.int32(0), jnp.int32(W) - jnp.int32(1))
    yi = jnp.clip(jnp.round(bullets.y).astype(jnp.int32), jnp.int32(0), jnp.int32(H) - jnp.int32(1))
    pixel_colors = terrain_map[yi, xi]
    hit_terrain_mask = jnp.sum(pixel_colors, axis=-1) > jnp.int32(0)
    final_hit_mask = bullets.alive & hit_terrain_mask
    return bullets._replace(alive=bullets.alive & ~final_hit_mask)

@jax.jit
def _ufo_ground_safe_y_at(terrain_bank, terrain_bank_idx, xf):
    W, H = jnp.float32(WINDOW_WIDTH), jnp.float32(WINDOW_HEIGHT)
    bank_idx = jnp.clip(terrain_bank_idx, jnp.int32(0), terrain_bank.shape[0] - jnp.int32(1))
    terrain_page = terrain_bank[bank_idx]
    col_x = jnp.clip(xf.astype(jnp.int32), jnp.int32(0), jnp.int32(W) - jnp.int32(1))
    is_ground_in_col = jnp.sum(terrain_page[:, col_x], axis=-1) > jnp.int32(0)
    y_indices = jnp.arange(WINDOW_HEIGHT, dtype=jnp.int32)
    ground_indices = jnp.where(is_ground_in_col, y_indices, jnp.int32(H))
    ground_y = jnp.min(ground_indices)
    return jnp.float32(ground_y) - jnp.float32(20.0)

@jax.jit
def _ufo_alive_step(e, ship, bullets):
    u = e.ufo
    LEFT_BOUNDARY, RIGHT_BOUNDARY = jnp.float32(8.0), jnp.float32(WINDOW_WIDTH) - jnp.float32(8.0)
    MIN_ALTITUDE = jnp.float32(HUD_HEIGHT) + jnp.float32(20.0)
    VERTICAL_ADJUST_SPEED = jnp.float32(0.5) / WORLD_SCALE
    next_x = u.x + u.vx
    hit_horizontal_boundary = (next_x <= LEFT_BOUNDARY) | (next_x >= RIGHT_BOUNDARY)
    final_vx = jnp.where(hit_horizontal_boundary, -u.vx, u.vx)
    final_x = jnp.clip(u.x + final_vx, LEFT_BOUNDARY, RIGHT_BOUNDARY)
    safe_y_here = _ufo_ground_safe_y_at(e.terrain_bank, e.terrain_bank_idx, final_x)
    target_y = jnp.maximum(safe_y_here - jnp.float32(20.0), MIN_ALTITUDE)
    y_difference = target_y - u.y
    final_vy = jnp.clip(y_difference, -VERTICAL_ADJUST_SPEED, VERTICAL_ADJUST_SPEED)
    final_y = jnp.clip(u.y + final_vy, MIN_ALTITUDE, jnp.float32(WINDOW_HEIGHT) - jnp.float32(20.0))
    u_after_move = u._replace(x=final_x, y=final_y, vx=final_vx, vy=final_vy)
    hit_by_ship = _circle_hit(ship.x, ship.y, SHIP_RADIUS, u_after_move.x, u_after_move.y, UFO_HIT_RADIUS) & u_after_move.alive
    bullets_after_hit, hit_by_bullet = _bullets_hit_ufo(bullets, u_after_move)
    hp_after_hit = u_after_move.hp - jnp.where(hit_by_bullet, jnp.int32(1), jnp.int32(0))
    was_alive = u_after_move.alive
    is_dead_now = hit_by_ship | (hp_after_hit <= jnp.int32(0))
    just_died = was_alive & is_dead_now
    u_final = u_after_move._replace(
        hp=hp_after_hit, alive=was_alive & ~is_dead_now,
        death_timer=jnp.where(just_died, jnp.int32(SAUCER_EXPLOSION_FRAMES), u_after_move.death_timer)
    )
    FIRE_COOLDOWN = jnp.int32(45)
    no_ufo_bullet_alive = ~jnp.any(e.ufo_bullets.alive)
    cd_ok = (e.mode_timer % FIRE_COOLDOWN) == jnp.int32(0)
    can_shoot = u_final.alive & no_ufo_bullet_alive & cd_ok
    def _fire_one(bul):
        ufo_bullet_speed = jnp.float32(2.2) / WORLD_SCALE
        return _fire_single_from_to(bul, u_final.x, u_final.y, ship.x, ship.y, ufo_bullet_speed)
    ufo_bullets = jax.lax.cond(can_shoot, _fire_one, lambda b: b, e.ufo_bullets)
    return e._replace(ufo=u_final, bullets=bullets_after_hit, ufo_bullets=ufo_bullets), just_died

@jax.jit
def _ufo_dead_step(e, ship, bullets):
    u = e.ufo
    u2 = u._replace(death_timer=jnp.maximum(u.death_timer - jnp.int32(1), jnp.int32(0)))
    return e._replace(ufo=u2, ufo_bullets=create_empty_bullets_16(), bullets=bullets), jnp.array(False, dtype=jnp.bool_)

@jax.jit
def _update_ufo(env: EnvState, ship: ShipState, bullets: Bullets) -> Tuple[EnvState, jnp.ndarray]:
    return jax.lax.cond(env.ufo.alive, _ufo_alive_step, _ufo_dead_step, env, ship, bullets)

@jax.jit
def _step_level_core(env_state: EnvState, action: int) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any], jnp.ndarray, jnp.ndarray]:
    def _spawn_ufo_once(env):
        W, H = jnp.float32(WINDOW_WIDTH), jnp.float32(WINDOW_HEIGHT)
        b = env.terrain_bank_idx
        is_born_on_left = (b == jnp.int32(2)) | (b == jnp.int32(4))
        x0 = jnp.where(is_born_on_left, jnp.float32(0.15) * W, jnp.float32(0.85) * W)
        vx = jnp.where(is_born_on_left, jnp.float32(0.6) / WORLD_SCALE, jnp.float32(-0.6) / WORLD_SCALE)
        bank_idx = jnp.clip(env.terrain_bank_idx, jnp.int32(0), env.terrain_bank.shape[0] - jnp.int32(1))
        terrain_page = env.terrain_bank[bank_idx]
        is_ground = jnp.sum(terrain_page, axis=-1) > jnp.int32(0)
        y_indices = jnp.arange(WINDOW_HEIGHT, dtype=jnp.int32)[:, None]
        ground_indices = jnp.where(is_ground, y_indices, jnp.int32(H))
        highest_point_on_map = jnp.min(ground_indices)
        safe_y = jnp.float32(highest_point_on_map) - jnp.float32(10.0)
        final_y0 = jnp.clip(safe_y, jnp.float32(HUD_HEIGHT) + jnp.float32(20.0), H - jnp.float32(20.0))
        return env._replace(
            ufo=UFOState(x=x0, y=final_y0, vx=vx, vy=jnp.float32(0.0), hp=jnp.int32(1), alive=jnp.array(True, dtype=jnp.bool_), death_timer=jnp.int32(0)),
            ufo_used=jnp.array(True, dtype=jnp.bool_), ufo_home_x=x0, ufo_home_y=final_y0, ufo_bullets=create_empty_bullets_16(),
        )
    can_spawn_ufo = (env_state.mode == jnp.int32(1)) & ~env_state.ufo_used & (env_state.terrain_bank_idx != jnp.int32(5))
    state_after_spawn = jax.lax.cond(can_spawn_ufo, _spawn_ufo_once, lambda e: e, env_state)
    was_crashing = state_after_spawn.crash_timer > 0
    ship_state_before_move = state_after_spawn.state._replace(
        vx=jnp.where(was_crashing, 0.0, state_after_spawn.state.vx),
        vy=jnp.where(was_crashing, 0.0, state_after_spawn.state.vy)
    )
    actual_action = jnp.where(was_crashing, NOOP, action)
    ship_after_move = ship_step(ship_state_before_move, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT)
    can_fire_player = jnp.isin(action, jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])) & (state_after_spawn.cooldown == 0)
    bullets = jax.lax.cond(can_fire_player, lambda b: fire_bullet(b, ship_after_move.x, ship_after_move.y, ship_after_move.angle, PLAYER_BULLET_SPEED), lambda b: b, state_after_spawn.bullets)
    cooldown = jnp.where(can_fire_player, PLAYER_FIRE_COOLDOWN_FRAMES, jnp.maximum(state_after_spawn.cooldown - 1, 0))
    tanks = state_after_spawn.fuel_tanks
    collision_mask = tanks.active & (ship_after_move.x + SHIP_RADIUS > tanks.x - tanks.w / 2) & (ship_after_move.x - SHIP_RADIUS < tanks.x + tanks.w / 2) & (ship_after_move.y + SHIP_RADIUS > tanks.y - tanks.h / 2) & (ship_after_move.y - SHIP_RADIUS < tanks.y + tanks.h / 2)
    new_fuel_tanks = tanks._replace(active=tanks.active & ~collision_mask)
    score_from_tanks = jnp.sum(collision_mask) * 150.0
    state_after_ufo, ufo_just_died = _update_ufo(state_after_spawn, ship_after_move, bullets)
    bullets, ufo_bullets = state_after_ufo.bullets, state_after_ufo.ufo_bullets
    enemies = state_after_ufo.enemies
    is_exploding = enemies.death_timer > 0
    enemies = enemies._replace(
        death_timer=jnp.maximum(enemies.death_timer - 1, 0),
        w=jnp.where(is_exploding & (enemies.death_timer == 1), 0.0, enemies.w),
        h=jnp.where(is_exploding & (enemies.death_timer == 1), 0.0, enemies.h)
    )
    current_fire_cooldown, current_key, current_enemy_bullets = state_after_ufo.fire_cooldown, state_after_ufo.key, state_after_ufo.enemy_bullets
    can_fire_globally = jnp.sum(current_enemy_bullets.alive) < 1
    is_turret = (enemies.sprite_idx == int(SpriteIdx.ENEMY_ORANGE)) | (enemies.sprite_idx == int(SpriteIdx.ENEMY_GREEN)) | (enemies.sprite_idx == int(SpriteIdx.ENEMY_ORANGE_FLIPPED))
    turrets_ready_mask = (enemies.hp > 0) & (current_fire_cooldown == 0) & is_turret
    should_fire_mask = turrets_ready_mask & can_fire_globally
    any_turret_firing = jnp.any(should_fire_mask)
    next_frame_cooldown = jnp.maximum(current_fire_cooldown - 1, 0)
    next_frame_cooldown = jnp.where(should_fire_mask, 60, next_frame_cooldown)
    def _generate_bullets(_):
        ex_center, ey_center = enemies.x + enemies.w / 2.0, enemies.y - enemies.h / 2.0
        dx, dy = ship_after_move.x - ex_center, ship_after_move.y - ey_center
        dist = jnp.maximum(jnp.sqrt(dx ** 2 + dy ** 2), 1e-3)
        vx, vy = dx / dist * (2.0 * 0.2 * 0.3), dy / dist * (2.0 * 0.2 * 0.3)
        return Bullets(
            x=jnp.where(should_fire_mask, ex_center, -1.0),
            y=jnp.where(should_fire_mask, ey_center, -1.0),
            vx=jnp.where(should_fire_mask, vx, 0.0),
            vy=jnp.where(should_fire_mask, vy, 0.0),
            alive=should_fire_mask
        )
    new_enemy_bullets = jax.lax.cond(any_turret_firing, _generate_bullets, lambda _: create_empty_bullets_16(), None)
    enemy_bullets = _fire_single_from_to(current_enemy_bullets, -1, -1, -1, -1, 0) # Placeholder to match structures
    enemy_bullets = jax.lax.cond(any_turret_firing, lambda: _fire_single_from_to(current_enemy_bullets, new_enemy_bullets.x[jnp.argmax(should_fire_mask)], new_enemy_bullets.y[jnp.argmax(should_fire_mask)], ship_after_move.x, ship_after_move.y), lambda: current_enemy_bullets)

    bullets, enemy_bullets, ufo_bullets = update_bullets(bullets), update_bullets(enemy_bullets), update_bullets(ufo_bullets)
    bullets, enemy_bullets, ufo_bullets = _bullets_hit_terrain(state_after_ufo, bullets), _bullets_hit_terrain(state_after_ufo, enemy_bullets), _bullets_hit_terrain(state_after_ufo, ufo_bullets)
    bullets, enemies, just_killed_mask = check_enemy_hit(bullets, enemies)
    hit_enemy_mask = check_ship_enemy_collisions(ship_after_move, enemies, SHIP_RADIUS)
    enemies = enemies._replace(death_timer=jnp.where(hit_enemy_mask, ENEMY_EXPLOSION_FRAMES, enemies.death_timer))
    enemy_bullets, hit_by_enemy_bullet = consume_ship_hits(ship_after_move, enemy_bullets, SHIP_RADIUS)
    ufo_bullets, hit_by_ufo_bullet = consume_ship_hits(ship_after_move, ufo_bullets, SHIP_RADIUS)
    hit_terr = terrain_hit(state_after_ufo, ship_after_move.x, ship_after_move.y, 2.0)
    hit_enemy_types = jnp.where(hit_enemy_mask, enemies.sprite_idx, -1)
    crashed_on_turret = jnp.any((hit_enemy_types == int(SpriteIdx.ENEMY_ORANGE)) | (hit_enemy_types == int(SpriteIdx.ENEMY_GREEN)))
    dead = crashed_on_turret | hit_by_enemy_bullet | hit_by_ufo_bullet | hit_terr
    is_in_reactor = (state_after_ufo.current_level == 4)
    dx_dest, dy_dest = ship_after_move.x - env_state.reactor_dest_x, ship_after_move.y - env_state.reactor_dest_y
    win_reactor = is_in_reactor & env_state.reactor_dest_active & (dx_dest ** 2 + dy_dest ** 2 < (SHIP_RADIUS + 5.0) ** 2)
    score_from_enemies = jnp.sum(just_killed_mask.astype(jnp.float32) * jnp.where(enemies.sprite_idx == int(SpriteIdx.ENEMY_ORANGE), 250.0, 350.0))
    score_from_reactor = jnp.where(win_reactor, 500.0, 0.0)
    score_from_ufo = jnp.where(ufo_just_died, 100.0, 0.0)
    reward = score_from_enemies + score_from_reactor + score_from_ufo + score_from_tanks
    start_crash = dead & ~was_crashing
    crash_timer_next = jnp.where(start_crash, 30, jnp.maximum(state_after_spawn.crash_timer - 1, 0))
    crash_animation_finished = (state_after_spawn.crash_timer == 1)
    reset_from_reactor_crash = (dead & is_in_reactor) & crash_animation_finished
    death_event = (crash_animation_finished & ~is_in_reactor) | reset_from_reactor_crash
    lives_next = state_after_spawn.lives - jnp.where(death_event, 1, 0)
    all_enemies_gone = jnp.all(enemies.hp <= 0) & ~state_after_ufo.ufo.alive & (state_after_ufo.ufo.death_timer == 0)
    has_meaningful_enemies = jnp.any(state_after_ufo.enemies.hp > 0)
    reset_level_win = all_enemies_gone & has_meaningful_enemies & ~is_in_reactor
    reset = reset_level_win | win_reactor | reset_from_reactor_crash
    game_over = (death_event & (lives_next <= 0)) | win_reactor
    def _respawn_level_state(s):
        ship_respawn = make_level_start_state(s.current_level)._replace(x=make_level_start_state(s.current_level).x + s.respawn_shift_x)
        return s._replace(state=ship_respawn, bullets=create_empty_bullets_64(), enemy_bullets=create_empty_bullets_16(), fire_cooldown=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32), cooldown=jnp.int32(0))
    final_env_state = jax.lax.cond(death_event & ~game_over, _respawn_level_state, lambda s: s, state_after_ufo)
    final_env_state = final_env_state._replace(
        state=ship_after_move, bullets=bullets, cooldown=cooldown, enemies=enemies, enemy_bullets=enemy_bullets,
        fire_cooldown=next_frame_cooldown, key=current_key, ufo=state_after_ufo.ufo, ufo_bullets=ufo_bullets,
        fuel_tanks=new_fuel_tanks, score=state_after_ufo.score + reward, crash_timer=crash_timer_next,
        lives=lives_next, done=game_over, reactor_destroyed=state_after_ufo.reactor_destroyed | win_reactor,
        planets_cleared_mask=jnp.where(reset_level_win, state_after_ufo.planets_cleared_mask.at[state_after_ufo.current_level].set(True), state_after_ufo.planets_cleared_mask),
        mode_timer=state_after_ufo.mode_timer + 1
    )
    obs = jnp.array([final_env_state.state.x, final_env_state.state.y, final_env_state.state.vx, final_env_state.state.vy, final_env_state.state.angle], dtype=jnp.float32)
    info = {
        "crash": start_crash, "hit_by_bullet": hit_by_enemy_bullet | hit_by_ufo_bullet, "reactor_crash_exit": reset_from_reactor_crash,
        "all_rewards": {
            "reward_enemies": score_from_enemies, "reward_reactor": score_from_reactor, "reward_ufo": score_from_ufo,
            "reward_tanks": score_from_tanks, "reward_saucer_kill": 0.0, "reward_penalty": 0.0,
        }
    }
    return obs, final_env_state, jnp.float32(reward), game_over, info, reset, jnp.int32(-1)

@jax.jit
def step_arena(env_state: EnvState, action: int) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any], jnp.ndarray, jnp.ndarray]:
    ship, saucer, is_crashing = env_state.state, env_state.saucer, env_state.crash_timer > 0
    actual_action = jnp.where(is_crashing, NOOP, action)
    ship_after_move = ship_step(ship, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT)
    can_fire = jnp.isin(action, jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])) & (env_state.cooldown == 0)
    bullets = jax.lax.cond(can_fire, lambda b: fire_bullet(b, ship_after_move.x, ship_after_move.y, ship_after_move.angle, PLAYER_BULLET_SPEED), lambda b: b, env_state.bullets)
    bullets = update_bullets(bullets)
    cooldown = jnp.where(can_fire, PLAYER_FIRE_COOLDOWN_FRAMES, jnp.maximum(env_state.cooldown - 1, 0))
    saucer_after_move = jax.lax.cond(saucer.alive, lambda s: _update_saucer_seek(s, ship_after_move.x, ship_after_move.y, SAUCER_SPEED_ARENA), lambda s: s, saucer)
    can_shoot_saucer = saucer_after_move.alive & (jnp.sum(env_state.enemy_bullets.alive) < 1) & ((env_state.mode_timer % SAUCER_FIRE_INTERVAL_FRAMES) == 0)
    enemy_bullets = jax.lax.cond(can_shoot_saucer, lambda eb: _fire_single_from_to(eb, saucer_after_move.x, saucer_after_move.y, ship_after_move.x, ship_after_move.y, SAUCER_BULLET_SPEED), lambda eb: eb, env_state.enemy_bullets)
    enemy_bullets = update_bullets(enemy_bullets)
    bullets, hit_saucer_by_bullet = _bullets_hit_saucer(bullets, saucer_after_move)
    hit_saucer_by_contact = _circle_hit(ship_after_move.x, ship_after_move.y, SHIP_RADIUS, saucer_after_move.x, saucer_after_move.y, SAUCER_RADIUS) & saucer_after_move.alive
    enemy_bullets, hit_ship_by_bullet = consume_ship_hits(ship_after_move, enemy_bullets, SHIP_RADIUS)
    saucer_is_hit = hit_saucer_by_bullet | hit_saucer_by_contact
    hp_after_hit = saucer_after_move.hp - jnp.where(saucer_is_hit, 1, 0)
    just_died = saucer_after_move.alive & (hp_after_hit <= 0)
    saucer_final = saucer_after_move._replace(
        hp=hp_after_hit, alive=hp_after_hit > 0,
        death_timer=jnp.where(just_died, SAUCER_EXPLOSION_FRAMES, jnp.maximum(saucer_after_move.death_timer - 1, 0))
    )
    ship_is_hit = hit_ship_by_bullet | hit_saucer_by_contact
    start_crash = ship_is_hit & ~is_crashing
    crash_timer_next = jnp.where(start_crash, 30, jnp.maximum(env_state.crash_timer - 1, 0))
    reset_signal = env_state.crash_timer == 1
    back_to_map_signal = ~ship_is_hit & ~saucer_final.alive & (saucer_final.death_timer == 0)
    obs = jnp.array([ship_after_move.x, ship_after_move.y, ship_after_move.vx, ship_after_move.vy, ship_after_move.angle], dtype=jnp.float32)
    reward = jnp.where(just_died, 300.0, 0.0)
    info = {
        "crash": start_crash, "hit_by_bullet": hit_ship_by_bullet, "reactor_crash_exit": jnp.array(False, dtype=jnp.bool_),
        "all_rewards": {
            "reward_enemies": 0.0, "reward_reactor": 0.0, "reward_ufo": 0.0, "reward_tanks": 0.0,
            "reward_saucer_kill": reward, "reward_penalty": 0.0,
        }
    }
    final_env_state = env_state._replace(
        state=ship_after_move, bullets=bullets, cooldown=cooldown, saucer=saucer_final,
        enemy_bullets=enemy_bullets, crash_timer=crash_timer_next, mode_timer=env_state.mode_timer + 1,
        score=env_state.score + reward,
    )
    def _go_to_map_win(env):
        return env._replace(mode=jnp.int32(0), saucer=make_default_saucer())
    final_env_state = jax.lax.cond(back_to_map_signal, _go_to_map_win, lambda e: e, final_env_state)
    return obs, final_env_state, jnp.float32(reward), jnp.array(False, dtype=jnp.bool_), info, reset_signal | back_to_map_signal, jnp.int32(-1)

@jax.jit
def step_core(env_state: EnvState, action: int):
    def _game_is_over(state, _):
        info = {
            "crash": jnp.array(False, dtype=jnp.bool_),
            "hit_by_bullet": jnp.array(False, dtype=jnp.bool_),
            "reactor_crash_exit": jnp.array(False, dtype=jnp.bool_),
            "all_rewards": {
                "reward_enemies": jnp.float32(0.0), "reward_reactor": jnp.float32(0.0),
                "reward_ufo": jnp.float32(0.0), "reward_tanks": jnp.float32(0.0),
                "reward_saucer_kill": jnp.float32(0.0), "reward_penalty": jnp.float32(0.0),
            }
        }
        obs = jnp.array([state.state.x, state.state.y, state.state.vx, state.state.vy, state.state.angle], dtype=jnp.float32)
        return obs, state, jnp.float32(0.0), jnp.array(True, dtype=jnp.bool_), info, jnp.array(False, dtype=jnp.bool_), jnp.int32(-1)

    def _game_is_running(state, act):
        return jax.lax.switch(
            jnp.clip(state.mode, 0, 2),
            [step_map, _step_level_core, step_arena],
            state, act
        )
    return jax.lax.cond(env_state.done, _game_is_over, _game_is_running, env_state, action)

@partial(jax.jit, static_argnums=(2,))
def step_full(env_state: EnvState, action: int, env_instance: 'JaxGravitar'):
    def _handle_reset(operands):
        obs, current_state, reward, done, info, reset, level = operands
        default_rewards = {
            "reward_enemies": jnp.float32(0.0), "reward_reactor": jnp.float32(0.0),
            "reward_ufo": jnp.float32(0.0), "reward_tanks": jnp.float32(0.0),
            "reward_saucer_kill": jnp.float32(0.0), "reward_penalty": jnp.float32(0.0),
        }
        def _enter_level(_):
            def _reset_level_py(key, level_val, state_val):
                return env_instance.reset_level(key, level_val, state_val)
            new_main_key, subkey_for_reset = jax.random.split(current_state.key)
            obs_reset, next_state = jax.pure_callback(
                _reset_level_py, env_instance.reset_level_out_struct,
                subkey_for_reset, level, current_state
            )
            next_state = next_state._replace(key=new_main_key)
            enter_info = {
                "level_cleared": jnp.array(False, dtype=jnp.bool_),
                "crash": info.get("crash", jnp.array(False, dtype=jnp.bool_)),
                "hit_by_bullet": info.get("hit_by_bullet", jnp.array(False, dtype=jnp.bool_)),
                "reactor_crash_exit": info.get("reactor_crash_exit", jnp.array(False, dtype=jnp.bool_)),
                "all_rewards": info.get("all_rewards", default_rewards),
            }
            return obs_reset, next_state, jnp.float32(reward), jnp.array(False, dtype=jnp.bool_), enter_info, jnp.array(True, dtype=jnp.bool_), level
        def _return_to_map(_):
            is_a_death_event = (level == -2) | info.get("crash", False) | info.get("hit_by_bullet", False) | info.get("reactor_crash_exit", False)
            def _on_win(_):
                new_main_key, subkey_for_reset = jax.random.split(current_state.key)
                obs_reset, map_state = env_instance.reset_map(
                    subkey_for_reset, lives=current_state.lives, score=current_state.score,
                    reactor_destroyed=current_state.reactor_destroyed, planets_cleared_mask=current_state.planets_cleared_mask
                )
                map_state = map_state._replace(key=new_main_key)
                win_info = {
                    "level_cleared": jnp.array(True, dtype=jnp.bool_),
                    "crash": info.get("crash", jnp.array(False, dtype=jnp.bool_)),
                    "hit_by_bullet": info.get("hit_by_bullet", jnp.array(False, dtype=jnp.bool_)),
                    "reactor_crash_exit": info.get("reactor_crash_exit", jnp.array(False, dtype=jnp.bool_)),
                    "all_rewards": info.get("all_rewards", default_rewards),
                }
                return obs_reset, map_state, jnp.float32(reward), jnp.array(False, dtype=jnp.bool_), win_info, jnp.array(True, dtype=jnp.bool_), level
            def _on_death(_):
                lives_after_death = current_state.lives - 1
                is_game_over = (lives_after_death <= 0)
                death_info = {
                    "level_cleared": jnp.array(False, dtype=jnp.bool_),
                    "crash": info.get("crash", jnp.array(False, dtype=jnp.bool_)),
                    "hit_by_bullet": info.get("hit_by_bullet", jnp.array(False, dtype=jnp.bool_)),
                    "reactor_crash_exit": info.get("reactor_crash_exit", jnp.array(False, dtype=jnp.bool_)),
                    "all_rewards": info.get("all_rewards", default_rewards),
                }
                new_main_key, subkey_for_reset = jax.random.split(current_state.key)
                obs_reset, map_state = env_instance.reset_map(
                    subkey_for_reset, lives=lives_after_death, score=current_state.score,
                    reactor_destroyed=current_state.reactor_destroyed, planets_cleared_mask=current_state.planets_cleared_mask
                )
                final_map_state = map_state._replace(key=new_main_key, done=is_game_over)
                return obs_reset, final_map_state, jnp.float32(reward), is_game_over, death_info, jnp.array(True, dtype=jnp.bool_), level
            return jax.lax.cond(is_a_death_event, _on_death, _on_win, operand=None)
        return jax.lax.cond(level >= 0, _enter_level, _return_to_map, operand=None)
    def _no_reset(operands):
        obs, new_env_state, reward, done, info, reset, level = operands
        default_rewards = {
            "reward_enemies": jnp.float32(0.0), "reward_reactor": jnp.float32(0.0),
            "reward_ufo": jnp.float32(0.0), "reward_tanks": jnp.float32(0.0),
            "reward_saucer_kill": jnp.float32(0.0), "reward_penalty": jnp.float32(0.0),
        }
        no_reset_info = {
            "level_cleared": jnp.array(False, dtype=jnp.bool_),
            "crash": info.get("crash", jnp.array(False, dtype=jnp.bool_)),
            "hit_by_bullet": info.get("hit_by_bullet", jnp.array(False, dtype=jnp.bool_)),
            "reactor_crash_exit": info.get("reactor_crash_exit", jnp.array(False, dtype=jnp.bool_)),
            "all_rewards": info.get("all_rewards", default_rewards),
        }
        return obs, new_env_state, jnp.float32(reward), done, no_reset_info, reset, level
    obs, new_env_state, reward, done, info, reset, level = step_core(env_state, action)
    operands = (obs, new_env_state, reward, done, info, reset, level)
    return jax.lax.cond(reset, _handle_reset, _no_reset, operands)

class JaxGravitar(JaxEnvironment):
    def __init__(self, obs_type="vector"):
        super().__init__()
        self.obs_type = obs_type
        self.obs_shape = (5,)
        self.num_actions = 18

        pygame.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        self.sprites = load_sprites_tuple()
        self.renderer = GravitarRenderer(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

        self.sprite_dims = {}
        sprites_to_measure = [
            SpriteIdx.ENEMY_ORANGE, SpriteIdx.ENEMY_GREEN,
            SpriteIdx.FUEL_TANK, SpriteIdx.ENEMY_UFO,
            SpriteIdx.ENEMY_ORANGE_FLIPPED,
        ]
        for sprite_idx in sprites_to_measure:
            sprite_surf = self.sprites[sprite_idx]
            if sprite_surf:
                self.sprite_dims[int(sprite_idx)] = (sprite_surf.get_width(), sprite_surf.get_height())

        MAP_SCALE, HITBOX_SCALE = 3, 0.90
        layout = [
            (SpriteIdx.PLANET1, 0.82, 0.18), (SpriteIdx.PLANET2, 0.22, 0.24),
            (SpriteIdx.REACTOR, 0.18, 0.43), (SpriteIdx.SPAWN_LOC, 0.50, 0.56),
            (SpriteIdx.OBSTACLE, 0.57, 0.38), (SpriteIdx.PLANET3, 0.76, 0.76),
            (SpriteIdx.PLANET4, 0.14, 0.88),
        ]
        px, py, pr, pi = [], [], [], []
        for idx, xp, yp in layout:
            cx, cy = xp * WINDOW_WIDTH, yp * WINDOW_HEIGHT
            spr = self.sprites[idx]
            r = (8.0 / WORLD_SCALE if idx == SpriteIdx.OBSTACLE else 0.3 * max(spr.get_width(), spr.get_height()) * MAP_SCALE * HITBOX_SCALE) if spr else 4
            px.append(cx); py.append(cy); pr.append(r); pi.append(int(idx))
        self.planets = (np.array(px, dtype=np.float32), np.array(py, dtype=np.float32), np.array(pr, dtype=np.float32), np.array(pi, dtype=np.int32))
        self.terrain_bank = self._build_terrain_bank()

        num_levels = max(LEVEL_LAYOUTS.keys()) + 1
        max_objects = max(len(v) for v in LEVEL_LAYOUTS.values()) if LEVEL_LAYOUTS else 0
        layout_types = np.full((num_levels, max_objects), -1, dtype=np.int32)
        layout_coords_x = np.zeros((num_levels, max_objects), dtype=np.float32)
        layout_coords_y = np.zeros((num_levels, max_objects), dtype=np.float32)
        for level_id, layout_data in LEVEL_LAYOUTS.items():
            for i, obj in enumerate(layout_data):
                layout_types[level_id, i] = obj['type']
                layout_coords_x[level_id, i] = obj['coords'][0]
                layout_coords_y[level_id, i] = obj['coords'][1]
        self.jax_layout = {"types": jnp.array(layout_types), "coords_x": jnp.array(layout_coords_x), "coords_y": jnp.array(layout_coords_y)}

        max_sprite_id = max(int(e) for e in SpriteIdx)
        dims_array = np.zeros((max_sprite_id + 1, 2), dtype=np.float32)
        for k, v in self.sprite_dims.items():
            dims_array[k] = v
        self.jax_sprite_dims = jnp.array(dims_array)

        level_ids_sorted = sorted(LEVEL_ID_TO_TERRAIN_SPRITE.keys())
        self.jax_level_to_terrain = jnp.array([LEVEL_ID_TO_TERRAIN_SPRITE[k] for k in level_ids_sorted])
        self.jax_level_to_bank = jnp.array([LEVEL_ID_TO_BANK_IDX[k] for k in level_ids_sorted])
        self.jax_level_offsets = jnp.array([LEVEL_OFFSETS[k] for k in level_ids_sorted], dtype=jnp.float32)

        level_transforms = np.zeros((num_levels, 3), dtype=np.float32)
        for level_id in level_ids_sorted:
            terrain_sprite_enum = LEVEL_ID_TO_TERRAIN_SPRITE[level_id]
            terr_surf = self.sprites[terrain_sprite_enum]
            tw, th = terr_surf.get_width(), terr_surf.get_height()
            scale = min(WINDOW_WIDTH / tw, WINDOW_HEIGHT / th) * TERRANT_SCALE_OVERRIDES.get(terrain_sprite_enum, 1.0)
            sw, sh = int(tw * scale), int(th * scale)
            level_offset = LEVEL_OFFSETS.get(level_id, (0,0))
            ox, oy = (WINDOW_WIDTH - sw) // 2 + level_offset[0], (WINDOW_HEIGHT - sh) // 2 + level_offset[1]
            level_transforms[level_id] = [scale, ox, oy]
        self.jax_level_transforms = jnp.array(level_transforms)

        dummy_key = jax.random.PRNGKey(0)
        _, dummy_state = self.reset(dummy_key)
        tmp_obs, tmp_state = self.reset_level(dummy_key, jnp.int32(0), dummy_state)
        self.reset_level_out_struct = (
            jax.ShapeDtypeStruct(tmp_obs.shape, tmp_obs.dtype),
            jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tmp_state)
        )

    def reset(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, EnvState]:
        vec_obs, state = self.reset_map(key)
        return self.get_object_centric_obs(state) if self.obs_type == "object_centric" else vec_obs, state

    def step(self, env_state: EnvState, action: int):
        vec_obs, ns, reward, done, info, _, _ = step_full(env_state, action, self)
        obs = self.get_object_centric_obs(ns) if self.obs_type == "object_centric" else vec_obs
        return obs, ns, reward.astype(jnp.float32), done, info

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def observation_space(self) -> spaces.Space:
        if self.obs_type == "object_centric":
            return self.object_centric_observation_space()
        low = jnp.array([0.0, 0.0, -10.0, -10.0, -jnp.pi], dtype=jnp.float32)
        high = jnp.array([float(WINDOW_WIDTH), float(WINDOW_HEIGHT), 10.0, 10.0, jnp.pi], dtype=jnp.float32)
        return spaces.Box(low=low, high=high, shape=self.obs_shape, dtype=jnp.float32)

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=jnp.uint8)

    def obs_to_flat_array(self, obs: Any) -> jnp.ndarray:
        leaves, _ = jax.tree_util.tree_flatten(obs)
        return jnp.concatenate([leaf.flatten() for leaf in leaves])

    def object_centric_observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "ship": spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(5,), dtype=jnp.float32),
            "player_bullets": spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(MAX_BULLETS, 5), dtype=jnp.float32),
            "enemies": spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(MAX_ENEMIES, 5), dtype=jnp.float32),
            "enemy_bullets": spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(16, 5), dtype=jnp.float32),
            "ufo": spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(5,), dtype=jnp.float32),
            "saucer": spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(5,), dtype=jnp.float32),
        })

    def get_object_centric_obs(self, state: EnvState) -> Dict[str, jnp.ndarray]:
        return {
            "ship": jnp.array([state.state.x, state.state.y, state.state.vx, state.state.vy, state.state.angle]),
            "player_bullets": jnp.stack([state.bullets.x, state.bullets.y, state.bullets.vx, state.bullets.vy, state.bullets.alive.astype(jnp.float32)], axis=-1),
            "enemies": jnp.stack([state.enemies.x, state.enemies.y, state.enemies.w, state.enemies.h, (state.enemies.hp > 0).astype(jnp.float32)], axis=-1),
            "enemy_bullets": jnp.stack([state.enemy_bullets.x, state.enemy_bullets.y, state.enemy_bullets.vx, state.enemy_bullets.vy, state.enemy_bullets.alive.astype(jnp.float32)], axis=-1),
            "ufo": jnp.array([state.ufo.x, state.ufo.y, state.ufo.vx, state.ufo.vy, state.ufo.alive.astype(jnp.float32)]),
            "saucer": jnp.array([state.saucer.x, state.saucer.y, state.saucer.vx, state.saucer.vy, state.saucer.alive.astype(jnp.float32)]),
        }

    def get_ram(self, state: EnvState) -> jnp.ndarray:
        return jnp.zeros(128, dtype=jnp.uint8)

    def get_ale_lives(self, state: EnvState) -> jnp.ndarray:
        return state.lives

    def close(self):
        pass

    def seed(self, seed: Optional[int] = None) -> jnp.ndarray:
        return jax.random.PRNGKey(seed if seed is not None else 0)

    def render(self, env_state: EnvState) -> jnp.ndarray:
        return self.renderer.render(env_state)

    def reset_map(self, key: jnp.ndarray, lives: Optional[int] = None, score: Optional[float] = None, reactor_destroyed: Optional[jnp.ndarray] = None, planets_cleared_mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, EnvState]:
        ship_state = ShipState(
            x=jnp.array(WINDOW_WIDTH / 2, dtype=jnp.float32),
            y=jnp.array((WINDOW_HEIGHT + HUD_HEIGHT) / 2, dtype=jnp.float32),
            vx=jnp.array(jnp.cos(-jnp.pi / 4) * 0.3, dtype=jnp.float32),
            vy=jnp.array(jnp.sin(-jnp.pi / 4) * 0.3, dtype=jnp.float32),
            angle=jnp.array(-jnp.pi / 2, dtype=jnp.float32)
        )
        px_np, py_np, pr_np, pi_np = self.planets
        ids_np = [SPRITE_TO_LEVEL_ID.get(sprite_idx, -1) for sprite_idx in pi_np]
        env_state = EnvState(
            mode=jnp.int32(0), state=ship_state, bullets=create_empty_bullets_64(),
            cooldown=jnp.int32(0), enemies=create_empty_enemies(),
            fuel_tanks=FuelTanks(x=jnp.full((MAX_ENEMIES,), -1.0), y=jnp.full((MAX_ENEMIES,), -1.0), w=jnp.zeros((MAX_ENEMIES,)), h=jnp.zeros((MAX_ENEMIES,)), sprite_idx=jnp.full((MAX_ENEMIES,), -1), active=jnp.zeros((MAX_ENEMIES,), dtype=jnp.bool_)),
            enemy_bullets=create_empty_bullets_16(), fire_cooldown=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),
            key=key, key_alt=key, score=jnp.float32(score if score is not None else 0.0),
            done=jnp.array(False, dtype=jnp.bool_), lives=jnp.int32(lives if lives is not None else MAX_LIVES),
            crash_timer=jnp.int32(0), planets_px=jnp.array(px_np), planets_py=jnp.array(py_np),
            planets_pr=jnp.array(pr_np), planets_pi=jnp.array(pi_np), planets_id=jnp.array(ids_np),
            current_level=jnp.int32(-1), terrain_sprite_idx=jnp.int32(-1),
            terrain_mask=jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=jnp.uint8),
            terrain_scale=jnp.float32(1.0), terrain_offset=jnp.zeros(2, dtype=jnp.float32),
            terrain_bank=self.terrain_bank, terrain_bank_idx=jnp.int32(0), respawn_shift_x=jnp.float32(0.0),
            reactor_dest_active=jnp.array(False, dtype=jnp.bool_), reactor_dest_x=jnp.float32(0.0), reactor_dest_y=jnp.float32(0.0),
            reactor_dest_radius=jnp.float32(0.4), mode_timer=jnp.int32(0), saucer=make_default_saucer(),
            saucer_spawn_timer=jnp.int32(SAUCER_SPAWN_DELAY_FRAMES), map_return_x=jnp.float32(0.0), map_return_y=jnp.float32(0.0),
            ufo=make_empty_ufo(), ufo_used=jnp.array(False, dtype=jnp.bool_), ufo_home_x=jnp.float32(0.0), ufo_home_y=jnp.float32(0.0),
            ufo_bullets=create_empty_bullets_16(), level_offset=jnp.zeros(2, dtype=jnp.float32),
            reactor_destroyed=reactor_destroyed if reactor_destroyed is not None else jnp.array(False, dtype=jnp.bool_),
            planets_cleared_mask=planets_cleared_mask if planets_cleared_mask is not None else jnp.zeros_like(self.planets[0], dtype=jnp.bool_),
        )
        obs = jnp.array([ship_state.x, ship_state.y, ship_state.vx, ship_state.vy, ship_state.angle], dtype=jnp.float32)
        return obs, env_state

    def reset_level(self, key: jnp.ndarray, level_id: jnp.ndarray, prev_env_state: EnvState):
        level_id = jnp.asarray(level_id, dtype=jnp.int32)
        transform = self.jax_level_transforms[level_id]
        scale, ox, oy = transform[0], transform[1], transform[2]
        def loop_body(i, carry):
            enemies, tanks, e_idx, t_idx = carry
            obj_type = self.jax_layout["types"][level_id, i]
            def place_obj(val):
                enemies_in, tanks_in, e_idx_in, t_idx_in = val
                orig_idx = jnp.where(obj_type == int(SpriteIdx.ENEMY_ORANGE_FLIPPED), int(SpriteIdx.ENEMY_ORANGE), obj_type)
                w, h = self.jax_sprite_dims[orig_idx]
                x, y = ox + self.jax_layout["coords_x"][level_id, i] * scale, oy + self.jax_layout["coords_y"][level_id, i] * scale
                is_tank = (obj_type == int(SpriteIdx.FUEL_TANK)).astype(jnp.int32)
                new_enemies = enemies_in._replace(
                    x=enemies_in.x.at[e_idx_in].set(jnp.where(is_tank, -1.0, x)),
                    y=enemies_in.y.at[e_idx_in].set(jnp.where(is_tank, -1.0, y)),
                    w=enemies_in.w.at[e_idx_in].set(jnp.where(is_tank, 0.0, w)),
                    h=enemies_in.h.at[e_idx_in].set(jnp.where(is_tank, 0.0, h)),
                    sprite_idx=enemies_in.sprite_idx.at[e_idx_in].set(jnp.where(is_tank, -1, obj_type)),
                    hp=enemies_in.hp.at[e_idx_in].set(jnp.where(is_tank, 0, 1)),
                )
                new_tanks = tanks_in._replace(
                    x=tanks_in.x.at[t_idx_in].set(jnp.where(is_tank, x, -1.0)),
                    y=tanks_in.y.at[t_idx_in].set(jnp.where(is_tank, y, -1.0)),
                    w=tanks_in.w.at[t_idx_in].set(jnp.where(is_tank, w, 0.0)),
                    h=tanks_in.h.at[t_idx_in].set(jnp.where(is_tank, h, 0.0)),
                    sprite_idx=tanks_in.sprite_idx.at[t_idx_in].set(jnp.where(is_tank, obj_type, -1)),
                    active=tanks_in.active.at[t_idx_in].set(jnp.where(is_tank, True, False)),
                )
                return new_enemies, new_tanks, e_idx_in + (1 - is_tank), t_idx_in + is_tank
            return jax.lax.cond(obj_type != -1, place_obj, lambda x: x, (enemies, tanks, e_idx, t_idx))
        init_enemies = create_empty_enemies()
        init_tanks = FuelTanks(x=jnp.full((MAX_ENEMIES,), -1.0), y=jnp.full((MAX_ENEMIES,), -1.0), w=jnp.zeros((MAX_ENEMIES,)), h=jnp.zeros((MAX_ENEMIES,)), sprite_idx=jnp.full((MAX_ENEMIES,), -1), active=jnp.zeros((MAX_ENEMIES,), dtype=jnp.bool_))
        enemies, fuel_tanks, _, _ = jax.lax.fori_loop(0, self.jax_layout["types"].shape[1], loop_body, (init_enemies, init_tanks, 0, 0))
        ship_state = make_level_start_state(level_id)
        env_state = prev_env_state._replace(
            mode=jnp.int32(1), state=ship_state, bullets=create_empty_bullets_64(), cooldown=jnp.int32(0),
            enemies=enemies, fuel_tanks=fuel_tanks, enemy_bullets=create_empty_bullets_16(),
            fire_cooldown=jnp.full((MAX_ENEMIES,), 60, dtype=jnp.int32),
            key=key, crash_timer=jnp.int32(0), current_level=level_id,
            terrain_sprite_idx=self.jax_level_to_terrain[level_id],
            terrain_mask=jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=jnp.uint8),
            terrain_scale=scale, terrain_offset=jnp.array([ox, oy]),
            terrain_bank_idx=self.jax_level_to_bank[level_id],
            reactor_dest_active=(level_id == 4), reactor_dest_x=jnp.float32(95), reactor_dest_y=jnp.float32(114),
            mode_timer=jnp.int32(0), ufo=make_empty_ufo(), ufo_used=jnp.array(False, dtype=jnp.bool_),
            level_offset=self.jax_level_offsets[level_id],
        )
        obs = jnp.array([ship_state.x, ship_state.y, ship_state.vx, ship_state.vy, ship_state.angle], dtype=jnp.float32)
        return obs, env_state

    def _build_terrain_bank(self) -> jnp.ndarray:
        W, H = WINDOW_WIDTH, WINDOW_HEIGHT
        bank = [np.zeros((H, W, 3), dtype=np.uint8)]
        BANK_IDX_TO_LEVEL_ID = {v: k for k, v in LEVEL_ID_TO_BANK_IDX.items()}
        def sprite_to_mask(idx: int, bank_idx: int) -> np.ndarray:
            surf = self.sprites[SpriteIdx(idx)]
            tw, th = surf.get_width(), surf.get_height()
            scale = min(W / tw, H / th) * TERRANT_SCALE_OVERRIDES.get(SpriteIdx(idx), 1.0)
            sw, sh = int(tw * scale), int(th * scale)
            level_id = BANK_IDX_TO_LEVEL_ID.get(bank_idx)
            level_offset_x, level_offset_y = LEVEL_OFFSETS.get(level_id, (0, 0)) if level_id is not None else (0,0)
            ox, oy = (W - sw) // 2 + level_offset_x, (H - sh) // 2 + level_offset_y
            scaled_surf = pygame.transform.scale(surf, (sw, sh))
            rgb_array_hwc = pygame.surfarray.pixels3d(scaled_surf).transpose((1, 0, 2))
            color_map = np.zeros((H, W, 3), dtype=np.uint8)
            src_w, src_h = rgb_array_hwc.shape[1], rgb_array_hwc.shape[0]
            dst_x, dst_y = max(ox, 0), max(oy, 0)
            src_x, src_y = abs(min(ox, 0)), abs(min(oy, 0))
            copy_w, copy_h = min(W - dst_x, src_w - src_x), min(H - dst_y, src_h - src_y)
            if copy_w > 0 and copy_h > 0:
                color_map[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = rgb_array_hwc[src_y:src_y + copy_h, src_x:src_x + copy_w]
            return color_map
        terrains_to_build = [
            (SpriteIdx.TERRANT1, 1), (SpriteIdx.TERRANT2, 2), (SpriteIdx.TERRANT3, 3),
            (SpriteIdx.TERRANT4, 4), (SpriteIdx.REACTOR_TERR, 5),
        ]
        for sprite_idx, bank_idx in terrains_to_build:
            bank.append(sprite_to_mask(int(sprite_idx), bank_idx))
        return jnp.array(np.stack(bank, axis=0), dtype=jnp.uint8)

class GravitarRenderer(JAXGameRenderer):
    def __init__(self, width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT):
        super().__init__()
        self.width, self.height = width, height
        jax_sprites = _load_and_convert_sprites()
        blit_functions = {}
        for sprite_idx, sprite_array_rgba in jax_sprites.items():
            if sprite_array_rgba is None: continue
            h, w, _ = sprite_array_rgba.shape
            def make_blit_func(sprite_data, static_h, static_w):
                sprite_rgb = sprite_data[..., :3]
                sprite_alpha = (sprite_data[..., 3] / 255.0)[..., None]
                def _blit_sprite(frame, x, y):
                    start_x = jnp.round(x - static_w / 2.0).astype(jnp.int32)
                    start_y = jnp.round(y - static_h / 2.0).astype(jnp.int32)
                    target_patch = jax.lax.dynamic_slice(frame, (start_y, start_x, 0), (static_h, static_w, 3))
                    blended_patch = sprite_rgb * sprite_alpha + target_patch * (1.0 - sprite_alpha)
                    return jax.lax.dynamic_update_slice(frame, blended_patch.astype(jnp.uint8), (start_y, start_x, 0))
                return _blit_sprite
            blit_functions[sprite_idx] = make_blit_func(sprite_array_rgba, h, w)
        max_idx = max(jax_sprites.keys()) if jax_sprites else -1
        self.blit_branches_for_switch = tuple(blit_functions.get(i, lambda f, x, y: f) for i in range(max_idx + 1))
        idle_sprite, crash_sprite = jax_sprites.get(int(SpriteIdx.SHIP_IDLE)), jax_sprites.get(int(SpriteIdx.SHIP_CRASH))
        if all(s is not None for s in [idle_sprite, crash_sprite]):
            h, w = max(s.shape[0] for s in [idle_sprite, crash_sprite]), max(s.shape[1] for s in [idle_sprite, crash_sprite])
            def pad(sprite): return jnp.pad(sprite, (((h - sprite.shape[0]) // 2, h - sprite.shape[0] - (h - sprite.shape[0]) // 2), ((w - sprite.shape[1]) // 2, w - sprite.shape[1] - (w - sprite.shape[1]) // 2), (0, 0)))
            self.padded_ship_idle, self.padded_ship_crash = pad(idle_sprite), pad(crash_sprite)
        else: self.padded_ship_idle, self.padded_ship_crash = jnp.zeros((1, 1, 4), dtype=jnp.uint8), jnp.zeros((1, 1, 4), dtype=jnp.uint8)

    @partial(jax.jit, static_argnames=('self',))
    def render(self, state: EnvState) -> jnp.ndarray:
        frame = jnp.zeros((self.height, self.width, 3), dtype=jnp.uint8)
        def blit(sprite_id, f, x, y):
            safe_idx = jnp.clip(sprite_id, 0, len(self.blit_branches_for_switch) - 1)
            return jax.lax.switch(safe_idx, self.blit_branches_for_switch, f, x, y)
        def draw_map(f):
            def draw_one(i, fc):
                sid, x, y = state.planets_pi[i], state.planets_px[i], state.planets_py[i]
                is_cleared = (i < state.planets_cleared_mask.shape[0]) & state.planets_cleared_mask[i]
                is_destroyed = (sid == int(SpriteIdx.REACTOR)) & state.reactor_destroyed
                return jax.lax.cond(~(is_cleared | is_destroyed), lambda f_in: blit(sid, f_in, x, y), lambda f_in: f_in, fc)
            return jax.lax.fori_loop(0, state.planets_pi.shape[0], draw_one, f)
        def draw_terrain(f):
            bank_idx = jnp.clip(state.terrain_bank_idx, 0, state.terrain_bank.shape[0] - 1)
            terrain_map = state.terrain_bank[bank_idx]
            return jnp.where(jnp.sum(terrain_map, axis=-1, keepdims=True) > 0, terrain_map, f)
        frame = jax.lax.cond(state.mode == 0, draw_map, lambda f: jax.lax.cond(state.mode == 1, draw_terrain, lambda f_in: f_in, f), frame)
        def draw_actors(f):
            def draw_one_enemy(i, fc):
                is_alive, is_exploding = state.enemies.hp[i] > 0, state.enemies.death_timer[i] > 0
                sid, x, y = state.enemies.sprite_idx[i], state.enemies.x[i], state.enemies.y[i]
                f_alive = jax.lax.cond(is_alive & ~is_exploding, lambda f_in: blit(sid, f_in, x, y), lambda f_in: f_in, fc)
                return jax.lax.cond(is_exploding, lambda f_in: blit(int(SpriteIdx.ENEMY_CRASH), f_in, x, y), lambda f_in: f_in, f_alive)
            f = jax.lax.fori_loop(0, MAX_ENEMIES, draw_one_enemy, f)
            def draw_one_tank(i, fc):
                is_active, sid, x, y = state.fuel_tanks.active[i], state.fuel_tanks.sprite_idx[i], state.fuel_tanks.x[i], state.fuel_tanks.y[i]
                return jax.lax.cond(is_active, lambda f_in: blit(sid, f_in, x, y), lambda f_in: f_in, fc)
            f = jax.lax.fori_loop(0, MAX_ENEMIES, draw_one_tank, f)
            ufo = state.ufo
            f_ufo = jax.lax.cond(ufo.alive, lambda f_in: blit(int(SpriteIdx.ENEMY_UFO), f_in, ufo.x, ufo.y), lambda f_in: f_in, f)
            f_ufo = jax.lax.cond(ufo.death_timer > 0, lambda f_in: blit(int(SpriteIdx.ENEMY_CRASH), f_in, ufo.x, ufo.y), lambda f_in: f_in, f_ufo)
            return f_ufo
        frame = jax.lax.cond(state.mode == 1, draw_actors, lambda f: f, frame)
        saucer = state.saucer
        f_saucer = jax.lax.cond(saucer.alive, lambda f_in: blit(int(SpriteIdx.ENEMY_SAUCER), f_in, saucer.x, saucer.y), lambda f_in: f_in, frame)
        frame = jax.lax.cond((state.mode == 0) | (state.mode == 2), lambda: jax.lax.cond(saucer.death_timer > 0, lambda f_in: blit(int(SpriteIdx.SAUCER_CRASH), f_in, saucer.x, saucer.y), lambda f_in: f_in, f_saucer), lambda: frame)
        should_draw_dest = (state.mode == 1) & (state.terrain_sprite_idx == int(SpriteIdx.REACTOR_TERR)) & state.reactor_dest_active
        frame = jax.lax.cond(should_draw_dest, lambda f: blit(int(SpriteIdx.REACTOR_DEST), f, state.reactor_dest_x, state.reactor_dest_y), lambda f: f, frame)
        def draw_bullets(f, bullets, sid):
            def draw_one(i, fc):
                return jax.lax.cond(bullets.alive[i], lambda f_in: blit(sid, f_in, bullets.x[i], bullets.y[i]), lambda f_in: f_in, fc)
            return jax.lax.fori_loop(0, bullets.x.shape[0], draw_one, f)
        frame = draw_bullets(frame, state.bullets, int(SpriteIdx.SHIP_BULLET))
        frame = draw_bullets(frame, state.enemy_bullets, int(SpriteIdx.ENEMY_BULLET))
        frame = draw_bullets(frame, state.ufo_bullets, int(SpriteIdx.ENEMY_BULLET))
        ship_sprite_data = jax.lax.select(state.crash_timer > 0, self.padded_ship_crash, self.padded_ship_idle)
        angle_deg = jnp.degrees(state.state.angle) + 90.0
        rotated_ship_rgba = _jax_rotate(ship_sprite_data, angle_deg, reshape=False)
        ship_h, ship_w, _ = rotated_ship_rgba.shape
        ship_rgb, ship_alpha = rotated_ship_rgba[..., :3], (rotated_ship_rgba[..., 3] / 255.0)[..., None]
        start_x, start_y = jnp.round(state.state.x - ship_w / 2.0).astype(jnp.int32), jnp.round(state.state.y - ship_h / 2.0).astype(jnp.int32)
        target_patch = jax.lax.dynamic_slice(frame, (start_y, start_x, 0), (ship_h, ship_w, 3))
        blended_patch = ship_rgb * ship_alpha + target_patch * (1.0 - ship_alpha)
        frame = jax.lax.dynamic_update_slice(frame, blended_patch.astype(jnp.uint8), (start_y, start_x, 0))
        score_val = state.score.astype(jnp.int32)
        digits = jnp.array([(score_val // 10 ** (5 - i)) % 10 for i in range(6)], dtype=jnp.int32)
        def draw_one_digit(i, fc):
            sid = digits[i] + int(SpriteIdx.DIGIT_0)
            x_pos, y_pos = self.width - 15 - (6 - i) * 8, 10
            return blit(sid, fc, x_pos, y_pos)
        frame = jax.lax.fori_loop(0, 6, draw_one_digit, frame)
        def draw_one_life(i, fc):
            x_pos, y_pos = self.width - 15 - (MAX_LIVES - i) * 8, 22
            return jax.lax.cond(i < state.lives, lambda f_in: blit(int(SpriteIdx.HP_UI), f_in, x_pos, y_pos), lambda f_in: f_in, fc)
        frame = jax.lax.fori_loop(0, MAX_LIVES, draw_one_life, frame)
        return frame

__all__ = ["JaxGravitar"]