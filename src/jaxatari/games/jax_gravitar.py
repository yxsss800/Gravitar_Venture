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
HUD_HEIGHT = 72
MAX_LIVES = 6
HUD_PADDING = 5
HUD_SHIP_WIDTH = 10
HUD_SHIP_HEIGHT = 12
HUD_SHIP_SPACING = 12
# Pygame window dimensions
WINDOW_WIDTH = 160 
WINDOW_HEIGHT = 210 

SAUCER_SPAWN_DELAY_FRAMES = 60 *3
SAUCER_RESPAWN_DELAY_FRAMES = 180 * 3
SAUCER_SPEED_MAP = jnp.float32(0.2)
SAUCER_SPEED_ARENA = jnp.float32(0.2)
SAUCER_RADIUS = jnp.float32(0.3)
SHIP_RADIUS   = jnp.float32(0.2)
SAUCER_INIT_HP = jnp.int32(1)

SAUCER_SCALE              = 2.2
ENEMY_ORANGE_SCALE = 2.5
ENEMY_GREEN_SCALE = 2.5
FUEL_TANK_SCALE = 2.5
UFO_SCALE = 2.5    

SAUCER_EXPLOSION_FRAMES = jnp.int32(60) 
SAUCER_FIRE_INTERVAL_FRAMES = jnp.int32(24)  
SAUCER_BULLET_SPEED         = jnp.float32(0.4)
ENEMY_EXPLOSION_FRAMES = jnp.int32(60) 
UFO_HIT_RADIUS = jnp.float32(0.3) 

def _jax_rotate(image, angle_deg, reshape=False, order=1, mode='constant', cval=0):
    angle_rad = jnp.deg2rad(angle_deg)
    height, width = image.shape[:2]
    center_y, center_x = height / 2, width / 2
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
    # Ship & bullets
    SHIP_IDLE = 0            # spaceship.npy
    SHIP_THRUST = 1          # ship_thrust.npy
    SHIP_BULLET = 2          # ship_bullet.npy

    # Enemy bullets
    ENEMY_BULLET = 3         # enemy_bullet.npy
    ENEMY_GREEN_BULLET = 4   # enemy_green_bullet.npy

    # Enemies
    ENEMY_ORANGE = 5         # enemy_orange.npy
    ENEMY_GREEN  = 6         # enemy_green.npy
    ENEMY_SAUCER = 7         # saucer.npy
    ENEMY_UFO    = 8         # UFO.npy  

    # Explosions / crashes
    ENEMY_CRASH  = 9         # enemy_crash.npy
    SAUCER_CRASH = 10        # saucer_crash.npy
    SHIP_CRASH   = 11        # ship_crash.npy

    # World objects
    FUEL_TANK    = 12        # fuel_tank.npy
    OBSTACLE     = 13        # obstacle.npy
    SPAWN_LOC    = 14        # spawn_location.npy

    # Reactor & terrain
    REACTOR      = 15        # reactor.npy
    REACTOR_TERR = 16        # reactor_terrant.npy
    TERRANT1     = 17        # terrant1.npy
    TERRANT2     = 18        # terrant2.npy
    TERRANT3     = 19        # terrant_3.npy
    TERRANT4     = 20        # terrant_4.npy

    # Planets & UI
    PLANET1      = 21        # planet1.npy
    PLANET2      = 22        # planet2.npy
    PLANET3      = 23        # planet3.npy
    PLANET4      = 24        # planet4.npy
    REACTOR_DEST = 25        # reactor_destination.npy
    SCORE_UI     = 26        # score.npy
    HP_UI        = 27        # HP.npy
    SHIP_THRUST_BACK = 28
    #Score digits
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

TERRANT_SCALE_OVERRIDES = {
    SpriteIdx.TERRANT2: 0.80,  
}


LEVEL_LAYOUTS = {
    # Level 0 (Planet 1)
    0: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'pos_ratio': (0.23, 0.52), 'flip_y': False},
        {'type': SpriteIdx.ENEMY_ORANGE, 'pos_ratio': (0.51, 0.46), 'flip_y': False},
        {'type': SpriteIdx.ENEMY_ORANGE, 'pos_ratio': (0.94, 0.30), 'flip_y': False},
        {'type': SpriteIdx.ENEMY_GREEN,  'pos_ratio': (0.13, 0.66), 'flip_y': False},
        {'type': SpriteIdx.FUEL_TANK,    'pos_ratio': (0.65, 0.60), 'flip_y': False},
    ],
    # Level 1 (Planet 2)
    1: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'pos_ratio': (0.87, 0.39), 'flip_y': False},
        {'type': SpriteIdx.ENEMY_ORANGE, 'pos_ratio': (0.52, 0.76), 'flip_y': True},
        {'type': SpriteIdx.ENEMY_ORANGE, 'pos_ratio': (0.16, 0.50), 'flip_y': True},
        {'type': SpriteIdx.ENEMY_GREEN,  'pos_ratio': (0.20, 0.65), 'flip_y': False},
        {'type': SpriteIdx.FUEL_TANK,    'pos_ratio': (0.34, 0.26), 'flip_y': False},
    ],
    # Level 2 (Planet 3)
    2: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'pos_ratio': (0.15, 0.47), 'flip_y': False},
        {'type': SpriteIdx.ENEMY_GREEN,  'pos_ratio': (0.27, 0.68), 'flip_y': False},
        {'type': SpriteIdx.ENEMY_ORANGE, 'pos_ratio': (0.37, 0.28), 'flip_y': False},
        {'type': SpriteIdx.ENEMY_GREEN,  'pos_ratio': (0.67, 0.40), 'flip_y': False},
        {'type': SpriteIdx.FUEL_TANK,    'pos_ratio': (0.84, 0.61), 'flip_y': False},
    ],
    # Level 3 (Planet 4)
    3: [
        {'type': SpriteIdx.ENEMY_ORANGE, 'pos_ratio': (0.55, 0.33), 'flip_y': False},
        {'type': SpriteIdx.ENEMY_ORANGE, 'pos_ratio': (0.72, 0.25), 'flip_y': True},
        {'type': SpriteIdx.ENEMY_ORANGE, 'pos_ratio': (0.76, 0.74), 'flip_y': False},
        {'type': SpriteIdx.ENEMY_GREEN,  'pos_ratio': (0.47, 0.48), 'flip_y': False},
        {'type': SpriteIdx.FUEL_TANK,    'pos_ratio': (0.12, 0.65), 'flip_y': False},
    ],
    # Level 4 (Reactor)
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

# 2. Maps the Level ID to the Terrain Bank Index (0=empty, 1=T1, 2=T2, etc.)
LEVEL_ID_TO_BANK_IDX = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
}
# ========== Bullet State ==========
# Defines the state of bullets
class Bullets(NamedTuple):
    x: jnp.ndarray  # shape(MAX_BULLETS, )
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    alive: jnp.ndarray  # boolean array


# ========== Enemies States ==========
# Initializes the state of enemies
class Enemies(NamedTuple):
    x: jnp.ndarray  # shape (MAX_ENEMIES,)
    y: jnp.ndarray
    w: jnp.ndarray
    h: jnp.ndarray
    vx: jnp.ndarray
    sprite_idx: jnp.ndarray
    flip_y: jnp.ndarray
    death_timer: jnp.ndarray
    hp: jnp.ndarray


# ========== Ship State ==========
class ShipState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    angle: jnp.ndarray

# ========== Saucer State ==========
class SaucerState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    hp: jnp.ndarray
    alive: jnp.ndarray
    death_timer: jnp.ndarray

# ========== UFO State ==========
class UFOState(NamedTuple):
    x: jnp.ndarray      # f32
    y: jnp.ndarray      # f32
    vx: jnp.ndarray     # f32
    vy: jnp.ndarray     # f32
    hp: jnp.ndarray     # i32
    alive: jnp.ndarray  # bool
    death_timer: jnp.ndarray  

# ========== Env State ==========
class EnvState(NamedTuple):
    mode: jnp.ndarray
    state: ShipState
    bullets: Bullets
    cooldown: jnp.ndarray
    enemies: Enemies
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
    planets_id: jnp.ndarray              # The ID of the entered level (int32)

    current_level: jnp.ndarray           # int32, current level ID (typically -1 in map mode)
    terrain_sprite_idx: jnp.ndarray      # int32, terrain sprite for the current level (TERRANT* / REACTOR_TERR)
    terrain_mask: jnp.ndarray            # (Hmask, Wmask) bool/uint8
    terrain_scale: jnp.ndarray           # float32, rendering scale factor
    terrain_offset: jnp.ndarray          # (2,) float32, screen-top-left offset [ox, oy]

    terrain_bank: jnp.ndarray            # uint8，shape (B, H, W)
    terrain_bank_idx: jnp.ndarray        # int32, index of the currently used bank (0 = no terrain)
    respawn_shift_x: jnp.ndarray         # float32
    reactor_dest_active: jnp.ndarray     # bool
    reactor_dest_x: jnp.ndarray          # float32, world coordinates
    reactor_dest_y: jnp.ndarray          # float32
    reactor_dest_radius: jnp.ndarray     # float32, world coordinate reach radius


    # --- saucer / arena ---
    mode_timer: jnp.ndarray              # int32, cumulative frames in the current mode
    saucer: SaucerState
    map_return_x: jnp.ndarray            # float32
    map_return_y: jnp.ndarray            # float32
    saucer_spawn_timer: jnp.ndarray      # Tracks if a saucer has spawned in the current level

    ufo: UFOState
    ufo_used: jnp.ndarray                # bool, marks if a UFO has been spawned in this level
    ufo_home_x: jnp.ndarray              # f32
    ufo_home_y: jnp.ndarray              # f32
    ufo_bullets: Bullets  
    level_offset: jnp.ndarray 
    reactor_destroyed: jnp.ndarray
    planets_cleared_mask: jnp.ndarray


# ========== Init Function ========== 

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
        vx=jnp.float32(0.0),   vy=jnp.float32(0.0),
        hp=jnp.int32(0),
        alive=jnp.array(False),
        death_timer=jnp.int32(0),
    )

# Maps planet sprite indices to terrain bank indices (0=empty, 1..4 correspond to TERRANT1..4)
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
        # Normalize to uint8; scale float 0-1 or uint8 0/1 to 0-255
        if np.issubdtype(arr.dtype, np.floating):
            if 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0:
                arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype == np.uint8 and arr.max() <= 1:
            arr = (arr * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Convert black background
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
    sprite_keys_to_convert = [
        SpriteIdx.ENEMY_ORANGE, SpriteIdx.ENEMY_GREEN, SpriteIdx.FUEL_TANK, SpriteIdx.ENEMY_UFO,
        SpriteIdx.SHIP_IDLE, SpriteIdx.SHIP_THRUST, SpriteIdx.SHIP_THRUST_BACK, SpriteIdx.SHIP_CRASH,
        SpriteIdx.SHIP_BULLET, SpriteIdx.ENEMY_BULLET,
        SpriteIdx.DIGIT_0, SpriteIdx.DIGIT_1, SpriteIdx.DIGIT_2, SpriteIdx.DIGIT_3, SpriteIdx.DIGIT_4,
        SpriteIdx.DIGIT_5, SpriteIdx.DIGIT_6, SpriteIdx.DIGIT_7, SpriteIdx.DIGIT_8, SpriteIdx.DIGIT_9,
        SpriteIdx.HP_UI,
        SpriteIdx.PLANET1, SpriteIdx.PLANET2, SpriteIdx.PLANET3, SpriteIdx.PLANET4,
        SpriteIdx.REACTOR,
        SpriteIdx.SPAWN_LOC, 
        SpriteIdx.OBSTACLE,
        SpriteIdx.ENEMY_SAUCER,
    ]
    for key in sprite_keys_to_convert:
        if pygame_sprites[key] is not None:
            jax_sprites[int(key)] = surface_to_jax(pygame_sprites[key])
            
    return jax_sprites

def load_sprites_tuple() -> tuple:
    names = [
        "spaceship",            # 0  SHIP_IDLE
        "ship_thrust",          # 1  SHIP_THRUST
        "ship_bullet",          # 2  SHIP_BULLET 
        "enemy_bullet",         # 3  ENEMY_BULLET
        "enemy_green_bullet",   # 4  ENEMY_GREEN_BULLET

        "enemy_orange",         # 5  ENEMY_ORANGE
        "enemy_green",          # 6  ENEMY_GREEN
        "saucer",               # 7  ENEMY_SAUCER
        "UFO",                  # 8  ENEMY_UFO

        "enemy_crash",          # 9  ENEMY_CRASH
        "saucer_crash",         # 10 SAUCER_CRASH
        "ship_crash",           # 11 SHIP_CRASH

        "fuel_tank",            # 12 FUEL_TANK
        "obstacle",             # 13 OBSTACLE
        "spawn_location",       # 14 SPAWN_LOC

        "reactor",              # 15 REACTOR
        "reactor_terrant",      # 16 REACTOR_TERR
        "terrant1",             # 17 TERRANT1
        "terrant2",             # 18 TERRANT2
        "terrant_3",            # 19 TERRANT3
        "terrant_4",            # 20 TERRANT4

        "planet1",              # 21 PLANET1
        "planet2",              # 22 PLANET2
        "planet3",              # 23 PLANET3
        "planet4",              # 24 PLANET4

        "reactor_destination",  # 25 REACTOR_DEST
        "score",                # 26 SCORE_UI
        "HP",                   # 27 HP_UI

        "ship_thrust_back",     # 28  SHIP_THRUST_BACK
    ]
    base = [_opt(n) for n in names]

    # ------- Digits (0..9): strictly use score_0 ... score_9 -------
    digits = []
    for d in range(10):
        spr = _opt(f"score_{d}")
        if spr is None:
            surf = pygame.Surface((8, 12), pygame.SRCALPHA)
            digits.append(surf)
        else:
            digits.append(spr)

    return tuple(base + digits)

# Initializes an empty bullet pool
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
        flip_y=jnp.zeros((MAX_ENEMIES,), dtype=bool),
        death_timer=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),
        hp=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32)
    )


@jax.jit
def create_env_state(rng: jnp.ndarray) -> EnvState:
    return EnvState(
        mode=jnp.int32(1),
        state=ShipState(
            x=jnp.array(100.0),
            y=jnp.array(100.0),
            vx=jnp.array(0.0),
            vy=jnp.array(0.0),
            angle=jnp.array(0.0),
        ),
        bullets=create_empty_bullets_64(),
        cooldown=jnp.array(0, dtype=jnp.int32),
        enemies=create_empty_enemies(),
        enemy_bullets=create_empty_bullets_16(),
        fire_cooldown=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),
        key=rng,
        key_alt=rng,
        score=jnp.array(0.0),
        done=jnp.array(False),
        lives=jnp.array(6, dtype=jnp.int32), 
        respawn_shift_x=jnp.float32(0.0),   
        reactor_dest_active=jnp.array(False), 
        reactor_dest_x=jnp.float32(0.0),
        reactor_dest_y=jnp.float32(0.0),
        reactor_dest_radius=jnp.float32(0.25), 
        saucer=make_default_saucer(),
        saucer_spawn_timer=jnp.int32(SAUCER_SPAWN_DELAY_FRAMES),
        map_return_x=jnp.float32(0.0),
        map_return_y=jnp.float32(0.0),
        ufo=make_empty_ufo(),
        ufo_used=jnp.array(False),
        ufo_home_x=jnp.float32(0.0),
        ufo_home_y=jnp.float32(0.0),
        ufo_bullets=create_empty_bullets_16(),
    )

@jax.jit
def make_level_start_state(level_id: int) -> ShipState:
    START_Y = jnp.float32(30.0)
    
    x = jnp.array(WINDOW_WIDTH / 2, dtype=jnp.float32)
    y = jnp.array(START_Y, dtype=jnp.float32)
    
    angle = jnp.array(jnp.pi / 2, dtype=jnp.float32)
    
    is_reactor = (jnp.asarray(level_id, dtype=jnp.int32) == 4)
    x = jnp.where(is_reactor, x - 60.0, x)
    
    return ShipState(x=x, y=y, vx=jnp.float32(0.0), vy=jnp.float32(0.0), angle=angle)

# ========== Update Bullets ==========
@jax.jit
def update_bullets(bullets: Bullets) -> Bullets:
    new_x = bullets.x + bullets.vx
    new_y = bullets.y + bullets.vy

    valid_x = (new_x >= 0) & (new_x <= WINDOW_WIDTH)
    valid_y = (new_y >= HUD_HEIGHT) & (new_y <= WINDOW_HEIGHT)
    valid = valid_x & valid_y & bullets.alive

    return Bullets(
        x=new_x,
        y=new_y,
        vx=bullets.vx,
        vy=bullets.vy,
        alive=valid
    )


# ========== Merge Bullets ==========
@jax.jit
def merge_bullets(prev: Bullets, add: Bullets, max_len: int | None = None) -> Bullets:
    # provided, default to the capacity of the previous pool (static shape)
    cap = prev.x.shape[0] if (max_len is None) else max_len    
    cap_i = jnp.int32(cap)
    
    # First, enforce the cap on the old bullet pool to prioritize keeping older bullets
    prev0 = _enforce_cap_keep_old(prev, cap_i)  
    used = jnp.minimum(_bullets_alive_count(prev0), cap_i)
    space_left0 = cap_i - used
    def place_one(carry, i):
        b, space_left = carry
        take = add.alive[i] & (space_left > 0)
        
        # Find the first available empty slot
        dead_mask = ~b.alive
        has_slot  = jnp.any(dead_mask)
        take      = take & has_slot
        idx       = jnp.argmax(dead_mask.astype(jnp.int32))
        b2 = jax.lax.cond(
            take,
            lambda _: Bullets(
                x=b.x.at[idx].set(add.x[i]),
                y=b.y.at[idx].set(add.y[i]),
                vx=b.vx.at[idx].set(add.vx[i]),
                vy=b.vy.at[idx].set(add.vy[i]),
                alive=b.alive.at[idx].set(True),
            ),
            lambda _: b,
            operand=None
        )
        space2 = jnp.where(take, space_left - 1, space_left)
        return (b2, space2), None
    n_add = add.x.shape[0]   # Static length
    (merged, _), _ = jax.lax.scan(place_one, (prev0, space_left0), jnp.arange(n_add))
    
    # Re-enforce the cap to ensure the total number of alive bullets does not exceed the capacity
    merged = _enforce_cap_keep_old(merged, cap_i)
    return merged

@jax.jit
def _bullets_alive_count(bullets: Bullets):
    return jnp.asarray(jnp.sum(bullets.alive.astype(jnp.int32)), dtype=jnp.int32)

@jax.jit
def _enforce_cap_keep_old(b: Bullets, cap: int) -> Bullets:
    cap_i = jnp.int32(cap)
    rank = jnp.cumsum(b.alive.astype(jnp.int32)) - 1  # Sequential number for each alive bullet (0,1,2,...)
    keep = b.alive & (rank < cap_i)
    return Bullets(x=b.x, y=b.y, vx=b.vx, vy=b.vy, alive=keep)

# ========== Fire Bullet ==========
# Fires a new bullet
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

    def skip_bullet(_):
        return bullets

    can_fire = jnp.any(bullets.alive == False)
    return jax.lax.cond(can_fire, add_bullet, skip_bullet, operand=None)

@jax.jit
def _fire_single_from_to(bullets: Bullets, sx, sy, tx, ty, speed=jnp.float32(0.7)) -> Bullets:
    dx = tx - sx
    dy = ty - sy
    d = jnp.maximum(jnp.sqrt(dx*dx + dy*dy), 1e-3)
    vx = speed * dx / d
    vy = speed * dy / d
    one = Bullets(
        x=jnp.array([sx], dtype=jnp.float32),
        y=jnp.array([sy], dtype=jnp.float32),
        vx=jnp.array([vx], dtype=jnp.float32),
        vy=jnp.array([vy], dtype=jnp.float32),
        alive=jnp.array([True])
    )
    return merge_bullets(bullets, one, max_len=16)

# ========== Ship Collision Utilities ==========
# Ship collision logic
@jax.jit
def check_ship_crash(state: ShipState, enemies: Enemies, hitbox_size: float) -> bool:
    sx1 = state.x - hitbox_size
    sx2 = state.x + hitbox_size
    sy1 = state.y - hitbox_size
    sy2 = state.y + hitbox_size

    ex1 = enemies.x - enemies.w / 2
    ex2 = enemies.x + enemies.w / 2
    ey1 = enemies.y - enemies.h / 2
    ey2 = enemies.y + enemies.h / 2

    overlap_x = (sx1 <= ex2) & (sx2 >= ex1)
    overlap_y = (sy1 <= ey2) & (sy2 >= ey1)

    return jnp.any(overlap_x & overlap_y)

@jax.jit
def check_ship_enemy_collisions(ship: ShipState, enemies: Enemies, ship_radius: float) -> jnp.ndarray:
    # Treat enemy coordinates as the center of a rectangle
    enemy_half_w = enemies.w / 2
    enemy_half_h = enemies.h / 2
    
    # Calculate the distance between the ship's center and each enemy's center
    delta_x = ship.x - enemies.x
    delta_y = ship.y - enemies.y
    
    # Clamp the distance to the enemy's rectangular bounds
    clamped_x = jnp.clip(delta_x, -enemy_half_w, enemy_half_w)
    clamped_y = jnp.clip(delta_y, -enemy_half_h, enemy_half_h)
    
    # Find the vector from the ship's center to the closest point on the rectangle
    closest_point_dx = delta_x - clamped_x
    closest_point_dy = delta_y - clamped_y
    
    # Calculate the squared distance
    distance_sq = closest_point_dx**2 + closest_point_dy**2
    
    # A collision occurs if the distance is less than the ship's radius
    # Also ensure the enemy is "alive" (width > 0)）
    collided_mask = (distance_sq < ship_radius**2) & (enemies.w > 0.0)
    return collided_mask

@jax.jit
def check_ship_hit(state: ShipState, bullets: Bullets, hitbox_size: float) -> bool:
    sx1 = state.x - hitbox_size
    sx2 = state.x + hitbox_size
    sy1 = state.y - hitbox_size
    sy2 = state.y + hitbox_size
    within_x = (bullets.x >= sx1) & (bullets.x <= sx2)
    within_y = (bullets.y >= sy1) & (bullets.y <= sy2)
    return jnp.any(within_x & within_y & bullets.alive)

@jax.jit
def check_enemy_hit(bullets: Bullets, enemies: Enemies) -> Tuple[Bullets, Enemies]:
    #1. Perform all collision detection calculations first
    padding = 0.1
    ex1 = enemies.x - enemies.w / 2 - padding
    ex2 = enemies.x + enemies.w / 2 + padding
    ey1 = enemies.y - enemies.h / 2 - padding
    ey2 = enemies.y + enemies.h / 2
    bx = bullets.x[:, None]
    by = bullets.y[:, None]
    cond_x = (bx >= ex1) & (bx <= ex2)
    cond_y = (by >= ey1) & (by <= ey2)
    hit_matrix = cond_x & cond_y & bullets.alive[:, None] & (enemies.w > 0)[:, None].T
    bullet_hit = jnp.any(hit_matrix, axis=1)
    enemy_hit = jnp.any(hit_matrix, axis=0)
    
    #2. Update bullet states
    new_bullets = Bullets(
        x=bullets.x,
        y=bullets.y,
        vx=bullets.vx,
        vy=bullets.vy,
        alive=bullets.alive & (~bullet_hit)
    )
    
    #3. Calculate all the new values for the enemies to be updated externally
    # a) Calculate the new HP after being hit
    hp_after_hit = enemies.hp - jnp.where(enemy_hit, 1, 0)
    
    # b) Determine which enemies have "just died"
    was_alive = (enemies.hp > 0)
    is_dead_now = (hp_after_hit <= 0)
    just_died = was_alive & is_dead_now
    
    # c) Calculate the updated death timer
    death_timer_after_hit = jnp.where(
        just_died,
        ENEMY_EXPLOSION_FRAMES,
        enemies.death_timer
    )
    
    #4. Finally, create a new Enemies object with the pre-calculated new values in a single step
    new_enemies = Enemies(
        x=enemies.x,
        y=enemies.y,
        w=enemies.w, # Width and height remain unchanged here
        h=enemies.h,
        vx=enemies.vx,
        sprite_idx=enemies.sprite_idx,
        flip_y=enemies.flip_y, 
        death_timer=death_timer_after_hit,
        hp=hp_after_hit
    )

    return new_bullets, new_enemies

@jax.jit
def terrain_hit(env_state: EnvState, x: jnp.ndarray, y: jnp.ndarray, radius=jnp.float32(0.3)) -> jnp.ndarray:
    offset_x, offset_y = env_state.level_offset[0], env_state.level_offset[1]
    adjusted_x, adjusted_y = x - offset_x, y - offset_y
    H, W = env_state.terrain_bank.shape[1], env_state.terrain_bank.shape[2]
    
    xi = jnp.clip(jnp.round(adjusted_x).astype(jnp.int32), 0, W - 1)
    yi = jnp.clip(jnp.round(adjusted_y).astype(jnp.int32), 0, H - 1)
    
    RMAX = 16
    dx = jnp.arange(-RMAX, RMAX + 1, dtype=jnp.int32)
    dy = jnp.arange(-RMAX, RMAX + 1, dtype=jnp.int32)
    xs = jnp.clip(xi + dx, 0, W - 1)
    ys = jnp.clip(yi + dy, 0, H - 1)
    
    bi = jnp.clip(env_state.terrain_bank_idx, 0, env_state.terrain_bank.shape[0] - 1)
    page = env_state.terrain_bank[bi]
    
    patch = page[ys[:, None], xs[None, :]]
    
    dxf, dyf = dx.astype(jnp.float32), dy.astype(jnp.float32)
    dist2 = dyf[:, None]**2 + dxf[None, :]**2
    r_eff = jnp.minimum(jnp.float32(radius), jnp.float32(RMAX))
    mask = dist2 <= (r_eff**2)
    is_not_black = jnp.sum(patch, axis=-1) > 0
    
    return jnp.any(is_not_black & mask)

@jax.jit
def consume_ship_hits(state, bullets, hitbox_size):
    # Ship's collision radius
    hs    = jnp.asarray(hitbox_size, dtype=jnp.float32)
    eff_r = hs + jnp.float32(0.04)
    hit_mask = bullets.alive & _segment_hits_circle(
        bullets.x, bullets.y, bullets.vx, bullets.vy,
        state.x,     state.y,   eff_r
    )
    any_hit = jnp.any(hit_mask)
    new_bullets = Bullets(
        x=bullets.x, y=bullets.y,
        vx=bullets.vx, vy=bullets.vy,
        alive=bullets.alive & (~hit_mask)    # Eliminate hit bullets
    )
    return new_bullets, any_hit

@jax.jit
def kill_bullets_hit_terrain_segment(prev: Bullets, nxt: Bullets, terrain_mask: jnp.ndarray,
                                     samples: int = 4) -> Bullets:
    H, W = terrain_mask.shape
    def body(i, acc_hit):
        t = jnp.float32(i) / jnp.float32(samples - 1)  # 0..1
        xs = prev.x + t * (nxt.x - prev.x)
        ys = prev.y + t * (nxt.y - prev.y)
        xi = jnp.clip(xs.astype(jnp.int32), 0, jnp.int32(W - 1))
        yi = jnp.clip(ys.astype(jnp.int32), 0, jnp.int32(H - 1))
        hit_i = terrain_mask[yi, xi] > 0
        return acc_hit | hit_i
    init = jnp.zeros_like(prev.alive, dtype=jnp.bool_)
    hits = jax.lax.fori_loop(0, samples, body, init)
    alive = nxt.alive & (~hits) & prev.alive  # Only active bullets are considered
    return Bullets(x=nxt.x, y=nxt.y, vx=nxt.vx, vy=nxt.vy, alive=alive)

# ========== Ship Step ==========
# Ship movement
@jax.jit
def ship_step(state: ShipState,
              action: int,
              window_size: tuple[int, int],
              hud_height: int) -> ShipState:
    rotation_speed = 0.03
    thrust_power = 0.03
    gravity = 0.01
    
    bounce_damping = 0.2  # Damping factor for bounce velocity
    damping_factor = 0.99
    max_speed = 0.6

    vx = state.vx * damping_factor
    vy = state.vy * damping_factor

    rotate_right_actions = jnp.array([3, 6, 8, 11, 14, 16])
    rotate_left_actions = jnp.array([4, 7, 9, 12, 15, 17])
    thrust_actions = jnp.array([2, 6, 7, 10, 14, 15])
    down_thrust_actions = jnp.array([5, 8, 9, 13, 16, 17])
    right = jnp.isin(action, rotate_right_actions)
    left = jnp.isin(action, rotate_left_actions)
    thrust = jnp.isin(action, thrust_actions)
    down_thrust = jnp.isin(action, down_thrust_actions)

    angle = jnp.where(right, state.angle + rotation_speed, state.angle)
    angle = jnp.where(left, angle - rotation_speed, angle)

    vx = jnp.where(thrust, state.vx + jnp.cos(angle) * thrust_power, state.vx)
    vy = jnp.where(thrust, state.vy + jnp.sin(angle) * thrust_power, state.vy)
    vx = jnp.where(down_thrust, vx - jnp.cos(angle) * thrust_power, vx)
    vy = jnp.where(down_thrust, vy - jnp.sin(angle) * thrust_power, vy)
    vy += gravity
    speed_sq = vx**2 + vy**2
    def cap_velocity(v_tuple):
        v_x, v_y, spd_sq = v_tuple
        speed = jnp.sqrt(spd_sq)
        scale = max_speed / speed
        return v_x * scale, v_y * scale
    
    def no_op(v_tuple):
        return v_tuple[0], v_tuple[1]

    vx, vy = jax.lax.cond(
        speed_sq > max_speed**2,
        cap_velocity,
        no_op,
        (vx, vy, speed_sq)
    )

    next_x_unclipped = state.x + vx
    next_y_unclipped = state.y + vy
    ship_half_size = 2
    window_width, window_height = window_size
   
    # --- Boundary collision and bounce logic ---
    # Horizontal bounce
    hit_left = next_x_unclipped < ship_half_size
    hit_right = next_x_unclipped > window_width - ship_half_size
    
    # Store old vx and vy for bounce calculation
    old_vx = vx
    old_vy = vy
    
    # Apply bounce to velocities
    vx = jnp.where(hit_left | hit_right, -old_vx * bounce_damping, old_vx)
    
    # Update position based on potentially bounced velocity, then clip
    x = state.x + vx  # Use the (potentially bounced) new vx
    x = jnp.clip(x, ship_half_size, window_width - ship_half_size)  # Clip to ensure it's precisely at the edge
    
    # Vertical bounce
    hit_top = next_y_unclipped < hud_height + ship_half_size
    hit_bottom = next_y_unclipped > window_height - ship_half_size
    vy = jnp.where(hit_top | hit_bottom, -old_vy * bounce_damping, old_vy)
    
    # Update position based on potentially bounced velocity, then clip
    y = state.y + vy  # Use the (potentially bounced) new vy
    y = jnp.clip(y, hud_height + ship_half_size, window_height - ship_half_size)
    return ShipState(x=x, y=y, vx=vx, vy=vy, angle=angle)


# ========== Logic about saucer ==========
@jax.jit
def _get_reactor_center(px, py, pi) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    REACTOR = jnp.int32(int(SpriteIdx.REACTOR))
    mask = (pi == REACTOR)
    any_reactor = jnp.any(mask)
    idx = jnp.argmax(mask.astype(jnp.int32))
    rx = jax.lax.cond(any_reactor, lambda _: px[idx], lambda _: jnp.float32(WINDOW_WIDTH * 0.18), operand=None)
    ry = jax.lax.cond(any_reactor, lambda _: py[idx], lambda _: jnp.float32(WINDOW_HEIGHT* 0.43), operand=None)
    return rx, ry, any_reactor

@jax.jit
def _spawn_saucer_at(x, y, towards_x, towards_y, speed=jnp.float32(0.8)) -> SaucerState:
    dx = towards_x - x
    dy = towards_y - y
    d = jnp.maximum(jnp.sqrt(dx*dx + dy*dy), 1e-3)
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
    d = jnp.maximum(jnp.sqrt(dx*dx + dy*dy), 1e-3)
    vx = speed * dx / d
    vy = speed * dy / d
    return s._replace(x=s.x + vx, y=s.y + vy, vx=vx, vy=vy)

@jax.jit
def _saucer_fire_one(
    sauc: SaucerState,
    ship_x: jnp.ndarray,
    ship_y: jnp.ndarray,
    prev_enemy_bullets: Bullets,
    mode_timer: jnp.ndarray,
) -> Bullets:
    can_fire = sauc.alive & ((mode_timer % SAUCER_FIRE_INTERVAL_FRAMES) == 0) \
               & (_bullets_alive_count(prev_enemy_bullets) < jnp.int32(1))
    def do_fire(_):
        merged = _fire_single_from_to(
            prev_enemy_bullets,
            sauc.x, sauc.y,
            ship_x, ship_y,
            SAUCER_BULLET_SPEED
        )
        return _enforce_cap_keep_old(merged, cap=1)
    return jax.lax.cond(can_fire, do_fire, lambda _: prev_enemy_bullets, operand=None)

@jax.jit
def _circle_hit(ax, ay, ar, bx, by, br) -> jnp.ndarray:
    dx = ax - bx
    dy = ay - by
    return (dx*dx + dy*dy) <= (ar+br)*(ar+br)

@jax.jit
def _segment_hits_circle(bx, by, vx, vy, cx, cy, r):
    # Bullet's previous position p0 = p1 - v
    px0 = bx - vx
    py0 = by - vy
    dx  = vx
    dy  = vy
    # Find the point on the line segment with parameter t* ∈ [0,1] that is closest to the circle's center
    a   = dx*dx + dy*dy + 1e-6
    t   = jnp.clip(-(((px0 - cx)*dx + (py0 - cy)*dy) / a), 0.0, 1.0)
    qx  = px0 + t*dx
    qy  = py0 + t*dy
    d2  = (qx - cx)*(qx - cx) + (qy - cy)*(qy - cy)
    return d2 <= (r*r)

@jax.jit
def _bullets_hit_saucer(bullets: Bullets, sauc: SaucerState):
    eff_r = SAUCER_RADIUS
    hit_mask = bullets.alive & _segment_hits_circle(
        bullets.x, bullets.y, bullets.vx, bullets.vy,
        sauc.x,     sauc.y,     eff_r
    )
    any_hit = jnp.any(hit_mask)
    new_bullets = Bullets(
        x=bullets.x, y=bullets.y,
        vx=bullets.vx, vy=bullets.vy,
        alive=bullets.alive & (~hit_mask)    # Eliminate hit bullets
    )
    return new_bullets, any_hit

@jax.jit
def _bullets_hit_ufo(bullets: Bullets, ufo) -> Tuple[Bullets, jnp.ndarray]:
    eff_r = SAUCER_RADIUS
    hit_mask = bullets.alive & _segment_hits_circle(
        bullets.x, bullets.y, bullets.vx, bullets.vy,
        ufo.x,     ufo.y,     eff_r
    )
    any_hit = jnp.any(hit_mask)
    new_bullets = Bullets(
        x=bullets.x, y=bullets.y,
        vx=bullets.vx, vy=bullets.vy,
        alive=bullets.alive & (~hit_mask) 
    )
    return new_bullets, any_hit

# ========== Enemy Step ==========
# Enemy Movement
@jax.jit
def enemy_step(enemies: Enemies, window_width: int) -> Enemies:
    x = enemies.x + enemies.vx
    left_hit = x <= 0
    right_hit = (x + enemies.w) >= window_width
    hit_edge = left_hit | right_hit
    vx = jnp.where(hit_edge, -enemies.vx, enemies.vx)
    return Enemies(x=x, y=enemies.y, w=enemies.w, h=enemies.h, vx=vx, sprite_idx=enemies.sprite_idx, flip_y=enemies.flip_y, death_timer=enemies.death_timer, hp=enemies.hp)


# ========== Enemy Fire ==========
@jax.jit
def enemy_fire(enemies: Enemies,
               ship_x: float,
               ship_y: float,
               enemy_bullet_speed: float,
               fire_cooldown: jnp.ndarray,  # shape should match len(enemies.x)
               fire_interval: int,
               key: jax.random.PRNGKey
               ) -> tuple[Bullets, jnp.ndarray, jax.random.PRNGKey]:
    ex_center = enemies.x + enemies.w / 2
    ey_center = enemies.y - enemies.h / 2
    dx = ship_x - ex_center                 # shape=(N,)
    dy = ship_y - ey_center                 # shape=(N,)
    dist = jnp.sqrt(dx ** 2 + dy ** 2)
    dist = jnp.where(dist < 1e-3, 1.0, dist)
    vx = dx / dist * enemy_bullet_speed
    vy = dy / dist * enemy_bullet_speed
    
    # 1. Determine which turrets "should" fire in this frame
    alive_mask = (enemies.w > 0.0) & (enemies.death_timer == 0)
    should_fire = (fire_cooldown == 0) & alive_mask
    
    # 2. Calculate the cooldown for the "next frame"
    # - If a turret fired this frame (`should_fire` is True), its cooldown is reset to `fire_interval`
    # - Otherwise, the cooldown remains unchanged (since the decrement happens in `_step_level_core`)
    new_fire_cooldown = jnp.where(should_fire, fire_interval, fire_cooldown)
    x_out = jnp.where(should_fire, ex_center, -1.0)
    y_out = jnp.where(should_fire, ey_center, -1.0)
    vx_out = jnp.where(should_fire, vx, 0.0)
    vy_out = jnp.where(should_fire, vy, 0.0)
    bullets_out = Bullets(
        x=x_out,
        y=y_out,
        vx=vx_out,
        vy=vy_out,
        alive=should_fire
    )
    return bullets_out, new_fire_cooldown, key


# ========== Collision Detection ==========
@jax.jit
def check_collision(bullets: Bullets, enemies: Enemies):
    def bullet_hits_enemy(i, carry):            # `carry` is the cumulative result, a boolean array of shape (MAX_BULLETS,)
        x = bullets.x[i]                        # x, y are the current bullet coordinates
        y = bullets.y[i]
        alive = bullets.alive[i]

        def check_each_enemy(j, hit):
            # Rectangle bounding box, Enemy's bounding box: (x, x+w), (y, y+h)
            within_x = (x > enemies.x[j]) & (x < enemies.x[j] + enemies.w[j])
            within_y = (y > enemies.y[j]) & (y < enemies.y[j] + enemies.h[j])
            return hit | (within_x & within_y)
        hit_any = jax.lax.fori_loop(0, MAX_ENEMIES, check_each_enemy, False)
        return carry.at[i].set(hit_any & alive)
    hits = jnp.zeros((MAX_BULLETS,), dtype=bool)
    hits = jax.lax.fori_loop(0, MAX_BULLETS, bullet_hits_enemy, hits)
    return hits


# ========== Step Core Map ==========
@jax.jit
def step_core_map(state: ShipState,
                  action: int,
                  window_size: Tuple[int, int],
                  hud_height: int
                  ) -> Tuple[jnp.ndarray, ShipState, float, bool, dict, bool, int]:
    new_state = ship_step(state, action, window_size, hud_height)

    obs = jnp.array([
        new_state.x,
        new_state.y,
        new_state.vx,
        new_state.vy,
        new_state.angle
    ])
    reward = 0.0
    done = False
    info = {}
    planet_x = jnp.array([60.0, 120.0, 200.0])
    planet_y = jnp.array([120.0, 200.0, 80.0])
    planet_r = jnp.array([3, 3, 3])
    level_ids = jnp.array([0, 1, 2])
    dx = planet_x - new_state.x
    dy = planet_y - new_state.y
    dists = jnp.sqrt(dx ** 2 + dy ** 2)
    within_planet = dists < planet_r
    reset = jnp.any(within_planet)
    level_idx = jnp.argmax(within_planet)
    level = jnp.where(reset, level_ids[level_idx], -1)
    return obs, new_state, reward, done, info, reset, level

# ========== Step Core Level Skeleton ==========
@jax.jit
def terrain_hit_mask(mask: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, radius: float = 2) -> jnp.ndarray:
    H, W = mask.shape
    R_MAX = 8
    r = jnp.int32(jnp.clip(radius, 1.0, float(R_MAX)))
    dx_full = jnp.arange(-R_MAX, R_MAX + 1, dtype=jnp.int32)
    dy_full = jnp.arange(-R_MAX, R_MAX + 1, dtype=jnp.int32)
    DX, DY = jnp.meshgrid(dx_full, dy_full, indexing='xy')
    valid = (jnp.abs(DX) <= r) & (jnp.abs(DY) <= r) & ((DX * DX + DY * DY) <= (r * r))
    mx = jnp.floor(x).astype(jnp.int32)
    my = jnp.floor(y).astype(jnp.int32)
    sx = jnp.clip(mx + DX, 0, W - 1)
    sy = jnp.clip(my + DY, 0, H - 1)
    samples = mask[sy, sx].astype(jnp.uint8)
    samples = jnp.where(valid, samples, 0)
    return jnp.any(samples > 0)

@jax.jit
def step_map(env_state: EnvState, action: int):
    # --- 1. State Preparation ---
    # Check if the ship was crashing in the previous frame
    was_crashing = env_state.crash_timer > 0
   
    # --- 2. Ship Movement and Player Firing ---
    # If the ship is crashing, ignore player input and set velocity to zero to "freeze" the ship
    ship_state_before_move = env_state.state._replace(
        vx=jnp.where(was_crashing, 0.0, env_state.state.vx),
        vy=jnp.where(was_crashing, 0.0, env_state.state.vy)
    )
    actual_action = jnp.where(was_crashing, NOOP, action)
    ship_after_move = ship_step(ship_state_before_move, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT)
    can_fire = jnp.isin(action, jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])) & (env_state.cooldown == 0) & (_bullets_alive_count(env_state.bullets) < 2)
    bullets = jax.lax.cond(can_fire, lambda b: fire_bullet(b, ship_after_move.x, ship_after_move.y, ship_after_move.angle, 2), lambda b: b, env_state.bullets)
    bullets = update_bullets(bullets)
    cooldown = jnp.where(can_fire, 5, jnp.maximum(env_state.cooldown - 1, 0))
    
    # Initialize a temporary `env_state` for subsequent chained updates
    new_env = env_state._replace(state=ship_after_move, bullets=bullets, cooldown=cooldown)
    
    # --- 3. Saucer Logic ---
    saucer = new_env.saucer
    timer = new_env.saucer_spawn_timer
    should_tick_timer = (new_env.mode == 0) & (~saucer.alive) & (saucer.death_timer == 0)
    timer = jnp.where(should_tick_timer, jnp.maximum(timer - 1, 0), timer)
    
    rx, ry, has_reactor = _get_reactor_center(new_env.planets_px, new_env.planets_py, new_env.planets_pi)
    should_spawn = (timer == 0) & (~saucer.alive) & has_reactor
    saucer = jax.lax.cond(should_spawn, lambda: _spawn_saucer_at(rx, ry, new_env.state.x, new_env.state.y, SAUCER_SPEED_MAP), lambda: saucer)
    timer = jnp.where(should_spawn, 99999, timer)
    
    saucer_after_move = jax.lax.cond(saucer.alive, lambda s: _update_saucer_seek(s, new_env.state.x, new_env.state.y, SAUCER_SPEED_MAP), lambda s: s, operand=saucer)
    bullets_after_hit, hit_any_bullet = _bullets_hit_saucer(new_env.bullets, saucer_after_move)
    sauc_after_hp = saucer_after_move._replace(hp=saucer_after_move.hp - jnp.where(hit_any_bullet, 1, 0))
    just_died = (saucer_after_move.hp > 0) & (sauc_after_hp.hp <= 0) & saucer_after_move.alive
    
    timer = jnp.where(just_died, SAUCER_RESPAWN_DELAY_FRAMES, timer)
    sauc_final = sauc_after_hp._replace(
        alive=sauc_after_hp.hp > 0,
        death_timer=jnp.where(just_died, SAUCER_EXPLOSION_FRAMES, jnp.maximum(saucer_after_move.death_timer - 1, 0))
    )
    mode_timer = jnp.where(new_env.mode == 0, new_env.mode_timer + 1, 0)
    
    enemy_bullets = _saucer_fire_one(sauc_final, new_env.state.x, new_env.state.y, new_env.enemy_bullets, mode_timer)
    enemy_bullets = update_bullets(enemy_bullets)
    new_env = new_env._replace(
        bullets=bullets_after_hit, saucer=sauc_final, saucer_spawn_timer=timer,
        enemy_bullets=enemy_bullets, mode_timer=mode_timer
    )
    
    # --- 4. Collision and State Finalization ---
    # a) Saucer bullet hits ship
    enemy_bullets_after_hit, hit_ship_by_bullet = consume_ship_hits(new_env.state, new_env.enemy_bullets, SHIP_RADIUS)
    new_env = new_env._replace(enemy_bullets=enemy_bullets_after_hit)
    
    # b) Ship collides with an obstacle
    px, py, pr, pi, pid = new_env.planets_px, new_env.planets_py, new_env.planets_pr, new_env.planets_pi, new_env.planets_id
    dx, dy = px - new_env.state.x, py - new_env.state.y
    dist2 = dx*dx + dy*dy
    hit_obstacle = jnp.any((pi == SpriteIdx.OBSTACLE) & (dist2 <= (pr + SHIP_RADIUS)**2))
    
    # c) Unify the crash conditions
    ship_should_crash = hit_ship_by_bullet | hit_obstacle
    
    # d) Unify the crash timer logic
    start_crash = ship_should_crash & (~was_crashing)
    crash_timer_next = jnp.where(start_crash, 30, jnp.maximum(new_env.crash_timer - 1, 0))
    new_env = new_env._replace(crash_timer=crash_timer_next)
    is_crashing_now = new_env.crash_timer > 0
    
    # e) Disable other collisions during a crash ("ghost" state)
    allowed = jnp.any(jnp.stack([pi==SpriteIdx.PLANET1, pi==SpriteIdx.PLANET2, pi==SpriteIdx.PLANET3, pi==SpriteIdx.PLANET4, pi==SpriteIdx.REACTOR], 0), axis=0)
    allowed = allowed & (~new_env.planets_cleared_mask)
    is_reactor_and_destroyed = (pi == int(SpriteIdx.REACTOR)) & new_env.reactor_destroyed
    allowed = allowed & (~is_reactor_and_destroyed)
    hit_planet = allowed & (dist2 <= (pr*0.85 + SHIP_RADIUS)**2)
    can_enter_planet = jnp.any(hit_planet) & (~is_crashing_now)
    hit_to_arena = sauc_final.alive & _circle_hit(new_env.state.x, new_env.state.y, SHIP_RADIUS, sauc_final.x, sauc_final.y, SAUCER_RADIUS) & (~is_crashing_now)
    
    def _enter_arena(env):
        W, H = jnp.float32(WINDOW_WIDTH), jnp.float32(WINDOW_HEIGHT)
        return env._replace(
            mode=jnp.int32(2), mode_timer=jnp.int32(0),
            state=env.state._replace(x=W*0.20, y=H*0.50, vx=0.0, vy=0.0),
            saucer=sauc_final._replace(
                x=W*0.80, y=H*0.50, vx=-SAUCER_SPEED_ARENA, vy=0.0,
                hp=SAUCER_INIT_HP, alive=True, death_timer=0
            ),
            map_return_x=env.state.x, map_return_y=env.state.y
        )
    new_env = jax.lax.cond(hit_to_arena, _enter_arena, lambda e: e, new_env)
    
    # f) Only signal a Reset when the animation is finished
    reset_signal_from_crash = (env_state.crash_timer > 0) & (crash_timer_next == 0)
    # --- 5. Final Return Values ---
    hit_idx = jnp.argmax(hit_planet.astype(jnp.int32))
    level_id = jax.lax.cond(can_enter_planet, lambda: pid[hit_idx], lambda: -1)
    should_reset = can_enter_planet | reset_signal_from_crash
    final_level_id = jnp.where(reset_signal_from_crash, -2, level_id)
    
    obs = jnp.array([new_env.state.x, new_env.state.y, new_env.state.vx, new_env.state.vy, new_env.state.angle])
    
    reward = jnp.where(just_died, jnp.float32(300.0), jnp.float32(0.0))
    reward = jnp.where(start_crash & ~hit_obstacle, reward - 10.0, reward)

    info = {
        "crash": start_crash, 
        "hit_by_bullet": hit_ship_by_bullet,
        "reactor_crash_exit": jnp.array(False),
    }
    
    new_env = new_env._replace(score=new_env.score + reward) 
    
    return obs, new_env, reward, jnp.array(False), info, should_reset, final_level_id

@jax.jit
def _step_level_core(env_state: EnvState, action: int):  
    # --- 1. UFO Spawn ---
    def _spawn_ufo_once(env):
        W, H = WINDOW_WIDTH, WINDOW_HEIGHT
        b = env.terrain_bank_idx
        x_frac = jnp.where(b == 1, 0.85, 0.85)
        x_frac = jnp.where(b == 2, 0.15, x_frac)
        x_frac = jnp.where(b == 3, 0.80, x_frac)
        x_frac = jnp.where(b == 4, 0.20, x_frac)
        x0 = x_frac * W
        col_x = jnp.clip(jnp.int32(x0), 0, W - 1)
        bank_idx = jnp.clip(env.terrain_bank_idx, 0, env.terrain_bank.shape[0] - 1)
        terrain_page = env.terrain_bank[bank_idx]
        is_ground_in_col = jnp.sum(terrain_page[:, col_x], axis=-1) > 0
        y_indices = jnp.arange(H, dtype=jnp.int32)
        ground_indices = jnp.where(is_ground_in_col, y_indices, H)
        ground_y = jnp.min(ground_indices)
        safe_y_max = jnp.float32(ground_y) - 20.0
        y0 = jnp.minimum(jnp.float32(HUD_HEIGHT + 48.0), safe_y_max)
        return env._replace(
            ufo=UFOState(x=x0, y=y0, vx=-0.04, vy=0.0, hp=1, alive=True, death_timer=0),
            ufo_used=True, ufo_home_x=x0, ufo_home_y=y0,
            ufo_bullets=create_empty_bullets_16(),
        )
    can_spawn_ufo = (env_state.mode == 1) & (~env_state.ufo_used) & (env_state.terrain_bank_idx != 5)
    state_after_spawn = jax.lax.cond(can_spawn_ufo, _spawn_ufo_once, lambda e: e, env_state)
    
    # --- 2. State Update (Movement & Player Firing) ---
    ship_after_move = ship_step(state_after_spawn.state, action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT)
    can_fire_player = jnp.isin(action, jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])) & (state_after_spawn.cooldown == 0) & (_bullets_alive_count(state_after_spawn.bullets) < 2)
    bullets = jax.lax.cond(can_fire_player, lambda b: fire_bullet(b, ship_after_move.x, ship_after_move.y, ship_after_move.angle, 0.2), lambda b: b, state_after_spawn.bullets)
    cooldown = jnp.where(can_fire_player, 5, jnp.maximum(state_after_spawn.cooldown - 1, 0))
    
    # --- 3. Integrate UFO Logic ---
    # Call the new helper function that returns an `env_state` with partially updated UFO-related states
    state_after_ufo = _update_ufo(state_after_spawn, ship_after_move, bullets)
    # Extract the updated state from the return value
    ufo = state_after_ufo.ufo
    bullets = state_after_ufo.bullets
    ufo_bullets = state_after_ufo.ufo_bullets
    
    # --- 4. Ground Enemy (Turret) Logic ---
    enemies = enemy_step(state_after_ufo.enemies, WINDOW_WIDTH)
    is_exploding = enemies.death_timer > 0
    enemies = enemies._replace(
        death_timer=jnp.maximum(enemies.death_timer - 1, 0),
        w=jnp.where(is_exploding & (enemies.death_timer == 1), 0.0, enemies.w),
        h=jnp.where(is_exploding & (enemies.death_timer == 1), 0.0, enemies.h)
    )

    # === Enemy LOGIC ===
    # 1. Prepare the state for the current frame
    current_fire_cooldown = state_after_ufo.fire_cooldown
    current_key = state_after_ufo.key
    current_enemy_bullets = state_after_ufo.enemy_bullets
    
    # 2. Decide which turrets "can" fire now
    can_fire_globally = _bullets_alive_count(current_enemy_bullets) < 1
    is_turret = (enemies.sprite_idx == int(SpriteIdx.ENEMY_ORANGE)) | \
            (enemies.sprite_idx == int(SpriteIdx.ENEMY_GREEN))
    turrets_ready_mask = (enemies.w > 0) & (current_fire_cooldown == 0) & is_turret
    should_fire_mask = turrets_ready_mask & can_fire_globally
    any_turret_firing = jnp.any(should_fire_mask)
    
    # 3. Calculate the cooldown for the "next frame"
    # First, decrement the cooldown for all turrets
    next_frame_cooldown = jnp.maximum(current_fire_cooldown - 1, 0)
    # Then, for turrets that "just" fired, reset their cooldown to 60
    fire_interval = 60
    next_frame_cooldown = jnp.where(should_fire_mask, fire_interval, next_frame_cooldown)
    
    # 4. If any turrets are firing, generate new bullets
    def _generate_bullets(_):
        ex_center = enemies.x + enemies.w / 2
        ey_center = enemies.y - enemies.h / 2
        dx = ship_after_move.x - ex_center
        dy = ship_after_move.y - ey_center
        dist = jnp.sqrt(dx ** 2 + dy ** 2)
        dist = jnp.where(dist < 1e-3, 1.0, dist)
        vx = dx / dist * 0.4  # enemy_bullet_speed
        vy = dy / dist * 0.4
        x_out = jnp.where(should_fire_mask, ex_center, -1.0)
        y_out = jnp.where(should_fire_mask, ey_center, -1.0)
        vx_out = jnp.where(should_fire_mask, vx, 0.0)
        vy_out = jnp.where(should_fire_mask, vy, 0.0)
        return Bullets(x=x_out, y=y_out, vx=vx_out, vy=vy_out, alive=should_fire_mask)
    
    def _get_empty_bullets(_):
        return create_empty_bullets_16()

    new_enemy_bullets = jax.lax.cond(
        any_turret_firing,
        _generate_bullets,
        _get_empty_bullets,
        operand=None
    )

    # 5. Merge bullets and assign the state to final variables
    enemy_bullets = merge_bullets(current_enemy_bullets, new_enemy_bullets)
    fire_cooldown = next_frame_cooldown
    key = current_key 
    # === Enemy LOGIC Over===

    # --- 5. Advance All Bullets ---
    bullets = update_bullets(bullets)
    enemy_bullets = update_bullets(enemy_bullets)
    ufo_bullets = update_bullets(ufo_bullets)

    # --- 6. Collision Detection ---
    bullets = _bullets_hit_terrain(state_after_ufo, bullets)
    enemy_bullets = _bullets_hit_terrain(state_after_ufo, enemy_bullets)
    ufo_bullets = _bullets_hit_terrain(state_after_ufo, ufo_bullets)
    bullets, enemies = check_enemy_hit(bullets, enemies)
        
    hit_enemy_mask = check_ship_enemy_collisions(ship_after_move, enemies, SHIP_RADIUS)
    enemies = enemies._replace(death_timer=jnp.where(hit_enemy_mask, ENEMY_EXPLOSION_FRAMES, enemies.death_timer))
        
    enemy_bullets, hit_by_enemy_bullet = consume_ship_hits(ship_after_move, enemy_bullets, SHIP_RADIUS)
    ufo_bullets, hit_by_ufo_bullet = consume_ship_hits(ship_after_move, ufo_bullets, SHIP_RADIUS)
    hit_terr = terrain_hit(state_after_ufo, ship_after_move.x, ship_after_move.y, 2)
        
    # --- 7. State Finalization ---
    # a) Initial check for ship death
    hit_enemy_types = jnp.where(hit_enemy_mask, enemies.sprite_idx, -1)
    crashed_on_turret = jnp.any((hit_enemy_types == int(SpriteIdx.ENEMY_ORANGE)) | (hit_enemy_types == int(SpriteIdx.ENEMY_GREEN)))
    dead = crashed_on_turret | hit_by_enemy_bullet | hit_by_ufo_bullet | hit_terr

    # b) Special rules for the Reactor level
    is_in_reactor = (state_after_ufo.current_level == 4)
    is_dest_mask = (enemies.sprite_idx == int(SpriteIdx.REACTOR_DEST))
    win_reactor = jnp.any(hit_enemy_mask & is_dest_mask) & is_in_reactor
    crash_in_reactor = dead & is_in_reactor

    # c) Score calculation
    w_before_hit = state_after_ufo.enemies.w
    just_killed_mask = (w_before_hit > 0) & (enemies.w == 0)
    # -- Score based on enemy type --
    is_orange = enemies.sprite_idx == jnp.int32(int(SpriteIdx.ENEMY_ORANGE))
    is_green  = enemies.sprite_idx == jnp.int32(int(SpriteIdx.ENEMY_GREEN))
    is_fuel   = enemies.sprite_idx == jnp.int32(int(SpriteIdx.FUEL_TANK))

    k_orange = jnp.sum(just_killed_mask & is_orange).astype(jnp.float32)
    k_green  = jnp.sum(just_killed_mask & is_green ).astype(jnp.float32)
    k_fuel   = jnp.sum(just_killed_mask & is_fuel ).astype(jnp.float32) 

    score_from_enemies = 250.0 * k_orange + 350.0 * k_green + 150.0 * k_fuel
    # -- Reactor objective --
    score_from_reactor = jnp.where(win_reactor, 500.0, 0.0)
    # -- UFO: was alive last frame and is now dead (and has an explosion timer) --
    ufo_just_died = (state_after_ufo.ufo.alive == False) & (env_state.ufo.alive == True) & (state_after_ufo.ufo.death_timer > 0)
    score_from_ufo = jnp.where(ufo_just_died, 100.0, 0.0)
    score_delta = score_from_enemies + score_from_reactor + score_from_ufo

    # d) Crash and respawn logic
    was_crashing = state_after_ufo.crash_timer > 0
    start_crash = dead & (~was_crashing)
    crash_timer_next = jnp.where(start_crash, 30, jnp.maximum(state_after_ufo.crash_timer - 1, 0))
    
    # No in-level respawn in the Reactor
    respawn_now = (state_after_ufo.crash_timer == 1) & (~is_in_reactor) 
    lives_next = state_after_ufo.lives - jnp.where(respawn_now, 1, 0)
    
    # e) The final Reset signal
    all_enemies_gone = jnp.all(enemies.w == 0) & (~ufo.alive) & (ufo.death_timer == 0)
    has_meaningful_enemies = jnp.any(w_before_hit > 0)
    reset_level_win = all_enemies_gone & has_meaningful_enemies & (~is_in_reactor)
    # A reset from a reactor crash should only happen AFTER the crash animation is done.
    # The animation is finished when the crash_timer from the *previous* state was 1.
    crash_animation_finished = (state_after_ufo.crash_timer == 1)
    reset_from_reactor_crash = crash_in_reactor & crash_animation_finished

    # Total Reset signal = Normal win OR Reactor win OR (Reactor crash AND animation finished)
    reset = reset_level_win | win_reactor | reset_from_reactor_crash

    # f) Is the game over?
    game_over = respawn_now & (lives_next <= 0)

    # --- 8. Respawn State Transition ---
    def _respawn_level_state(operands):
        s, b, eb, fc, cd = operands  # <-- THIS IS THE FIX: Unpack the arguments here.
        ship_respawn = make_level_start_state(s.current_level)
        ship_respawn = ship_respawn._replace(x=ship_respawn.x + s.respawn_shift_x)
        return (ship_respawn, create_empty_bullets_64(), create_empty_bullets_16(), jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32), 0)

    def _keep_state_no_respawn(operands):
        # This function doesn't use the input operands, but instead directly
        # gets the latest state variables we just calculated from the outer scope
        return (ship_after_move, bullets, enemy_bullets, fire_cooldown, cooldown)

    state, bullets, enemy_bullets, fire_cooldown, cooldown = jax.lax.cond(
        respawn_now & ~game_over,
        _respawn_level_state,
        _keep_state_no_respawn,
        # The operands are now only used to trigger the `_respawn_level_state` branch
        operand=(state_after_ufo, bullets, enemy_bullets, fire_cooldown, cooldown)
    )
    
    # --- 9. Assemble and Return Final State ---
    reactor_destroyed_next = state_after_ufo.reactor_destroyed | win_reactor
    
    current_planet_idx = state_after_ufo.current_level  # level_id usually corresponds to the physical index

    # 1. Get the physical index of the current level
    # If the level is won, update the mask
    cleared_mask_next = jnp.where(
        reset_level_win,
        state_after_ufo.planets_cleared_mask.at[current_planet_idx].set(True),
        state_after_ufo.planets_cleared_mask
    )

    final_env_state = state_after_ufo._replace(
        state=state, bullets=bullets, cooldown=cooldown, enemies=enemies,
        enemy_bullets=enemy_bullets, fire_cooldown=fire_cooldown, key=key,
        ufo=ufo, ufo_bullets=ufo_bullets,
        score=state_after_ufo.score + score_delta,          # Use the merged score
        crash_timer=crash_timer_next,
        lives=lives_next,
        done=game_over,
        reactor_destroyed=reactor_destroyed_next,
        planets_cleared_mask=cleared_mask_next,
        mode_timer=state_after_ufo.mode_timer + 1
    )
    
    obs = jnp.array([state.x, state.y, state.vx, state.vy, state.angle])
    
    crash_animation_finished = (state_after_ufo.crash_timer == 1)
    reset_from_reactor_crash = crash_in_reactor & crash_animation_finished
    info = {
        "crash": start_crash,
        "hit_by_bullet": hit_by_enemy_bullet | hit_by_ufo_bullet,
        "reactor_crash_exit": reset_from_reactor_crash, 
    }
    reward = score_delta                # Let reward equal the score change

    return obs, final_env_state, reward, game_over, info, reset, jnp.int32(-1)

# Note: For radius, the same value is passed to each call.
batched_terrain_hit = jax.vmap(terrain_hit, in_axes=(None, 0, 0, None))

@jax.jit
def step_core_level(env_state: EnvState,
                    state_in: ShipState,
                    action: int,
                    bullets: Bullets,
                    cooldown: int,
                    enemies: Enemies,
                    enemy_bullets: Bullets,
                    fire_cooldown: jnp.ndarray,
                    key: jax.random.PRNGKey,
                    terrain_mask: jnp.ndarray,
                    window_size: Tuple[int, int],
                    hud_height: int,
                    cooldown_max: int,
                    bullet_speed: float,
                    enemy_bullet_speed: float,
                    fire_interval: int) -> Tuple[
    jnp.ndarray, ShipState, Bullets, int, Enemies, Bullets, jnp.ndarray, jax.random.PRNGKey, float, float, bool, bool, int]:
    # Ship movement
    state = ship_step(state_in, action, window_size, hud_height)

    # Handle shooting
    fire_actions = jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])
    is_firing = jnp.isin(action, fire_actions)
    PLAYER_MAX_BULLETS = jnp.int32(2)
    alive_cnt = _bullets_alive_count(bullets) 
    bullets = jax.lax.cond(
        is_firing & (cooldown == 0) & (alive_cnt < PLAYER_MAX_BULLETS),
        lambda _: fire_bullet(bullets, state.x, state.y, state.angle, bullet_speed),
        lambda _: bullets,
        operand=None
    )
    bullets = update_bullets(bullets)
    cooldown = jnp.where(is_firing & (cooldown == 0), cooldown_max, jnp.maximum(cooldown - 1, 0))


    # Player bullet collision with terrain
    player_bullet_radius = 0.4              # Adjustable bullet collision radius
    # Only check for live bullets
    bullets_to_check_mask = bullets.alive
    # Use the batched collision detection function for the bullets' coordinates
    hit_terrain_mask = batched_terrain_hit(env_state, bullets.x, bullets.y, player_bullet_radius)
    # If a bullet is alive and has hit the terrain, it's no longer alive
    new_player_bullets_alive = bullets.alive & ~hit_terrain_mask
    bullets = bullets._replace(alive=new_player_bullets_alive)

    # Enemy movement
    enemies = enemy_step(enemies, window_width=window_size[0])
    is_exploding = enemies.death_timer > 0
    enemies = enemies._replace(
        death_timer=jnp.maximum(enemies.death_timer - 1, 0),
        # Width and height only become 0 after the timer expires
        w=jnp.where(is_exploding & (enemies.death_timer == 1), 0.0, enemies.w),
        h=jnp.where(is_exploding & (enemies.death_timer == 1), 0.0, enemies.h)
    )
    can_fire_globally = _bullets_alive_count(enemy_bullets) < 1 

    def _do_fire(_):
        return enemy_fire(enemies, state.x, state.y, enemy_bullet_speed, fire_cooldown, fire_interval, key)
    def _do_not_fire(_):
        empty_bullets = Bullets(x=jnp.zeros_like(enemies.x), y=jnp.zeros_like(enemies.y), vx=jnp.zeros_like(enemies.vx), vy=jnp.zeros_like(enemies.vx), alive=jnp.zeros_like(enemies.w, dtype=bool))
        new_cooldown = jnp.maximum(fire_cooldown - 1, 0)
        return empty_bullets, new_cooldown, key
    
    new_enemy_bullets, fire_cooldown, key = jax.lax.cond(can_fire_globally, _do_fire, _do_not_fire, operand=None)
    
    # ========== Bullet Merging + Update (Static Length Control) ==========

    def pad_bullets(b: Bullets, target_len=16):
        def pad_array(arr, pad_val=0.0):
            cur_len = arr.shape[0]
            pad_len = max(target_len - cur_len, 0)
            return jnp.pad(arr[:target_len], (0, pad_len), constant_values=pad_val)

        return Bullets(
            x=pad_array(b.x),
            y=pad_array(b.y),
            vx=pad_array(b.vx),
            vy=pad_array(b.vy),
            alive=pad_array(b.alive, pad_val=False)
        )

    def truncate_bullets(b: Bullets, max_len=16):
        return Bullets(
            x=b.x[:max_len],
            y=b.y[:max_len],
            vx=b.vx[:max_len],
            vy=b.vy[:max_len],
            alive=b.alive[:max_len]
        )

    new_enemy_bullets = pad_bullets(new_enemy_bullets, target_len=16)
    enemy_bullets = pad_bullets(enemy_bullets, target_len=16)
    enemy_bullets = merge_bullets(enemy_bullets, new_enemy_bullets)
    enemy_bullets = update_bullets(enemy_bullets)
    enemy_bullets = truncate_bullets(enemy_bullets, max_len=16)

    # Enemy bullet collision with terrain
    enemy_bullet_radius = 0.4
    enemies_to_check_mask = enemy_bullets.alive
    hit_terrain_mask_enemy = batched_terrain_hit(env_state, enemy_bullets.x, enemy_bullets.y, enemy_bullet_radius)
    new_enemy_bullets_alive = enemy_bullets.alive & ~hit_terrain_mask_enemy
    enemy_bullets = enemy_bullets._replace(alive=new_enemy_bullets_alive)
    # Bullet-to-enemy collision detection
    # Save previous enemy width (to check for new deaths)
    w_prev = enemies.w

    # Hit detection
    bullets, enemies = check_enemy_hit(bullets, enemies)

    # Find newly killed enemies (transition from w > 0 to w == 0)
    new_killed = (w_prev > 0.0) & (enemies.w == 0.0)
    score_delta = jnp.sum(new_killed).astype(jnp.float32) * 10.0

    # Enemy or bullet hits ship
    # 1. Call the new collision function to get a mask of hit enemies
    hit_enemy_mask = check_ship_enemy_collisions(state, enemies, ship_radius=SHIP_RADIUS)
    
    # 2. Check if any enemy was hit
    any_enemy_hit = jnp.any(hit_enemy_mask)

    # 3. Find the types (sprite_idx) of the hit enemies
    #    jnp.where(condition, value_if_true, value_if_false)
    hit_enemy_types = jnp.where(hit_enemy_mask, enemies.sprite_idx, -1)

    # 4. Determine if the ship should crash
    # Rule: The ship only crashes if it hits a turret (orange or green)
    crashed_on_turret = jnp.any(
        (hit_enemy_types == int(SpriteIdx.ENEMY_ORANGE)) |
        (hit_enemy_types == int(SpriteIdx.ENEMY_GREEN))
    )

    # 5. Update the ship's death status
    # old crash condition OR new collision with turret condition
    crashed = crashed_on_turret 
    hit_by_bullet = check_ship_hit(state, enemy_bullets, hitbox_size=2)
    hit_terr = terrain_hit(env_state, state.x, state.y, jnp.float32(2.0))
    dead = crashed | hit_by_bullet | hit_terr

    # 6. Make all hit enemies "disappear" (width and height become 0)
    enemies = enemies._replace(
        death_timer=jnp.where(
            hit_enemy_mask, # If hit
            ENEMY_EXPLOSION_FRAMES, # Set the timer to 120
            enemies.death_timer # Otherwise, keep it the same
        )
    )

    # ====== Crash Pipeline (Animation -> Lose Life -> Respawn/Game Over) ======
    CRASH_FRAMES = jnp.int32(30)

    # Is the crash animation already playing? (timer > 0 in the previous frame) 
    was_crashing = env_state.crash_timer > 0
    # Did the ship "just crash" this frame? (start the animation)
    start_crash = (~was_crashing) & dead

    # The next frame's crash timer (if just crashed -> set to 30; otherwise, count down to 0)
    crash_timer_next = jnp.where(
        start_crash, CRASH_FRAMES,
        jnp.maximum(env_state.crash_timer - 1, 0)
    )

    # "Will the ship respawn at the end of this frame?" - If the previous timer was 1, it becomes 0 this frame, triggering respawn/life loss
    respawn_now = (env_state.crash_timer == 1)

    # Only lose a life at the moment of respawn; otherwise, it remains the same
    lives_next = env_state.lives - jnp.where(respawn_now, jnp.int32(1), jnp.int32(0))

    # Are all lives gone? (After losing this one, lives <= 0)
    game_over = respawn_now & (lives_next <= 0)

    # Respawn: Only reset the "player/player bullets/enemy bullets/cooldowns", keep the enemy state
    def _respawn(_):
        respawn_state = make_level_start_state(env_state.current_level)
        # Apply scene-specific offset (reactor is -30, others are 0)
        respawn_state = respawn_state._replace(
            x = respawn_state.x + env_state.respawn_shift_x
        )
        return (
            respawn_state,
            create_empty_bullets_64(),                            # Player bullets cleared
            create_empty_bullets_16(),                            # Enemy bullets cleared
            jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),           # Enemy fire cooldown cleared
            jnp.int32(0),                                         # Player cooldown cleared
        )

    def _keep(_):
        return (state, bullets, enemy_bullets, fire_cooldown, cooldown)

    # Use `lax.cond` to select which set of runtime values to return based on whether a respawn should occur
    state, bullets, enemy_bullets, fire_cooldown, cooldown = jax.lax.cond(
        respawn_now & ~game_over, _respawn, _keep, operand=None
    )

    # Reward: A large penalty for starting a crash, normal decay for other frames
    reward = jnp.where(start_crash, -10.0, -1.0)

    # A win only counts if there were "meaningful enemies" in the level that are now all gone
    has_enemies     = jnp.any(enemies.w > 0.0)
    all_dead        = jnp.all(enemies.w == 0.0)
    reset_level_win = has_enemies & (~start_crash) & (crash_timer_next == 0) & all_dead
    reset = reset_level_win

    # ====== Reactor Objective Check ======
    is_reactor = (env_state.terrain_sprite_idx == jnp.int32(int(SpriteIdx.REACTOR_TERR)))

    # Check for reach (only in the reactor and if the destination is active)
    dxg = state.x - env_state.reactor_dest_x
    dyg = state.y - env_state.reactor_dest_y
    dist2_goal = dxg*dxg + dyg*dyg
    reach_goal = is_reactor & env_state.reactor_dest_active & (dist2_goal <= env_state.reactor_dest_radius**2)

    # If reached: Level win (return to map) + bonus score
    goal_reward = jnp.float32(100.0) 
    score_delta = score_delta + jnp.where(reach_goal, goal_reward, jnp.float32(0.0))

    # Merge resets (all enemies cleared OR reached goal -> return to map)
    reset = reset | reach_goal

    # Set active to False to prevent re-triggering
    reactor_dest_active_next = jnp.where(reach_goal, jnp.array(False), env_state.reactor_dest_active)

    # The game is only truly "done" when lives run out; a regular crash doesn't end the level
    done = game_over

    # Level ID doesn't change here; maintain the function signature
    level = jnp.int32(-1)

    # obs
    obs = jnp.array([state.x, state.y, state.vx, state.vy, state.angle])

    return (
        obs, state, bullets, cooldown, enemies, enemy_bullets, fire_cooldown,
        key, reward, score_delta,
        crash_timer_next, lives_next, 
        reactor_dest_active_next,
        done, reset, level
    )

# ========== Arena Step Core ==========
@jax.jit
def step_arena(env_state: EnvState, action: int):
    # --- 1. Setup ---
    ship = env_state.state
    saucer = env_state.saucer
    is_crashing = env_state.crash_timer > 0

    # --- 2. Ship Movement and Player Firing ---
    # If the ship is crashing, ignore player input and force no movement
    actual_action = jnp.where(is_crashing, NOOP, action)
    ship_after_move = ship_step(ship, actual_action, (WINDOW_WIDTH, WINDOW_HEIGHT), HUD_HEIGHT)
    
    can_fire = jnp.isin(action, jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])) & (env_state.cooldown == 0) & (_bullets_alive_count(env_state.bullets) < 2)
    bullets = jax.lax.cond(can_fire, lambda b: fire_bullet(b, ship_after_move.x, ship_after_move.y, ship_after_move.angle, 2.0), lambda b: b, env_state.bullets)
    bullets = update_bullets(bullets)
    cooldown = jnp.where(can_fire, 5, jnp.maximum(env_state.cooldown - 1, 0))

    # --- 3. Saucer Movement and Firing ---
    saucer_after_move = jax.lax.cond(saucer.alive, lambda s: _update_saucer_seek(s, ship_after_move.x, ship_after_move.y, SAUCER_SPEED_ARENA), lambda s: s, operand=saucer)
    can_shoot_saucer = saucer_after_move.alive & (_bullets_alive_count(env_state.enemy_bullets) < 1) & ((env_state.mode_timer % SAUCER_FIRE_INTERVAL_FRAMES) == 0)
    enemy_bullets = jax.lax.cond(can_shoot_saucer, lambda eb: _fire_single_from_to(eb, saucer_after_move.x, saucer_after_move.y, ship_after_move.x, ship_after_move.y, SAUCER_BULLET_SPEED), lambda eb: eb, operand=env_state.enemy_bullets)
    enemy_bullets = update_bullets(enemy_bullets)

    # --- 4. Collision Detection ---
    # a) Player bullet hits Saucer
    bullets, hit_saucer_by_bullet = _bullets_hit_saucer(bullets, saucer_after_move)
    
    # b) Ship and Saucer collide directly
    hit_saucer_by_contact = _circle_hit(ship_after_move.x, ship_after_move.y, SHIP_RADIUS, saucer_after_move.x, saucer_after_move.y, SAUCER_RADIUS) & saucer_after_move.alive

    # c) Saucer bullet hits ship
    enemy_bullets, hit_ship_by_bullet = consume_ship_hits(ship_after_move, enemy_bullets, SHIP_RADIUS)

    # --- 5. State Finalization ---
    # a) Is the Saucer dead?
    # Death conditions: Hit by bullet OR collided with ship
    saucer_is_hit = hit_saucer_by_bullet | hit_saucer_by_contact
    hp_after_hit = saucer_after_move.hp - jnp.where(saucer_is_hit, 1, 0) # Simplified: 1 HP is lost per hit
    was_alive = saucer_after_move.alive
    is_dead_now = hp_after_hit <= 0
    just_died = was_alive & is_dead_now
    saucer_final = saucer_after_move._replace(
        hp=hp_after_hit,
        alive=was_alive & (~is_dead_now),
        death_timer=jnp.where(just_died, SAUCER_EXPLOSION_FRAMES, jnp.maximum(saucer_after_move.death_timer - 1, 0))
    )

    # b) Is the ship dead?
    # Death conditions: Hit by Saucer bullet OR collided with Saucer
    ship_is_hit = hit_ship_by_bullet | hit_saucer_by_contact
    
    # c) Update the ship's crash timer
    start_crash = ship_is_hit & (~is_crashing) # A crash is only initiated if hit while not already crashing
    crash_timer_next = jnp.where(start_crash, 30, jnp.maximum(env_state.crash_timer - 1, 0))

    # d) Determine if a reset signal should be sent
    # Signal condition: The ship's crash animation has just finished playing (timer goes from 1 to 0)
    reset_signal = (env_state.crash_timer == 1)
    # If the ship didn't die but the saucer's explosion animation is finished, also exit the Arena
    back_to_map_signal = (~ship_is_hit) & (~saucer_final.alive) & (saucer_final.death_timer == 0)
    
    # --- 6. Assemble and Return ---
    obs = jnp.array([ship_after_move.x, ship_after_move.y, ship_after_move.vx, ship_after_move.vy, ship_after_move.angle])
    reward = jnp.where(just_died, 300.0, 0.0) 
    
    info = {
        "crash": start_crash, 
        "hit_by_bullet": hit_ship_by_bullet,
        "reactor_crash_exit": jnp.array(False), 
    }

    final_env_state = env_state._replace(
        state=ship_after_move,
        bullets=bullets,
        cooldown=cooldown,
        saucer=saucer_final,
        enemy_bullets=enemy_bullets,
        crash_timer=crash_timer_next,
        mode_timer=env_state.mode_timer + 1,
        score=env_state.score + reward,
    )
    
    # If it's a "win" exit, return directly to the map
    def _go_to_map_win(env):
        return env._replace(mode=jnp.int32(0), saucer=make_default_saucer())
    final_env_state = jax.lax.cond(back_to_map_signal, _go_to_map_win, lambda e: e, final_env_state)

    # The final `reset` signal is either "crash finished" or "win exit"
    return obs, final_env_state, reward, jnp.array(False), info, reset_signal | back_to_map_signal, jnp.int32(-1)


@jax.jit
def _bullets_hit_terrain(env_state: EnvState, bullets: Bullets) -> Bullets:

    H, W = env_state.terrain_bank.shape[1], env_state.terrain_bank.shape[2]
    
    bank_idx = jnp.clip(env_state.terrain_bank_idx, 0, env_state.terrain_bank.shape[0] - 1)
    terrain_map = env_state.terrain_bank[bank_idx]

    xi = jnp.clip(jnp.round(bullets.x).astype(jnp.int32), 0, W - 1)
    yi = jnp.clip(jnp.round(bullets.y).astype(jnp.int32), 0, H - 1)

    pixel_colors = terrain_map[yi, xi]

    hit_terrain_mask = jnp.sum(pixel_colors, axis=-1) > 0

    final_hit_mask = bullets.alive & hit_terrain_mask

    return bullets._replace(alive=bullets.alive & ~final_hit_mask)

def _update_ufo(env: EnvState, ship: ShipState, bullets: Bullets) -> EnvState:
    
    # --- Nested Helper Function Definitions ---
    def ground_safe_y_at(e, xf):
        W, H = WINDOW_WIDTH, WINDOW_HEIGHT
        bank_idx = jnp.clip(e.terrain_bank_idx, 0, e.terrain_bank.shape[0] - 1)
        terrain_page = e.terrain_bank[bank_idx]
        col_x = jnp.clip(xf.astype(jnp.int32), 0, W - 1)
        is_ground_in_col = jnp.sum(terrain_page[:, col_x], axis=-1) > 0
        y_indices = jnp.arange(H, dtype=jnp.int32)
        ground_indices = jnp.where(is_ground_in_col, y_indices, H)
        ground_y = jnp.min(ground_indices)
        return jnp.float32(ground_y) - 20.0 # CLEARANCE

    # --- Logic for when the UFO is alive ---
    def _alive_step(e, ship, bullets):
        u = e.ufo
        MIN_ALT = jnp.float32(HUD_HEIGHT + 48.0)
        
        # 1. Movement
        nx = jnp.clip(u.x + u.vx, 8.0, WINDOW_WIDTH - 8.0)
        safe_next = ground_safe_y_at(e, nx)
        corridor_blocked = safe_next <= MIN_ALT
        vx_after_move = jnp.where(corridor_blocked, -u.vx, u.vx)
        x_after_move  = jnp.where(corridor_blocked, u.x, nx)
        safe_here = ground_safe_y_at(e, x_after_move)
        y_after_move = jnp.maximum(jnp.minimum(u.y, safe_here), MIN_ALT)
        u_after_move = u._replace(x=x_after_move, y=y_after_move, vx=vx_after_move)
        
        # 2. Collision Detection
        hit_by_ship = _circle_hit(ship.x, ship.y, SHIP_RADIUS, u_after_move.x, u_after_move.y, UFO_HIT_RADIUS) & u_after_move.alive
        bullets_after_hit, hit_by_bullet = _bullets_hit_ufo(bullets, u_after_move)

        # 3. State Update
        hp_after_hit = u_after_move.hp - jnp.where(hit_by_bullet, 1, 0)
        was_alive = u_after_move.alive
        is_dead_now = hit_by_ship | (hp_after_hit <= 0)
        just_died = was_alive & is_dead_now
        u_final = u_after_move._replace(
            hp=hp_after_hit,
            alive=was_alive & (~is_dead_now),
            death_timer=jnp.where(just_died, SAUCER_EXPLOSION_FRAMES, u_after_move.death_timer)
        )

        # 4. Firing Logic
        FIRE_COOLDOWN = jnp.int32(45)
        no_ufo_bullet_alive = ~jnp.any(e.ufo_bullets.alive)
        cd_ok = (e.mode_timer % FIRE_COOLDOWN) == 0
        can_shoot = u_final.alive & no_ufo_bullet_alive & cd_ok
        def _fire_one(bul):
            return _fire_single_from_to(bul, u_final.x, u_final.y, ship.x, ship.y, 0.4)
        ufo_bullets = jax.lax.cond(can_shoot, _fire_one, lambda b: b, e.ufo_bullets)
        
        # 5. Return the complete environment state
        return e._replace(ufo=u_final, bullets=bullets_after_hit, ufo_bullets=ufo_bullets)

    # --- Logic for when the UFO is dead ---
    def _dead_step(e, ship, bullets):
        u = e.ufo
        u2 = u._replace(death_timer=jnp.maximum(u.death_timer - 1, 0))
        # Return the COMPLETE environment state
        return e._replace(ufo=u2, ufo_bullets=create_empty_bullets_16(), bullets=bullets)
    
    # Call the appropriate branch based on whether the UFO is alive
    return jax.lax.cond(env.ufo.alive, _alive_step, _dead_step, env, ship, bullets)

# ========== Step Core ==========
@jax.jit
def step_core(env_state: EnvState, action: int):
    # `jax.lax.switch` selects a function from the list based on the value of `mode` (0, 1, or 2)
    # It automatically passes the `(env_state, action)` operands to the chosen function.
    return jax.lax.switch(
        jnp.clip(env_state.mode, 0, 2),
        [step_map, _step_level_core, step_arena],
        env_state,
        action
    )


def get_action_from_key():
    keys = pygame.key.get_pressed()

    thrust = keys[pygame.K_UP]
    rotate_left = keys[pygame.K_LEFT]
    rotate_right = keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]
    down = keys[pygame.K_DOWN]

    # First, check the most complex combinations (3 keys)
    if thrust and rotate_right and fire:
        return 14  # UPRIGHTFIRE
    elif thrust and rotate_left and fire:
        return 15  # UPLEFTFIRE
    elif down and rotate_right and fire:
        return 16  # DOWNRIGHTFIRE
    elif down and rotate_left and fire:
        return 17  # DOWNLEFTFIRE

    # Then, check for combinations of 2 keys
    elif thrust and fire:
        return 10  # UPFIRE
    elif rotate_right and fire:
        return 11  # RIGHTFIRE
    elif rotate_left and fire:
        return 12  # LEFTFIRE
    elif down and fire:
        return 13  # DOWNFIRE
    elif thrust and rotate_right:
        return 6   # UPRIGHT
    elif thrust and rotate_left:
        return 7   # UPLEFT
    elif down and rotate_right:
        return 8   # DOWNRIGHT
    elif down and rotate_left:
        return 9   # DOWNLEFT

    # Finally, check for single keys
    elif fire:
        return 1   # FIRE
    elif thrust:
        return 2   # UP
    elif rotate_right:
        return 3   # RIGHT
    elif rotate_left:
        return 4   # LEFT
    elif down:
        return 5   # DOWN

    # If no key is pressed, return NOOP (No Operation)
    return 0


class JaxGravitar(JaxEnvironment):
    def __init__(self, render_backend: str = "pygame"):
        super().__init__()
        self.obs_shape = (5,)
        self.num_actions = 18
        self.render_backend = render_backend

        self.base_w, self.base_h = 160, 192
        self.window_w, self.window_h = 720, 960

        self.world = pygame.Surface((self.base_w, self.base_h), pygame.SRCALPHA)
        self.num_actions = getattr(self, "num_actions", 18)
        # ---- Game State ----
        self.score = 0
        self.lives = 6
        self.current_level = None
        self.game_over = False
        self.done = False
        self.key = jrandom.PRNGKey(0)
        self.mode = "map"

        # ---- Bullet Containers ----
        self.bullets = Bullets(
            x=jnp.zeros((0,)), y=jnp.zeros((0,)),
            vx=jnp.zeros((0,)), vy=jnp.zeros((0,)),
            alive=jnp.zeros((0,), dtype=bool)
        )
        self.bullets_speed = 0.2

        self.enemy_bullets = Bullets(
            x=jnp.zeros((0,)), y=jnp.zeros((0,)),
            vx=jnp.zeros((0,)), vy=jnp.zeros((0,)),
            alive=jnp.zeros((0,), dtype=bool)
        )
        self.enemy_fire_cooldown = 0
        self.enemy_fire_interval = 60
        self.enemy_bullet_speed = 0.2

        self.cooldown = 0
        self.cooldown_max = 5

        # ---- Rendering Backend ----
        if self.render_backend == "pygame":
            pygame.init()
            pygame.font.init()
            self.screen_size = (WINDOW_WIDTH, WINDOW_HEIGHT)
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Gravitar")
            self.sprites = load_sprites_tuple()

        # ---- Unify "Map Layout" and "Collision Radii" (consistent with rendering) ----
        MAP_SCALE = 3           # Must match `MAP_SCALE` in `render()`
        HITBOX_SCALE = 0.90     # Shrink hitbox slightly to avoid "phantom collisions"

        layout = [
            (SpriteIdx.PLANET1,   0.82, 0.18),
            (SpriteIdx.PLANET2,   0.22, 0.24),
            (SpriteIdx.REACTOR,   0.18, 0.43),
            (SpriteIdx.SPAWN_LOC, 0.50, 0.56),
            (SpriteIdx.OBSTACLE,  0.57, 0.38),
            (SpriteIdx.PLANET3,   0.76, 0.76),
            (SpriteIdx.PLANET4,   0.14, 0.88),
        ]
        px, py, pr, pi = [], [], [], []

        for idx, xp, yp in layout:
            cx = int(xp * WINDOW_WIDTH)
            cy = int(yp * WINDOW_HEIGHT)
            spr = self.sprites[idx]

            if spr is not None:
                r = 0.1 * max(spr.get_width(), spr.get_height()) * MAP_SCALE * HITBOX_SCALE

            else:
                r = 4

            px.append(cx); py.append(cy); pr.append(r); pi.append(int(idx))

        self.planets = (
            np.array(px, dtype=np.float32),
            np.array(py, dtype=np.float32),
            np.array(pr, dtype=np.float32),
            np.array(pi, dtype=np.int32),
        )

        self.num_planets = len(px)
        self.terrain_bank = self._build_terrain_bank()

        dummy_key = jax.random.PRNGKey(0)

        _, dummy_state = self.reset(dummy_key) 

        tmp_obs, tmp_state = self.reset_level(dummy_key, 0, dummy_state)

        self.reset_level_out_struct = (
            jax.ShapeDtypeStruct(tmp_obs.shape, tmp_obs.dtype),
            jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tmp_state)
        )

    def action_space(self):
        # The `spaces.Discrete` space from the tutorial supports `sample(key)`
        # Most implementations only require `n`; if requires `dtype`, add `dtype=jnp.int32`
        return spaces.Discrete(self.num_actions)
    
    def draw_hud(self):
        scr = self.screen
        W = scr.get_width()

        # Style parameters: uniform scaling & right shift
        SCALE = 3.5          # Scaling factor
        RIGHT_OFFSET = 80    # Pixels to shift right
        TOP_MARGIN = 4
        GAP = 10             # Vertical gap between score and HP
        HP_GAP = 3           # Horizontal gap between lives
        DIGIT_SPACING = int(2 * SCALE)

        # Background bar
        pygame.draw.rect(scr, (0, 0, 0), (0, 0, W, HUD_HEIGHT))

        # ===== Score (Rendered with digit sprites) =====
        score_text = f"{int(self.score):06d}"  # Fixed 6 digits
        digit_surfs = []
        total_w = 0
        max_h = 0

        for ch in score_text:
            idx = getattr(SpriteIdx, f"DIGIT_{ch}")
            spr = self.sprites[idx] if self.sprites else None
            if spr is None:
                spr = pygame.Surface((8, 12), pygame.SRCALPHA)
            if SCALE != 1.0:
                w, h = spr.get_width(), spr.get_height()
                spr = pygame.transform.scale(spr, (int(w * SCALE), int(h * SCALE)))
            digit_surfs.append(spr)
            total_w += spr.get_width()
            max_h = max(max_h, spr.get_height())

        total_w += max(0, len(digit_surfs) - 1) * DIGIT_SPACING

        x0 = (W - total_w) // 2 + RIGHT_OFFSET
        y0 = TOP_MARGIN

        x = x0
        for i, spr in enumerate(digit_surfs):
            scr.blit(spr, (x, y0))
            x += spr.get_width()
            if i < len(digit_surfs) - 1:
                x += DIGIT_SPACING

        y_cursor = y0 + max_h + GAP

        # ===== Lives: Use the `HP_UI` sprite, repeated `lives` times; same scaling & right shift =====
        hp_spr = self.sprites[SpriteIdx.HP_UI] if self.sprites else None
        if hp_spr is not None and self.lives > 0:
            hw, hh = hp_spr.get_width(), hp_spr.get_height()
            hp_scaled = pygame.transform.scale(hp_spr, (hw * SCALE, hh * SCALE))

            lives = int(self.lives)
            seg_w = hp_scaled.get_width()
            total_w = lives * seg_w + (lives - 1) * HP_GAP
            x0 = (W - total_w) // 2 + RIGHT_OFFSET 
            for i in range(lives):
                scr.blit(hp_scaled, (x0 + i * (seg_w + HP_GAP), y_cursor))

    def get_action_space(self) -> Tuple[int]:
        """
        Returns the action space of the environment.
        Returns: The action space of the environment as a tuple.
        """
        # raise NotImplementedError("Abstract method")
        return (18,)  # Supports actions numbered 0 to 17   

    def get_observation_space(self) -> Tuple[int]:
        """
        Returns the observation space of the environment.
        Returns: The observation space of the environment as a tuple.
        """
        # raise NotImplementedError("Abstract method")
        return (5,)

    def render(self, env_state: EnvState) -> Tuple[jnp.ndarray]:
        # --------- pygame  ---------
        if self.render_backend == "pygame":
            scr = self.screen
            mode_val = int(env_state.mode)
            mode_str = "map" if mode_val == 0 else ("level" if mode_val == 1 else "arena")

             # 1) Fallback for base canvas (base resolution)
            if not hasattr(self, "base_w"):
                self.base_w, self.base_h = 160, 192
            if not hasattr(self, "world"):
                self.world = pygame.Surface((self.base_w, self.base_h), pygame.SRCALPHA)

            W, H = self.base_w, self.base_h

            # Layer 0: Background
            scr.fill((0, 0, 0))

            # Layer 1: Map planets
            if mode_str == "map":
                W, H = WINDOW_WIDTH, WINDOW_HEIGHT
                M = 12              # Margin to prevent items from being too close to the edge
                MAP_SCALE = 3

                def blit_center_scaled(spr, center, scale=1.0):
                    if spr is None:
                        return
                    if scale != 1.0:
                        w, h = spr.get_width(), spr.get_height()
                        spr_use = pygame.transform.scale(spr, (int(w * scale), int(h * scale)))
                    else:
                        spr_use = spr
                    rect = spr_use.get_rect(center=center)
                    
                    # Safe margin
                    if rect.left < M: rect.left = M
                    if rect.top < M: rect.top = M
                    if rect.right > W - M: rect.right = W - M
                    if rect.bottom > H - M: rect.bottom = H - M
                    scr.blit(spr_use, rect.topleft)

                px, py, pr, pi = self.planets
                for k in range(len(px)):
                    idx = SpriteIdx(int(pi[k]))
                    if idx == SpriteIdx.REACTOR and bool(env_state.reactor_destroyed):
                        continue
                    if bool(env_state.planets_cleared_mask[k]):
                        continue
                    spr = self.sprites[idx]
                    blit_center_scaled(spr, (int(px[k]), int(py[k])), MAP_SCALE)

            # Layer 1.5: Saucer (visible in map/arena mode)
            if mode_str in ("map", "arena"):
                sau = env_state.saucer
                if bool(sau.alive):
                    sau_spr = self.sprites[SpriteIdx.ENEMY_SAUCER]
                    if sau_spr is not None:
                        if SAUCER_SCALE != 1.0:
                            w, h = sau_spr.get_width(), sau_spr.get_height()
                            sau_spr_use = pygame.transform.scale(
                                sau_spr, (int(w * SAUCER_SCALE), int(h * SAUCER_SCALE))
                            )
                        else:
                            sau_spr_use = sau_spr
                        rect = sau_spr_use.get_rect(center=(int(sau.x), int(sau.y)))
                        scr.blit(sau_spr_use, rect.topleft)
                    else:
                        pygame.draw.circle(scr, (200, 200, 255), (int(sau.x), int(sau.y)), int(SAUCER_RADIUS))
                elif int(sau.death_timer) > 0:
                    boom = self.sprites[SpriteIdx.SAUCER_CRASH] if self.sprites else None
                    t = int(sau.death_timer)
                    progress = 1.0 - (t / max(1, int(SAUCER_EXPLOSION_FRAMES)))
                    scale = 1.4 + 0.8 * progress
                    if boom is not None:
                        w, h = boom.get_width(), boom.get_height()
                        boom_use = pygame.transform.scale(boom, (int(w * scale), int(h * scale)))
                        rect = boom_use.get_rect(center=(int(sau.x), int(sau.y)))
                        scr.blit(boom_use, rect.topleft)
                    else:
                        r = int(SAUCER_RADIUS) + int(10 * (1.0 + progress))
                        pygame.draw.circle(scr, (255, 200, 120), (int(sau.x), int(sau.y)), r)

            # Layer 2: Level (Terrain -> UFO -> Enemies/Turrets)
            if mode_str == "level":
                # ----- Terrain -----
                terr_idx = int(env_state.terrain_sprite_idx)
                terr_spr = self.sprites[SpriteIdx(terr_idx)] if self.sprites else None
                if terr_spr is not None:
                    tw, th = terr_spr.get_width(), terr_spr.get_height()
                    W, H = WINDOW_WIDTH, WINDOW_HEIGHT
                    
                    # 1. Calculate the base scale factor for the terrain to fit the screen
                    scale = min(W / tw, H / th)
                    
                    # 2. Apply the special scale override (`TERRANT_SCALE_OVERRIDES`)
                    extra = TERRANT_SCALE_OVERRIDES.get(SpriteIdx(terr_idx), 1.0)
                    scale *= float(extra)
                    
                    # 3. Create the final scaled image and its dimensions
                    sw, sh = int(tw * scale), int(th * scale)
                    terr_scaled = pygame.transform.scale(terr_spr, (sw, sh))
                    
                    # 4. Get the manual offset for the current level (`LEVEL_OFFSETS`)
                    current_level_id = int(env_state.current_level)
                    level_offset = LEVEL_OFFSETS.get(current_level_id, (0, 0))
                    
                    # 5. First, calculate the coordinates to center the scaled terrain
                    ox = (W - sw) // 2
                    oy = (H - sh) // 2
                    
                    # 6. Then apply the manually set offset
                    ox += level_offset[0]
                    oy += level_offset[1]
                    
                    # 7. Draw using the final calculated coordinates
                    scr.blit(terr_scaled, (ox, oy))

                # ----- UFO (alive or exploding) -----
                if hasattr(env_state, "ufo"):
                    u = env_state.ufo

                    if int(u.death_timer) > 0:
                        # Explosion rendering (can be changed to a `UFO_CRASH` sprite if available)
                        boom = self.sprites[SpriteIdx.SAUCER_CRASH] if self.sprites else None
                        t = int(u.death_timer)
                        progress = 1.0 - (t / max(1, int(SAUCER_EXPLOSION_FRAMES)))  # 0..1
                        boom_scale = 1.4 + 0.8 * progress
                        if boom is not None:
                            w, h = boom.get_width(), boom.get_height()
                            boom_use = pygame.transform.scale(boom, (int(w * boom_scale), int(h * boom_scale)))
                            rect = boom_use.get_rect(center=(int(u.x), int(u.y)))
                            scr.blit(boom_use, rect.topleft)
                        else:
                            r = int(SAUCER_RADIUS * UFO_SCALE) + int(10 * (1.0 + progress))
                            pygame.draw.circle(scr, (255, 200, 120), (int(u.x), int(u.y)), r)

                    elif bool(u.alive):
                        ufo_spr = self.sprites[SpriteIdx.ENEMY_UFO] if self.sprites else None
                        if ufo_spr is not None:
                            w, h = ufo_spr.get_width(), ufo_spr.get_height()
                            ufo_use = pygame.transform.scale(ufo_spr, (int(w * UFO_SCALE), int(h * UFO_SCALE)))
                            rect = ufo_use.get_rect(center=(int(u.x), int(u.y)))
                            scr.blit(ufo_use, rect.topleft)
                        else:
                            pygame.draw.circle(scr, (180, 255, 180), (int(u.x), int(u.y)), int(SAUCER_RADIUS * UFO_SCALE))

                # ----- Enemies/Turrets (aligned to top-left) -----
                ens = env_state.enemies
                for i in range(len(ens.x)):
                    # Safety check, skip if enemy is invalid
                    if float(ens.w[i]) == 0.0 or int(ens.sprite_idx[i]) < 0:
                        continue
                    
                    # 1. Get the current enemy's type and original sprite
                    current_enemy_sprite_idx = int(ens.sprite_idx[i])
                    enemy_spr_original = self.sprites[SpriteIdx(current_enemy_sprite_idx)]
                    
                    if enemy_spr_original is None:
                        continue

                    # 2. Determine the scaling factor based on enemy type
                    scale = 1.0
                    if current_enemy_sprite_idx == int(SpriteIdx.ENEMY_ORANGE):
                        scale = ENEMY_ORANGE_SCALE
                    elif current_enemy_sprite_idx == int(SpriteIdx.ENEMY_GREEN):
                        scale = ENEMY_GREEN_SCALE
                    elif current_enemy_sprite_idx == int(SpriteIdx.FUEL_TANK):
                        scale = FUEL_TANK_SCALE
                    
                    # 3. Apply the scale to get the "scaled sprite"
                    if scale != 1.0:
                        w, h = enemy_spr_original.get_width(), enemy_spr_original.get_height()
                        scaled_spr = pygame.transform.scale(enemy_spr_original, (int(w * scale), int(h * scale)))
                    else:
                        scaled_spr = enemy_spr_original

                    # 4. Flip the scaled sprite based on the `flip_y` data
                    if bool(ens.flip_y[i]):
                        scaled_spr = pygame.transform.flip(scaled_spr, False, True) # True=horizontal, False=vertical
                    
                    # 5. Adjust the anchor point and calculate the final drawing coordinates
                    scaled_w, scaled_h = scaled_spr.get_width(), scaled_spr.get_height()
                    ex, ey = int(ens.x[i]), int(ens.y[i])
                    
                    draw_x = ex - scaled_w // 2
                    draw_y = ey - scaled_h

                    # Check if the current enemy is exploding
                    if int(ens.death_timer[i]) > 0:
                        # If exploding, draw the explosion sprite
                        crash_spr = self.sprites[SpriteIdx.ENEMY_CRASH]
                        if crash_spr is not None:
                            # To make the explosion more dynamic, scale it up during the animation
                            progress = 1.0 - (int(ens.death_timer[i]) / ENEMY_EXPLOSION_FRAMES)
                            crash_scale = 1.0 + progress * 0.5
                            w, h = crash_spr.get_width(), crash_spr.get_height()
                            scaled_crash_spr = pygame.transform.scale(crash_spr, (int(w * crash_scale), int(h * crash_scale)))

                            # Use the corrected coordinates to draw the explosion
                            crash_w, crash_h = scaled_crash_spr.get_width(), scaled_crash_spr.get_height()
                            crash_draw_x = int(ens.x[i]) - crash_w // 2
                            crash_draw_y = int(ens.y[i]) - crash_h // 2             # Explosion is centered
                            scr.blit(scaled_crash_spr, (crash_draw_x, crash_draw_y))
                    else:
                        # If not exploding, draw the normal enemy sprite
                        scr.blit(scaled_spr, (draw_x, draw_y))

            # Layer 3: Bullets
            pb = self.sprites[SpriteIdx.SHIP_BULLET] if self.sprites else None
            eb = self.sprites[SpriteIdx.ENEMY_BULLET] if self.sprites else None
            BULLET_SCALE = 3

            def blit_bullet_center(surface, spr, x, y):
                if spr is None:
                    return
                if BULLET_SCALE != 1.0:
                    w, h = spr.get_width(), spr.get_height()
                    spr_use = pygame.transform.scale(spr, (int(w * BULLET_SCALE), int(h * BULLET_SCALE)))
                else:
                    spr_use = spr
                rect = spr_use.get_rect(center=(int(x), int(y)))
                surface.blit(spr_use, rect.topleft)

            # Enemy bullets
            ebul = env_state.enemy_bullets
            for i in range(len(ebul.x)):
                if not bool(ebul.alive[i]): 
                    continue
                blit_bullet_center(scr, eb, float(ebul.x[i]), float(ebul.y[i]))

            # UFO bullets (max one)
            ubul = env_state.ufo_bullets
            for i in range(len(ubul.x)):
                if not bool(ubul.alive[i]): continue
                blit_bullet_center(scr, eb, float(ubul.x[i]), float(ubul.y[i]))

            # Player bullets
            pbul = env_state.bullets
            for i in range(len(pbul.x)):
                if not bool(pbul.alive[i]): 
                    continue
                blit_bullet_center(scr, pb, float(pbul.x[i]), float(pbul.y[i]))

            # Layer 4: Ship (on top of bullets)
            cx, cy = int(env_state.state.x), int(env_state.state.y)
            angle = float(env_state.state.angle)

            ship_idx = SpriteIdx.SHIP_IDLE              # Default to idle state
            keys = pygame.key.get_pressed()

            if int(env_state.crash_timer) > 0:
                ship_idx = SpriteIdx.SHIP_CRASH
            elif keys[pygame.K_UP]:
                ship_idx = SpriteIdx.SHIP_THRUST
            elif keys[pygame.K_DOWN]:
                ship_idx = SpriteIdx.SHIP_THRUST_BACK 

            ship = self.sprites[ship_idx] if self.sprites else None
            SHIP_SCALE = 3.0
            if SHIP_SCALE != 1.0:
                w, h = ship.get_width(), ship.get_height()
                ship_base = pygame.transform.scale(ship, (int(w * SHIP_SCALE), int(h * SHIP_SCALE)))
            else:
                ship_base = ship

            if ship_idx == SpriteIdx.SHIP_CRASH:
                ship_rot = ship_base
            else:
                if not hasattr(self, "_ship_rot_cache"):
                    self._ship_rot_cache = {}
                angle_deg = (-math.degrees(angle) - 90.0) % 360.0
                q = int(round(angle_deg / 5.0)) * 5
                cache_key = (int(ship_idx), q, int(SHIP_SCALE * 100))
                ship_rot = self._ship_rot_cache.get(cache_key)
                if ship_rot is None:
                    ship_rot = pygame.transform.rotate(ship_base, q)
                    self._ship_rot_cache[cache_key] = ship_rot

            rect = ship_rot.get_rect(center=(cx, cy))
            scr.blit(ship_rot, rect.topleft)

            # Layer 5: HUD
            self.score = int(env_state.score)
            self.lives = int(env_state.lives) if hasattr(env_state, "lives") else getattr(self, "lives", 3)
            self.draw_hud()

            # Reactor destination point (only in reactor mode & active)
            # === Only render the pink target at the CENTER of the game area if in reactor terrain and active ===
            if (hasattr(env_state, "terrain_sprite_idx")
                and env_state.terrain_sprite_idx == int(SpriteIdx.REACTOR_TERR)
                and bool(env_state.reactor_dest_active)):

                TARGET_X = 283  
                TARGET_Y = 333 

                dest_surf = self.sprites[SpriteIdx.REACTOR_DEST]
                dest_surf = pygame.transform.scale(dest_surf, (30, 10))
                rect = dest_surf.get_rect(center=(int(TARGET_X), int(TARGET_Y)))
                self.screen.blit(dest_surf, rect)

            # Game Over
            if bool(env_state.done):
                font_big = pygame.font.SysFont("Arial", 72, bold=True)
                text = font_big.render("GAME OVER", True, (255, 0, 0))
                text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
                scr.blit(text, text_rect)

            pygame.display.flip()

            # Pygame branch returns a placeholder (to maintain a consistent interface)
            dummy_image = jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=jnp.uint8)
            return (dummy_image,)

        # --------- JAX Branch ---------
        out = self.renderer.render(env_state.state)
        if out is None:
            frame = jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=jnp.uint8)
        else:
            arr = np.asarray(out)
            if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] != arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))
            frame = jnp.array(arr, dtype=jnp.uint8)
        return (frame,)
    
    def build_terrain_mask_and_transform(self, terr_idx: int):
        surf = self.sprites[SpriteIdx(terr_idx)]
        tw, th = surf.get_width(), surf.get_height()
        W, H = WINDOW_WIDTH, WINDOW_HEIGHT

        alpha = pygame.surfarray.pixels_alpha(surf)
        small_mask_np = (alpha.T > 0).astype('uint8')  # (H, W)
        del alpha

        scale = min(W / tw, H / th)

        # Sync extra scaling
        extra = TERRANT_SCALE_OVERRIDES.get(SpriteIdx(terr_idx), 1.0)
        scale *= float(extra)

        sw, sh = int(tw * scale), int(th * scale)
        ox = (W - sw) // 2
        oy = (H - sh) // 2

        full_mask = jnp.zeros((H, W), dtype=jnp.uint8)

        h_small, w_small = small_mask_np.shape
        full_mask = full_mask.at[oy:oy+h_small, ox:ox+w_small].set(small_mask_np)

        return (full_mask,
                jnp.array(scale, dtype=jnp.float32),
                jnp.array([ox, oy], dtype=jnp.float32))
    
    def _build_terrain_bank(self) -> jnp.ndarray:
        """Builds a bank of binary masks at screen size: 0=empty, 1=T1, 2=T2, 3=T3, 4=T4 (can add reactor=5)."""
        W, H = WINDOW_WIDTH, WINDOW_HEIGHT
        # Index 0: All zeros (map/no terrain)
        bank = [np.zeros((H, W, 3), dtype=np.uint8)]

        def sprite_to_mask(idx: int) -> np.ndarray:
            surf = self.sprites[SpriteIdx(idx)]
            tw, th = surf.get_width(), surf.get_height()
            scale = min(W / tw, H / th)
            extra = TERRANT_SCALE_OVERRIDES.get(SpriteIdx(idx), 1.0)
            scale *= float(extra)
            sw, sh = int(tw * scale), int(th * scale)
            ox, oy = (W - sw) // 2, (H - sh) // 2
            scaled_surf = pygame.transform.scale(surf, (sw, sh))

            rgb_array = pygame.surfarray.pixels3d(scaled_surf)
            rgb_array_hwc = rgb_array.transpose((1, 0, 2))

            color_map = np.zeros((H, W, 3), dtype=np.uint8)
            color_map[oy:oy+sh, ox:ox+sw] = rgb_array_hwc

            return color_map

        # 1..4：TERRANT1..4
        bank.append(sprite_to_mask(int(SpriteIdx.TERRANT1)))
        bank.append(sprite_to_mask(int(SpriteIdx.TERRANT2)))
        bank.append(sprite_to_mask(int(SpriteIdx.TERRANT3)))
        bank.append(sprite_to_mask(int(SpriteIdx.TERRANT4)))
        bank.append(sprite_to_mask(int(SpriteIdx.REACTOR_TERR)))

        return jnp.array(np.stack(bank, axis=0), dtype=jnp.uint8)

    def reset_map(self, key: jnp.ndarray, lives: Optional[int] = None, score: Optional[float] = None) -> Tuple[jnp.ndarray, EnvState]:  
        ship_state = ShipState(
            x=jnp.array(WINDOW_WIDTH / 2, dtype=jnp.float32),
            y=jnp.array((WINDOW_HEIGHT + HUD_HEIGHT) / 2, dtype=jnp.float32),
            vx=jnp.array(jnp.cos(-jnp.pi / 4) * 0.3, dtype=jnp.float32), # Speed of 1.5
            vy=jnp.array(jnp.sin(-jnp.pi / 4) * 0.3, dtype=jnp.float32),
            angle=jnp.array(-jnp.pi / 2, dtype=jnp.float32)
        )
        
        
        px_np, py_np, pr_np, pi_np = self.planets
        ids_np = [SPRITE_TO_LEVEL_ID.get(sprite_idx, -1) for sprite_idx in pi_np]

        env_state = EnvState(
            mode=jnp.int32(0),
            state=ship_state,
            bullets=create_empty_bullets_64(),
            cooldown=jnp.array(0, dtype=jnp.int32),
            enemies=create_empty_enemies(),
            enemy_bullets=create_empty_bullets_16(),
            fire_cooldown=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),
            key=key,
            key_alt=key,
            score=jnp.array(score if score is not None else 0.0, dtype=jnp.float32),
            done=jnp.array(False),
            lives=jnp.array(lives if lives is not None else MAX_LIVES, dtype=jnp.int32),
            crash_timer=jnp.int32(0), 

            planets_px=jnp.array(px_np, dtype=jnp.float32),
            planets_py=jnp.array(py_np, dtype=jnp.float32),
            planets_pr=jnp.array(pr_np, dtype=jnp.float32),
            planets_pi=jnp.array(pi_np, dtype=jnp.int32), 
            planets_id=jnp.array(ids_np, dtype=jnp.int32),
            current_level=jnp.int32(-1),

            terrain_sprite_idx=jnp.int32(-1),
            terrain_mask=jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=jnp.uint8),
            terrain_scale=jnp.array(1.0, dtype=jnp.float32),          
            terrain_offset=jnp.array([0.0, 0.0], dtype=jnp.float32), 
            terrain_bank=self.terrain_bank,
            terrain_bank_idx=jnp.int32(0),
            respawn_shift_x=jnp.float32(0.0),
            reactor_dest_active=jnp.array(False),
            reactor_dest_x=jnp.float32(0.0),
            reactor_dest_y=jnp.float32(0.0),
            reactor_dest_radius=jnp.float32(0.4), 
            mode_timer=jnp.int32(0),
            saucer=make_default_saucer(),
            saucer_spawn_timer=jnp.int32(SAUCER_SPAWN_DELAY_FRAMES),
            map_return_x=jnp.float32(0.0),
            map_return_y=jnp.float32(0.0),
            ufo=make_empty_ufo(),
            ufo_used=jnp.array(False),
            ufo_home_x=jnp.float32(0.0),
            ufo_home_y=jnp.float32(0.0),
            ufo_bullets=create_empty_bullets_16(),
            level_offset=jnp.array([0, 0], dtype=jnp.float32),
            reactor_destroyed=jnp.array(False),
            planets_cleared_mask=jnp.zeros_like(self.planets[0], dtype=bool),
        )

        obs = jnp.array([
            ship_state.x,
            ship_state.y,
            ship_state.vx,
            ship_state.vy,
            ship_state.angle
        ])

        return obs, env_state

    def reset(self, key: jnp.ndarray = jax.random.PRNGKey(0)) -> Tuple[jnp.ndarray, EnvState]:
        obs, env_state = self.reset_map(key)
        return obs, env_state
    
    def reset_level(self, key: jnp.ndarray, level_id: int, prev_env_state: EnvState):
        level_id_int = int(level_id)
        
        # === 1. Load level layout and offset data ===
        layout_list = LEVEL_LAYOUTS.get(level_id_int)
        
        level_offset = LEVEL_OFFSETS.get(level_id_int, (0, 0))

        # === 2. Create all objects within the level (Enemies) ===
        x_coords, y_coords, sprite_indices, flip_flags, hps = [], [], [], [], []
        enemy_w, enemy_h, enemy_vx = 14.0, 14.0, 0.0

        if layout_list:      # Only process if the layout list is not empty
            for item in layout_list:
                pos_ratio = item['pos_ratio']
                
                x_coords.append(pos_ratio[0] * WINDOW_WIDTH)
                y_coords.append(pos_ratio[1] * WINDOW_HEIGHT)
                sprite_indices.append(int(item['type']))
                flip_flags.append(item.get('flip_y', False)) 
                hps.append(item.get('hp', 1))

        num_enemies = len(x_coords)
        def pad(arr, fill_val=0.0):
            return jnp.pad(arr, (0, MAX_ENEMIES - arr.shape[0]), constant_values=fill_val)

        enemies = Enemies(
            x=pad(jnp.array(x_coords, dtype=jnp.float32)),
            y=pad(jnp.array(y_coords, dtype=jnp.float32)),
            w=pad(jnp.full((num_enemies,), enemy_w)),
            h=pad(jnp.full((num_enemies,), enemy_h)),
            vx=pad(jnp.full((num_enemies,), enemy_vx)),
            sprite_idx=pad(jnp.array(sprite_indices, dtype=jnp.int32), -1),
            flip_y=pad(jnp.array(flip_flags, dtype=bool), False),            # Ensure the name matches your `Enemies` definition
            death_timer=pad(jnp.zeros((num_enemies,), dtype=jnp.int32)),
            hp=pad(jnp.array(hps, dtype=jnp.int32))
        )
        fire_cooldown = jnp.full((MAX_ENEMIES,), 999, dtype=jnp.int32).at[:num_enemies].set(0)

        # === 3. Get level terrain data ===
        terrain_sprite_to_use = LEVEL_ID_TO_TERRAIN_SPRITE.get(level_id_int)

        terrain_sprite_idx = jnp.int32(int(terrain_sprite_to_use))
        mask, scale, offset = self.build_terrain_mask_and_transform(int(terrain_sprite_idx))
        bank_idx = LEVEL_ID_TO_BANK_IDX.get(level_id_int, 0)         # If not found, default to 0 (no terrain)

        # === 4. Assemble the final EnvState ===
        ship_state = make_level_start_state(level_id_int)
        px_np, py_np, pr_np, pi_np = self.planets
        
        env_state = EnvState(
            mode=jnp.int32(1),
            state=ship_state,
            bullets=create_empty_bullets_64(),
            cooldown=jnp.array(0, dtype=jnp.int32),
            enemies=enemies,
            enemy_bullets=create_empty_bullets_16(),
            fire_cooldown=fire_cooldown,
            key=key,
            key_alt=key,
            score=prev_env_state.score,
            done=jnp.array(False),
            lives=prev_env_state.lives,
            crash_timer=jnp.int32(0),

            planets_px=jnp.array(px_np, dtype=jnp.float32),
            planets_py=jnp.array(py_np, dtype=jnp.float32),
            planets_pr=jnp.array(pr_np, dtype=jnp.float32),
            planets_id=jnp.array([SPRITE_TO_LEVEL_ID.get(s, -1) for s in pi_np], dtype=jnp.int32),
            planets_pi=jnp.array(pi_np, dtype=jnp.int32),
            current_level=jnp.int32(level_id_int),

            terrain_sprite_idx=terrain_sprite_idx,
            terrain_mask=mask,
            terrain_scale=scale,
            terrain_offset=offset,
            terrain_bank=self.terrain_bank,
            terrain_bank_idx=jnp.int32(bank_idx),

            respawn_shift_x=jnp.float32(0.0),
            reactor_dest_active=jnp.array(level_id_int == 4),
            reactor_dest_x=jnp.float32(0.0),
            reactor_dest_y=jnp.float32(0.0),
            reactor_dest_radius=jnp.float32(0.4),
            mode_timer=jnp.int32(0),

            saucer=make_default_saucer(),
            saucer_spawn_timer=jnp.int32(SAUCER_SPAWN_DELAY_FRAMES),
            map_return_x=jnp.float32(0.0),
            map_return_y=jnp.float32(0.0),
            ufo=make_empty_ufo(),
            ufo_used=jnp.array(False),
            ufo_home_x=jnp.float32(0.0),
            ufo_home_y=jnp.float32(0.0),
            ufo_bullets=create_empty_bullets_16(),
            level_offset=jnp.array(level_offset, dtype=jnp.float32),
            reactor_destroyed=prev_env_state.reactor_destroyed,
            planets_cleared_mask=prev_env_state.planets_cleared_mask,
        )

        obs = jnp.array([ship_state.x, ship_state.y, ship_state.vx, ship_state.vy, ship_state.angle])
        return obs, env_state
    
    
   # Gets terrain for a level (TERRANT*)
    def _make_level_terrain(self, planet_idx_for_level):
        # This function converts the planet's sprite index into the corresponding terrain data
        terr_idx = map_planet_to_terrant(planet_idx_for_level)
        mask, scale, offset = self.build_terrain_mask_and_transform(int(terr_idx))
        return terr_idx, mask, scale, offset

    def step_full(self, env_state: EnvState, action: int):
    
        def _handle_reset(operands):

            obs, current_state, reward, done, info, reset, level = operands

            def _enter_level(inner_operands):
                _, state_to_reset, inner_reward, _, inner_info, _, inner_level = inner_operands
                
                def _reset_level_py(key, level_val, state_val):
                    return self.reset_level(key, level_val, state_val)

                result_shape_and_dtype = self.reset_level_out_struct
                new_main_key, subkey_for_reset = jax.random.split(state_to_reset.key)
                
                obs_reset, next_state = jax.pure_callback(
                    _reset_level_py, result_shape_and_dtype,
                    subkey_for_reset, inner_level, state_to_reset
                )

                next_state = next_state._replace(key=new_main_key)
                enter_info = {**inner_info, "level_cleared": jnp.array(False)}
                return obs_reset, next_state, inner_reward, jnp.array(False), enter_info, jnp.array(True), inner_level

            def _return_to_map(inner_operands):
                obs_map, state_to_reset, reward_map, _, info_map, _, level_map = inner_operands
                is_a_death_event = (level_map == -2) | info_map.get("crash", False) | info_map.get("hit_by_bullet", False) | info_map.get("reactor_crash_exit", False)

                def _on_death(_):
                    lives_next = state_to_reset.lives - 1
                    death_info = {**info_map, "level_cleared": jnp.array(False)}
                    def _game_over(_):
                        game_over_state = state_to_reset._replace(lives=jnp.int32(0), done=jnp.array(True))
                        return obs_map, game_over_state, reward_map, jnp.array(True), death_info, jnp.array(True), level_map
                    def _lose_life(_):
                        new_main_key, subkey_for_reset = jax.random.split(state_to_reset.key)
                        obs_reset, map_state = self.reset_map(subkey_for_reset, lives=lives_next, score=state_to_reset.score)
                        map_state = map_state._replace(key=new_main_key)
                        return obs_reset, map_state, reward_map, jnp.array(False), death_info, jnp.array(True), level_map
                    return jax.lax.cond(lives_next <= 0, _game_over, _lose_life, operand=None)

                def _on_win(_):
                    new_main_key, subkey_for_reset = jax.random.split(state_to_reset.key)
                    obs_reset, map_state = self.reset_map(subkey_for_reset, lives=state_to_reset.lives, score=state_to_reset.score)
                    map_state = map_state._replace(key=new_main_key)
                    win_info = {**info_map, "level_cleared": jnp.array(True)}
                    return obs_reset, map_state, reward_map, jnp.array(False), win_info, jnp.array(True), level_map

                return jax.lax.cond(is_a_death_event, _on_death, _on_win, operand=None)

            return jax.lax.cond(level >= 0, _enter_level, _return_to_map, operand=operands)

        def _no_reset(operands):
            obs, new_env_state, reward, done, info, reset, level = operands
            no_reset_info = {**info, "level_cleared": jnp.array(False)}
            return obs, new_env_state, reward, done, no_reset_info, reset, level

        obs, new_env_state, reward, done, info, reset, level = step_core(env_state, action)
        
        operands = (obs, new_env_state, reward, done, info, reset, level)

        return jax.lax.cond(reset, _handle_reset, _no_reset, operands)
            
    def step(self, env_state: EnvState, action: int):
        obs, ns, reward, done, info, _reset, _level = self.step_full(env_state, action)

        try:
            reward = float(reward.item() if hasattr(reward, "item") else reward)
        except Exception:
            pass
        try:
            done = bool(done.item() if hasattr(done, "item") else done)
        except Exception:
            pass

        return obs, ns, reward, done, info

class GravitarRenderer(JAXGameRenderer):
    def __init__(self, width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT):
        super().__init__()
        self.width = width
        self.height = height

        jax_sprites = _load_and_convert_sprites()
        
        def _no_op_blit(frame, x, y):
            return frame

        blit_functions = {}
        for sprite_idx, sprite_array_rgba in jax_sprites.items():
            if sprite_array_rgba is None: continue
            
            def make_blit_func(sprite_data):
                sprite_h, sprite_w, _ = sprite_data.shape
                sprite_rgb = sprite_data[..., :3]
                sprite_alpha = (sprite_data[..., 3] / 255.0)[..., None]
                def _blit_sprite(frame, x, y):
                    start_x = jnp.round(x - sprite_w / 2).astype(jnp.int32)
                    start_y = jnp.round(y - sprite_h / 2).astype(jnp.int32)
                    target_patch = jax.lax.dynamic_slice(frame, (start_y, start_x, 0), (sprite_h, sprite_w, 3))
                    blended_patch = sprite_rgb * sprite_alpha + target_patch * (1 - sprite_alpha)
                    return jax.lax.dynamic_update_slice(frame, blended_patch.astype(jnp.uint8), (start_y, start_x, 0))
                return _blit_sprite
            blit_functions[sprite_idx] = make_blit_func(sprite_array_rgba)

        max_idx = max(jax_sprites.keys()) if jax_sprites else -1
        self.blit_branches = tuple(blit_functions.get(i, _no_op_blit) for i in range(max_idx + 1))

        idle_sprite = jax_sprites.get(int(SpriteIdx.SHIP_IDLE))
        crash_sprite = jax_sprites.get(int(SpriteIdx.SHIP_CRASH))
        thrust_sprite = jax_sprites.get(int(SpriteIdx.SHIP_THRUST))

        # Unify the dimensions of all ship state sprites for use with lax.cond
        if all(s is not None for s in [idle_sprite, crash_sprite, thrust_sprite]):
            max_h = max(s.shape[0] for s in [idle_sprite, crash_sprite, thrust_sprite])
            max_w = max(s.shape[1] for s in [idle_sprite, crash_sprite, thrust_sprite])
            def pad_sprite(sprite, h, w):
                pad_h = (h - sprite.shape[0]) // 2; pad_w = (w - sprite.shape[1]) // 2
                return jnp.pad(sprite, ((pad_h, h - sprite.shape[0] - pad_h), (pad_w, w - sprite.shape[1] - pad_w), (0, 0)))
            self.padded_ship_idle = pad_sprite(idle_sprite, max_h, max_w)
            self.padded_ship_crash = pad_sprite(crash_sprite, max_h, max_w)
            self.padded_ship_thrust = pad_sprite(thrust_sprite, max_h, max_w)
        else:
            self.padded_ship_idle = self.padded_ship_crash = self.padded_ship_thrust = jnp.zeros((1,1,4), dtype=jnp.uint8)

    @partial(jax.jit, static_argnames=('self',))
    def render(self, state: EnvState) -> jnp.ndarray:
        H, W = self.height, self.width
        frame = jnp.zeros((H, W, 3), dtype=jnp.uint8)

        # --- 1. Draw map elements (only in map mode) ---
        def draw_map_elements(f):
            def draw_one_planet(i, frame_carry):
                sprite_idx = state.planets_pi[i]
                x, y = state.planets_px[i], state.planets_py[i]
                
                is_cleared = state.planets_cleared_mask[i]
                is_reactor_and_destroyed = (sprite_idx == int(SpriteIdx.REACTOR)) & state.reactor_destroyed
                should_draw = ~(is_cleared | is_reactor_and_destroyed)

                def perform_blit(fc):
                    safe_idx = jnp.clip(sprite_idx, 0, len(self.blit_branches) - 1)
                    branches = tuple(lambda op, b=b: b(op[0], op[1], op[2]) for b in self.blit_branches)
                    return jax.lax.switch(safe_idx, branches, (fc, x, y))
                
                return jax.lax.cond(should_draw, perform_blit, lambda fc: fc, frame_carry)

            return jax.lax.fori_loop(0, state.planets_pi.shape[0], draw_one_planet, f)
        frame = jax.lax.cond(state.mode == 0, draw_map_elements, lambda f: f, frame)

        # --- 2. Draw terrain (only in level mode) ---
        def draw_level_terrain(f):
            bank_idx = jnp.clip(state.terrain_bank_idx, 0, state.terrain_bank.shape[0] - 1)
            terrain_map = state.terrain_bank[bank_idx]
            is_terrain_pixel = jnp.sum(terrain_map, axis=-1) > 0
            return jnp.where(is_terrain_pixel[..., None], terrain_map, f)
        frame = jax.lax.cond(state.mode == 1, draw_level_terrain, lambda f: f, frame)

        # --- 3. Draw enemies (only in level mode) ---
        def draw_enemies(f):
            def draw_enemy_func(i, current_frame):
                is_alive = state.enemies.w[i] > 0
                sprite_idx = state.enemies.sprite_idx[i]
                x, y = state.enemies.x[i], state.enemies.y[i]
                
                def perform_blit(frame_in):
                    safe_idx = jnp.clip(sprite_idx, 0, len(self.blit_branches) - 1)
                    branches = tuple(lambda op, b=b: b(op[0], op[1], op[2]) for b in self.blit_branches)
                    return jax.lax.switch(safe_idx, branches, (frame_in, x, y))
                
                # It should pass `current_frame` through.
                return jax.lax.cond(is_alive, perform_blit, lambda f_in: f_in, current_frame)
            return jax.lax.fori_loop(0, MAX_ENEMIES, draw_enemy_func, f)
        
        def draw_ufo(f):
            ufo = state.ufo
            def perform_blit(frame_in):
                blit_func = self.blit_branches[int(SpriteIdx.ENEMY_UFO)]
                return blit_func(frame_in, ufo.x, ufo.y)
            return jax.lax.cond(ufo.alive, perform_blit, lambda f_in: f_in, f)

        # In level mode, draw enemies and UFO
        def draw_level_actors(f):
            frame_with_enemies = draw_enemies(f)
            frame_with_ufo = draw_ufo(frame_with_enemies)
            return frame_with_ufo
        
        frame = jax.lax.cond(state.mode == 1, draw_level_actors, lambda f: f, frame)

        def draw_saucer(f):
            saucer = state.saucer
            # Check if the Saucer is alive
            def perform_blit(frame_in):
                blit_func = self.blit_branches[int(SpriteIdx.ENEMY_SAUCER)]
                return blit_func(frame_in, saucer.x, saucer.y)

            return jax.lax.cond(saucer.alive, perform_blit, lambda f_in: f_in, f)
        
        # Only draw in mode 0 (map) or 2 (arena)
        frame = jax.lax.cond((state.mode == 0) | (state.mode == 2), draw_saucer, lambda f: f, frame)
        
        # --- 4. Draw bullets ---
        def draw_bullets_func(bullets, sprite_idx, current_frame):
            blit_func = self.blit_branches[sprite_idx]
            def draw_one_bullet(i, f):
                x, y = bullets.x[i], bullets.y[i]
                is_alive = bullets.alive[i]
                return jax.lax.cond(is_alive, lambda frame_in: blit_func(frame_in, x, y), lambda frame_in: frame_in, f)
            return jax.lax.fori_loop(0, bullets.x.shape[0], draw_one_bullet, current_frame)
        
        frame = draw_bullets_func(state.bullets, int(SpriteIdx.SHIP_BULLET), frame)
        frame = draw_bullets_func(state.enemy_bullets, int(SpriteIdx.ENEMY_BULLET), frame)
        frame = draw_bullets_func(state.ufo_bullets, int(SpriteIdx.ENEMY_BULLET), frame)

        # --- 5. Draw the ship ---
        ship_state = state.state
        is_crashing = state.crash_timer > 0
        velocity_sq = ship_state.vx**2 + ship_state.vy**2
        is_thrusting = velocity_sq > 0.01
        
        ship_sprite_data = jax.lax.select(is_crashing, self.padded_ship_crash,
                                        jax.lax.select(is_thrusting, self.padded_ship_thrust, self.padded_ship_idle))
        
        angle_deg = -jnp.degrees(ship_state.angle) - 90.0
        rotated_ship_rgba = _jax_rotate(ship_sprite_data, angle_deg, reshape=False)
        
        ship_h, ship_w, _ = rotated_ship_rgba.shape
        ship_rgb = rotated_ship_rgba[..., :3]; ship_alpha = (rotated_ship_rgba[..., 3] / 255.0)[..., None]
        start_x = jnp.round(ship_state.x - ship_w / 2).astype(jnp.int32)
        start_y = jnp.round(ship_state.y - ship_h / 2).astype(jnp.int32)

        target_patch = jax.lax.dynamic_slice(frame, (start_y, start_x, 0), (ship_h, ship_w, 3))
        blended_patch = ship_rgb * ship_alpha + target_patch * (1 - ship_alpha)
        frame = jax.lax.dynamic_update_slice(frame, blended_patch.astype(jnp.uint8), (start_y, start_x, 0))

        # --- 6. Draw the HUD ---
        def draw_hud(f):
            # --- Common parameters ---
            RIGHT_MARGIN = 15  # Distance of all elements from the right border
            DIGIT_WIDTH = 8    # Width of each digit
            HP_WIDTH = 8       # Width of each life icon
            Y_SCORE = 10       # Y-coordinate for the score
            Y_LIVES = 22       # Y-coordinate for the lives

            # --- Draw the score ---
            score_val = state.score.astype(jnp.int32)
            digits = jnp.array([(score_val // 10**(5-i)) % 10 for i in range(6)])
            
            def draw_one_digit(i, frame_carry):
                sprite_idx = digits[i] + int(SpriteIdx.DIGIT_0)
                # Calculate position starting from the right border
                x_pos = W - RIGHT_MARGIN - (6 - i) * DIGIT_WIDTH
                y_pos = Y_SCORE
                
                operand = (frame_carry, x_pos, y_pos)
                branches = tuple(lambda op, branch=b: branch(op[0], op[1], op[2]) for b in self.blit_branches)
                safe_idx = jnp.clip(sprite_idx, 0, len(branches) - 1)
                
                return jax.lax.switch(safe_idx, branches, operand)

            score_frame = jax.lax.fori_loop(0, 6, draw_one_digit, f)
            
            # --- Draw the lives ---
            hp_blit_func = self.blit_branches[int(SpriteIdx.HP_UI)]
            def draw_one_life(i, frame_carry):
                is_active = i < state.lives
                # Also calculate position starting from the right border
                x_pos = W - RIGHT_MARGIN - (MAX_LIVES - i) * HP_WIDTH
                y_pos = Y_LIVES # Use the new Y-coordinate to ensure it's below the score
                return jax.lax.cond(is_active, lambda fc: hp_blit_func(fc, x_pos, y_pos), lambda fc: fc, frame_carry)
            
            lives_frame = jax.lax.fori_loop(0, MAX_LIVES, draw_one_life, score_frame)
            return lives_frame
        
        frame = jax.lax.cond(state.mode < 2, draw_hud, lambda f: f, frame)

        return frame

__all__ = ["JaxGravitar", "get_env_and_renderer"]

def get_env_and_renderer():
    env = JaxGravitar()
    # Just instantiate it, or pass in your game resolution as parameters
    renderer = GravitarRenderer(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    return env, renderer
