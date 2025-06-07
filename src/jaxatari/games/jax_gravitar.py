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

# HUD settings
HUD_HEIGHT = 30
MAX_LIVES = 6
HUD_PADDING = 5
HUD_SHIP_WIDTH = 10
HUD_SHIP_HEIGHT = 12
HUD_SHIP_SPACING = 12

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# Fix maximum bullets number
MAX_PLAYER_BULLETS = 50
MAX_ENEMY_BULLETS = 100
MAX_ENEMIES = 5


def get_action_from_key():
    keys = pygame.key.get_pressed()

    thrust = keys[pygame.K_UP]
    rotate_left = keys[pygame.K_LEFT]
    rotate_right = keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]
    down = keys[pygame.K_DOWN]

    # Map action combinations    映射动作组合
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

    # default return to action 0     默认返回action 0
    return 0


def add_bullet(bullets: dict, new_bullet: dict) -> dict:
    # Convert the new bullet to jnp.array and concatenate    单个新子弹转为 jnp.array 并拼接
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
    """
        Updates the state of all bullets for one frame.

        This function moves active bullets based on their velocity, checks if they
        have gone off-screen, and updates their active status accordingly.
        Inactive bullets are not moved but their state is preserved.

        Args:
            bullets (dict): A dictionary representing the collection of bullets.
                            It must contain the following keys with jnp.ndarray values:
                            'x': x-coordinates
                            'y': y-coordinates
                            'vx': velocities on the x-axis
                            'vy': velocities on the y-axis
                            'active': a boolean array, True if the bullet is active.

        Returns:
            dict: The updated bullets dictionary with new positions and active statuses.
        """
    # Update positions for active bullets only 只更新活跃子弹
    new_x = jnp.where(bullets["active"], bullets["x"] + bullets["vx"], bullets["x"])
    new_y = jnp.where(bullets["active"], bullets["y"] + bullets["vy"], bullets["y"])

    #  Check which bullets are within the valid screen area
    valid = ((new_y >= HUD_HEIGHT) & (new_y <= WINDOW_HEIGHT) &
             (new_x >= 0) & (new_x <= WINDOW_WIDTH))

    return {
        # If a bullet is valid, use its new position. Otherwise, move it to an off-screen location (-1.0).
        "x": jnp.where(valid, new_x, -1.0),
        "y": jnp.where(valid, new_y, -1.0),
        "vx": bullets["vx"],
        "vy": bullets["vy"],
        "active": bullets["active"] & valid
    }


@jax.jit
def ship_step(state, action, window_size, hud_height) -> dict:
    """
       Updates the ship's state for one frame based on the given action.

       This version uses a simple "clipping" mechanism for boundary collisions,
       meaning the ship will stop dead when it hits a wall.

       Args:
           state (dict): A dictionary containing the ship's current state:
                         'x', 'y', 'vx', 'vy', 'angle'.
           action (int): An integer representing the player's action.
           window_size (tuple): A tuple (width, height) for the game window.
           hud_height (int): The height of the top HUD area, defining the top boundary.

       Returns:
           dict: The updated state dictionary for the ship.
       """
    # basic parameters   基本参数
    rotation_speed = 0.1
    thrust_power = 0.25
    gravity = 0.08
    bounce_damping = 0.2  # Damping factor for bounce velocity

    # Define action sets: These arrays map specific 'action' integers to corresponding movements.
    # A single action can trigger multiple effects if its number appears in multiple sets.
    rotate_right_actions = jnp.array([3, 6, 8, 11, 14, 16])
    rotate_left_actions  = jnp.array([4, 7, 9, 12, 15, 17])
    thrust_actions       = jnp.array([2, 6, 7, 10, 14, 15])
    down_thrust_actions  = jnp.array([5, 8, 9, 13, 16, 17])

    # Check which action is being performed
    right       = jnp.isin(action, rotate_right_actions)
    left        = jnp.isin(action, rotate_left_actions)
    thrust      = jnp.isin(action, thrust_actions)
    down_thrust = jnp.isin(action, down_thrust_actions)

    # Update Angle
    angle = jnp.where(right, state["angle"] + rotation_speed, state["angle"])
    angle = jnp.where(left,  angle - rotation_speed, angle)

    # Update Velocity
    vx = jnp.where(thrust,      state["vx"] + jnp.cos(angle) * thrust_power, state["vx"])
    vy = jnp.where(thrust,      state["vy"] + jnp.sin(angle) * thrust_power, state["vy"])

    # Apply downward thrust：Subtracts velocity component in the direction the ship is facing.
    vx = jnp.where(down_thrust, vx - jnp.cos(angle) * thrust_power, vx)
    vy = jnp.where(down_thrust, vy - jnp.sin(angle) * thrust_power, vy)

    vy += gravity

    # Predict Next Position (before collision checks)
    next_x_unclipped = state["x"] + vx
    next_y_unclipped = state["y"] + vy

    window_width, window_height = window_size

    ship_half_size = 5  # Approximate half size for simpler collision with boundaries

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
    x = state["x"] + vx  # Use the (potentially bounced) new vx
    x = jnp.clip(x, ship_half_size, window_width - ship_half_size)  # Clip to ensure it's precisely at the edge

    # Vertical bounce
    hit_top = next_y_unclipped < hud_height + ship_half_size
    hit_bottom = next_y_unclipped > window_height - ship_half_size

    vy = jnp.where(hit_top | hit_bottom, -old_vy * bounce_damping, old_vy)

    # Update position based on potentially bounced velocity, then clip
    y = state["y"] + vy  # Use the (potentially bounced) new vy
    y = jnp.clip(y, hud_height + ship_half_size, window_height - ship_half_size)

    return {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "angle": angle
    }

@jax.jit
def compact_bullets(bullets_dict: dict) -> dict:
    """
    Compacts the bullet dictionary by moving all active bullets to the front
    of the arrays and inactive bullets to the back. It then resets the data
    (position, velocity, active status) for the inactive bullets, preparing
    them for reuse in an object pooling system.

    This function is crucial for efficiently managing a fixed-size pool of
    game objects (like bullets) in a JAX/NumPy environment, avoiding repeated
    memory allocation and deallocation.

    Args:
        bullets_dict (dict): A dictionary containing bullet properties,
                             expected to have 'x', 'y', 'vx', 'vy', and 'active'
                             keys, where values are JAX NumPy arrays of the same
                             length.

    Returns:
        dict: The compacted bullet dictionary with active bullets ordered first
              and inactive bullets reset at the end, ready for recycling.
    """

    # Get the boolean mask indicating which bullets are currently active.
    active_mask = bullets_dict["active"]

    # Generate sorting keys to order active bullets before inactive ones                    创建用于索引排序的键数组
    sorting_keys = -active_mask.astype(jnp.int32)  # active=True -> -1, active=False -> 0   对子弹是否active的数组取负，让active的排在前面

    # Create an indices describe the new positions for each bullet's data                   创建索引排序数组，
    sorted_indices = jnp.argsort(sorting_keys)                                              # E.g., if sorting_keys was [0, -1, 0, -1], sorted_indices might be [2, 0, 3, 1].

    # Rearrange all bullet properties consistently using the sorted indices                 根据排序索引重新排列所有子弹属性
    x_compact = bullets_dict["x"][sorted_indices]
    y_compact = bullets_dict["y"][sorted_indices]
    vx_compact = bullets_dict["vx"][sorted_indices]
    vy_compact = bullets_dict["vy"][sorted_indices]
    active_compact = bullets_dict["active"][sorted_indices]

    # Create a boolean mask to mark the slots now occupied by inactive bullets              定义要重置的子弹的位置的掩码
    clear_mask = ~active_compact

    # Reset the properties of the inactive bullets                                          如果 clear_mask 为 True，则将对应元素设置为 -1.0/0.0/False
    x_compact = jnp.where(clear_mask, -1.0, x_compact)
    y_compact = jnp.where(clear_mask, -1.0, y_compact)
    vx_compact = jnp.where(clear_mask, 0.0, vx_compact)
    vy_compact = jnp.where(clear_mask, 0.0, vy_compact)

    return {
        "x": x_compact,
        "y": y_compact,
        "vx": vx_compact,
        "vy": vy_compact,
        "active": active_compact
    }


@jax.jit
def enemy_step(enemies: dict, window_width: int) -> dict:
    x = enemies["x"] + enemies["vx"]

    # Reverse direction at the boundaries   到边界就反向
    left_hit = x <= 0
    right_hit = (x + enemies["w"]) >= window_width
    hit_edge = left_hit | right_hit

    vx = jnp.where(hit_edge, -enemies["vx"], enemies["vx"])

    return {
        "x": x,
        "y": enemies["y"],
        "w": enemies["w"],
        "h": enemies["h"],
        "vx": vx,
        "active": enemies["active"]
    }


@jax.jit
def enemy_fire(enemies: dict,
               ship_x: float,
               ship_y: float,
               enemy_bullet_speed: float,
               global_fire_cooldown: int,  # 全局敌人开火CD
               fire_interval: int,
               key: jax.random.PRNGKey,
               max_bullets_to_spawn_this_tick: int
               ) -> Tuple[dict, int, jax.random.PRNGKey]:
    # 初始化这一帧要产生的子弹的临时存储 (固定大小为 max_bullets_to_spawn_this_tick)
    new_bullets_x = jnp.full((MAX_ENEMIES,), -1.0)
    new_bullets_y = jnp.full((MAX_ENEMIES,), -1.0)
    new_bullets_vx = jnp.zeros((MAX_ENEMIES,))
    new_bullets_vy = jnp.zeros((MAX_ENEMIES,))
    new_bullets_active = jnp.full((MAX_ENEMIES,), False, dtype=bool)

    # 判断哪些敌人可以在这一帧开火
    can_any_enemy_fire_this_step = (global_fire_cooldown == 0)

    # 循环体函数，由 jax.lax.fori_loop 调用
    def body_fun(i, carry_state):
        # carry_state: (current_nb_x, current_nb_y, current_nb_vx, current_nb_vy, current_nb_active, current_spawn_idx, cd_allows)
        c_nb_x, c_nb_y, c_nb_vx, c_nb_vy, c_nb_active, c_spawn_idx, cd_allows = carry_state

        # 当前敌人是否活跃，并且全局CD允许开火
        is_enemy_active = enemies["active"][i]
        should_spawn_bullet = is_enemy_active & cd_allows & (c_spawn_idx < max_bullets_to_spawn_this_tick)

        # 计算子弹方向和速度
        dx = ship_x - enemies["x"][i]
        dy = ship_y - enemies["y"][i]
        dist = jnp.sqrt(dx ** 2 + dy ** 2) + 1e-6  # 防止除以零
        bullet_vx = dx / dist * enemy_bullet_speed
        bullet_vy = dy / dist * enemy_bullet_speed

        # 更新临时子弹数组和激活掩码 (使用 c_spawn_idx 作为索引)
        # 只有在 should_spawn_bullet 为 True 时才写入新值
        new_nb_x = c_nb_x.at[c_spawn_idx].set(jnp.where(should_spawn_bullet, enemies["x"][i], c_nb_x[c_spawn_idx]))
        new_nb_y = c_nb_y.at[c_spawn_idx].set(jnp.where(should_spawn_bullet, enemies["y"][i], c_nb_y[c_spawn_idx]))
        new_nb_vx = c_nb_vx.at[c_spawn_idx].set(jnp.where(should_spawn_bullet, bullet_vx, c_nb_vx[c_spawn_idx]))
        new_nb_vy = c_nb_vy.at[c_spawn_idx].set(jnp.where(should_spawn_bullet, bullet_vy, c_nb_vy[c_spawn_idx]))
        new_nb_active = c_nb_active.at[c_spawn_idx].set(jnp.where(should_spawn_bullet, True, c_nb_active[c_spawn_idx]))

        # 更新下一个子弹写入的索引
        new_spawn_idx = c_spawn_idx + jnp.where(should_spawn_bullet, 1, 0)

        return (new_nb_x, new_nb_y, new_nb_vx, new_nb_vy, new_nb_active, new_spawn_idx, cd_allows)

    # 循环所有敌人（固定迭代次数为 MAX_ENEMIES）
    initial_carry = (new_bullets_x, new_bullets_y, new_bullets_vx, new_bullets_vy, new_bullets_active,
                     jnp.array(0, dtype=jnp.int32),  # initial spawn index
                     can_any_enemy_fire_this_step)

    # 注意：这里的循环次数是 MAX_ENEMIES，这是一个静态值
    final_nb_x, final_nb_y, final_nb_vx, final_nb_vy, final_nb_active, final_count, _ = jax.lax.fori_loop(
        0, MAX_ENEMIES, body_fun, initial_carry
    )

    bullets_fired_this_frame = {
        "x": final_nb_x,
        "y": final_nb_y,
        "vx": final_nb_vx,
        "vy": final_nb_vy,
        "active": final_nb_active  # 返回激活掩码
    }

    # 更新全局开火冷却
    new_global_fire_cooldown = jnp.where(
        can_any_enemy_fire_this_step & (final_count > 0),  # 只有在允许开火且确实有子弹发出时才重置CD
        fire_interval,
        jnp.maximum(global_fire_cooldown - 1, 0)
    )

    return bullets_fired_this_frame, new_global_fire_cooldown, key


@jax.jit
def check_collision(ship_state, bullets, enemy_bullets, enemies, ship_hitbox_size):
    # 移除 safe_filter calls. 传入的 bullets/enemy_bullets/enemies 已经是带有 active 掩码的字典

    # ship hitbox
    sx1 = ship_state["x"] - ship_hitbox_size
    sx2 = ship_state["x"] + ship_hitbox_size
    sy1 = ship_state["y"] - ship_hitbox_size
    sy2 = ship_state["y"] + ship_hitbox_size

    # 使用传入的 active 状态
    current_player_bullets_active = bullets["active"]
    current_enemies_active = enemies["active"]
    current_enemy_bullets_active = enemy_bullets["active"]

    # 玩家子弹与敌人碰撞
    ex1 = enemies["x"] - enemies["w"] / 2
    ex2 = enemies["x"] + enemies["w"] / 2
    ey1 = enemies["y"] - enemies["h"] / 2
    ey2 = enemies["y"] + enemies["h"] / 2

    bullet_enemy_collision_matrix = (
            (bullets["x"][:, None] >= ex1[None, :]) & (bullets["x"][:, None] <= ex2[None, :]) &
            (bullets["y"][:, None] >= ey1[None, :]) & (bullets["y"][:, None] <= ey2[None, :])
    )

    # 只有当玩家子弹和敌人都是活跃的时，才考虑碰撞
    actual_bullet_enemy_collisions = bullet_enemy_collision_matrix & \
                                     current_player_bullets_active[:, None] & \
                                     current_enemies_active[None, :]

    # 被击中的玩家子弹
    bullet_hit_any_enemy = jnp.any(actual_bullet_enemy_collisions, axis=1)  # 哪个玩家子弹击中了敌人

    # 被击中的敌人
    enemy_hit_by_any_bullet = jnp.any(actual_bullet_enemy_collisions, axis=0)  # 哪个敌人被子弹击中了

    # 更新玩家子弹的活跃状态
    new_player_bullets_active = current_player_bullets_active & ~bullet_hit_any_enemy

    # 更新敌人的活跃状态
    new_enemies_active = current_enemies_active & ~enemy_hit_by_any_bullet

    # 飞船与敌人碰撞
    ship_enemy_overlap_x = (sx1 <= ex2) & (sx2 >= ex1)
    ship_enemy_overlap_y = (sy1 <= ey2) & (sy2 >= ey1)

    # 飞船与所有活跃敌人的碰撞
    ship_enemy_collision_vector = ship_enemy_overlap_x & ship_enemy_overlap_y & new_enemies_active
    crashed = jnp.any(ship_enemy_collision_vector)

    # 敌方子弹与飞船碰撞
    enemy_bullet_ship_collision_vector = (
            (enemy_bullets["x"] >= sx1) & (enemy_bullets["x"] <= sx2) &
            (enemy_bullets["y"] >= sy1) & (enemy_bullets["y"] <= sy2)
    )

    # 只有当敌方子弹是活跃的时，才考虑碰撞
    active_enemy_bullet_collided_ship = enemy_bullet_ship_collision_vector & current_enemy_bullets_active
    hit_by_bullet = jnp.any(active_enemy_bullet_collided_ship)

    # 更新敌方子弹的活跃状态：它本来活跃 且 没有击中飞船
    new_enemy_bullets_active = current_enemy_bullets_active & ~active_enemy_bullet_collided_ship

    # 构建并返回更新后的字典
    updated_player_bullets = bullets.copy()
    updated_player_bullets["active"] = new_player_bullets_active

    updated_enemies = enemies.copy()
    updated_enemies["active"] = new_enemies_active

    updated_enemy_bullets = enemy_bullets.copy()
    updated_enemy_bullets["active"] = new_enemy_bullets_active

    return updated_player_bullets, updated_enemies, updated_enemy_bullets, crashed, hit_by_bullet


@jax.jit
def merge_bullets_into_pool(pool_bullets: dict, batch_bullets: dict, max_pool_size: int) -> dict:
    """
    将一批新子弹 (batch_bullets) 合并到预分配的子弹池 (pool_bullets) 中。
    pool_bullets 应该已经被 compact_bullets 处理过，以确保活跃子弹在数组前端。
    """
    # 获取当前池中已有的活跃子弹数量，作为写入新子弹的起始索引
    initial_pool_num_active = jnp.sum(pool_bullets["active"])

    def _merge_loop_body(i, carry):
        # carry: (current_pool_x, ..., current_pool_active, next_slot_in_pool_frontier_index)
        c_pool_x, c_pool_y, c_pool_vx, c_pool_vy, c_pool_active, c_frontier = carry

        # 检查此批次中的第 `i` 个子弹是否活跃
        is_source_bullet_active = batch_bullets["active"][i]

        # 检查池中是否有空间容纳新子弹，并且批次中的子弹是活跃的
        can_place = is_source_bullet_active & (c_frontier < max_pool_size)

        # 在前沿索引处更新池中的子弹数据
        new_pool_x = c_pool_x.at[c_frontier].set(jnp.where(can_place, batch_bullets["x"][i], c_pool_x[c_frontier]))
        new_pool_y = c_pool_y.at[c_frontier].set(jnp.where(can_place, batch_bullets["y"][i], c_pool_y[c_frontier]))
        new_pool_vx = c_pool_vx.at[c_frontier].set(jnp.where(can_place, batch_bullets["vx"][i], c_pool_vx[c_frontier]))
        new_pool_vy = c_pool_vy.at[c_frontier].set(jnp.where(can_place, batch_bullets["vy"][i], c_pool_vy[c_frontier]))
        new_pool_active = c_pool_active.at[c_frontier].set(jnp.where(can_place, True, c_pool_active[c_frontier]))

        # 如果成功放置了子弹，则前沿索引向前推进
        new_frontier = c_frontier + jnp.where(can_place, 1, 0)

        return (new_pool_x, new_pool_y, new_pool_vx, new_pool_vy, new_pool_active, new_frontier)

    num_batch_bullets_to_check = batch_bullets["x"].shape[0]

    final_px, final_py, final_pvx, final_pvy, final_pa, _ = jax.lax.fori_loop(
        0, num_batch_bullets_to_check, _merge_loop_body,
        (pool_bullets["x"], pool_bullets["y"], pool_bullets["vx"], pool_bullets["vy"],
         pool_bullets["active"], initial_pool_num_active)
    )

    return {
        "x": final_px, "y": final_py, "vx": final_pvx, "vy": final_pvy, "active": final_pa
    }


@jax.jit
def step_core_level(state: dict,
                    action: int,
                    player_bullets_pool: dict,  # 玩家子弹池
                    cooldown: int,
                    current_enemies_state: dict,  # 敌人状态
                    enemy_bullets_pool: dict,  # 敌方子弹池
                    enemy_fire_cooldown: int,
                    key: jax.random.PRNGKey,
                    window_size: Tuple[int, int],
                    hud_height: int,
                    cooldown_max: int,
                    bullet_speed: float,
                    enemy_bullet_speed: float,
                    fire_interval: int,
                    max_player_bullets: int,
                    max_enemy_bullets: int,
                    max_enemies_in_level: int
                    ) -> Tuple[jnp.ndarray, dict, dict, int, dict, dict, int, jax.random.PRNGKey, float, bool, dict]:
    # 飞船移动
    new_ship_state = ship_step(state, action, window_size, hud_height)

    # 玩家子弹生命周期
    fire_actions = jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])
    is_firing = jnp.isin(action, fire_actions)
    # 发射新的玩家子弹 (放入玩家子弹池)
    player_bullets_after_fire, new_cooldown = fire_bullet(
        new_ship_state, is_firing, cooldown, cooldown_max, bullet_speed, player_bullets_pool
    )
    # 更新所有玩家子弹位置
    player_bullets_after_movement = update_bullets(player_bullets_after_fire)

    # 敌人生命周期
    # 敌人移动
    enemies_after_movement = enemy_step(current_enemies_state, window_width=window_size[0])

    # 敌方子弹生命周期
    # 更新现有敌方子弹位置
    enemy_bullets_after_movement = update_bullets(enemy_bullets_pool)

    # 敌人开火 (产生一批新子弹)
    newly_fired_enemy_bullets_batch, new_enemy_fire_cooldown, key = enemy_fire(
        enemies_after_movement,  # 传入移动后的敌人状态
        new_ship_state["x"], new_ship_state["y"],
        enemy_bullet_speed,
        enemy_fire_cooldown,  # 当前敌方开火冷却
        fire_interval,
        key,
        max_bullets_to_spawn_this_tick=max_enemies_in_level  # 传入静态敌人最大数量作为批次大小
    )

    # 将新生成的敌方子弹合并到现有敌方子弹池中
    # 合并前先对池进行压缩（稍后处理冲突并压缩）或者这里直接合并
    enemy_bullets_after_merge = merge_bullets_into_pool(
        enemy_bullets_after_movement,  # 已移动的敌方子弹池
        newly_fired_enemy_bullets_batch,
        max_enemy_bullets
    )

    # 碰撞检测
    # 碰撞检测发生在所有物体都移动完毕，并且所有新的子弹都已生成并加入到各自的池中之后。
    player_bullets_after_collision, enemies_after_collision, enemy_bullets_after_collision, crashed, hit_by_bullet = check_collision(
        new_ship_state,
        player_bullets_after_movement,  # 移动后的玩家子弹
        enemy_bullets_after_merge,  # 移动并合并后的敌方子弹
        enemies_after_movement,  # 移动后的敌人
        ship_hitbox_size=10.0
    )

    # 整理子弹和敌人池 (将在碰撞中被击中的标记为非活跃，并将活跃的元素紧凑排列到数组前端)
    final_player_bullets_pool = compact_bullets(player_bullets_after_collision)
    final_enemy_bullets_pool = compact_bullets(enemy_bullets_after_collision)

    #construct the observation    构建观测
    obs = jnp.array([new_ship_state["x"], new_ship_state["y"], new_ship_state["vx"], new_ship_state["vy"], new_ship_state["angle"]])

    #if dead?
    dead = crashed | hit_by_bullet
    reward = jnp.where(dead, -10.0, -1.0)
    done = dead

    info = {
        "crash": crashed,
        "hit_by_bullet": hit_by_bullet
    }

    return obs, new_ship_state, final_player_bullets_pool, new_cooldown, \
        enemies_after_collision, final_enemy_bullets_pool, new_enemy_fire_cooldown, \
        key, reward, done, info


@jax.jit
def step_core_map(state: dict,
                  action: int,
                  window_size: Tuple[int, int],
                  hud_height: int
                  ) -> Tuple[jnp.ndarray, dict, float, bool, dict, bool, int]:

    # use the ship_step function          调用 ship_step 得到更新后的状态
    new_state = ship_step(state, action, window_size, hud_height)

    # construct observation   构建观测 obs
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

    # Find first inactive slot. Even if no bullet will be fired, this finds an index.
    inactive_indices = jnp.where(~bullets["active"], jnp.arange(MAX_PLAYER_BULLETS), MAX_PLAYER_BULLETS)
    idx_to_fire = jnp.min(inactive_indices)

    tip_x = state["x"] + jnp.cos(state["angle"]) * tip_offset
    tip_y = state["y"] + jnp.sin(state["angle"]) * tip_offset
    bullet_vx = jnp.cos(state["angle"]) * bullet_speed
    bullet_vy = jnp.sin(state["angle"]) * bullet_speed

    # Determine if a bullet can actually be placed at the found index
    can_fire_at_idx = should_fire & (idx_to_fire < MAX_PLAYER_BULLETS)

    new_bullets_x = bullets["x"].at[idx_to_fire].set(
        jnp.where(can_fire_at_idx, tip_x, bullets["x"][idx_to_fire])
    )
    new_bullets_y = bullets["y"].at[idx_to_fire].set(
        jnp.where(can_fire_at_idx, tip_y, bullets["y"][idx_to_fire])
    )
    new_bullets_vx = bullets["vx"].at[idx_to_fire].set(
        jnp.where(can_fire_at_idx, bullet_vx, bullets["vx"][idx_to_fire])
    )
    new_bullets_vy = bullets["vy"].at[idx_to_fire].set(
        jnp.where(can_fire_at_idx, bullet_vy, bullets["vy"][idx_to_fire])
    )
    new_bullets_active = bullets["active"].at[idx_to_fire].set(
        jnp.where(can_fire_at_idx, True, bullets["active"][idx_to_fire])
    )

    updated_bullets = {
        "x": new_bullets_x,
        "y": new_bullets_y,
        "vx": new_bullets_vx,
        "vy": new_bullets_vy,
        "active": new_bullets_active
    }

    new_cooldown = jnp.where(should_fire, cooldown_max, jnp.maximum(cooldown - 1, 0))

    return updated_bullets, new_cooldown


class JaxGravitar(JaxEnvironment):
    def __init__(self):
        super().__init__()
        self.obs_shape = (5,) # [x, y, vx, vy, angle]
        self.num_actions = 18

        # Initialize font library     初始化字体库
        pygame.font.init()
        # Initialize score and health    初始化分数和血条
        self.score = 0
        self.lives = MAX_LIVES
        # Initialize the map    初始化地图
        self.mode = "map"
        self.planets = []  # 在 reset_map 中初始化
        self.current_level = None
        self.game_over = False
        self.done = False
        self.key = jrandom.PRNGKey(0)

        #List of bullets     子弹列表
        # 初始化所有 JAX 状态属性为固定大小的 JAX 数组
        self.bullets = {
            "x": jnp.full((MAX_PLAYER_BULLETS,), -1.0, dtype=jnp.float32),
            "y": jnp.full((MAX_PLAYER_BULLETS,), -1.0, dtype=jnp.float32),
            "vx": jnp.zeros((MAX_PLAYER_BULLETS,), dtype=jnp.float32),
            "vy": jnp.zeros((MAX_PLAYER_BULLETS,), dtype=jnp.float32),
            "active": jnp.full((MAX_PLAYER_BULLETS,), False, dtype=jnp.bool_)
        }
        self.bullets_speed = 5.0

        # Enemy bullet list    enemy子弹列表
        self.enemy_bullets = {
            "x": jnp.full((MAX_ENEMY_BULLETS,), -1.0, dtype=jnp.float32),
            "y": jnp.full((MAX_ENEMY_BULLETS,), -1.0, dtype=jnp.float32),
            "vx": jnp.zeros((MAX_ENEMY_BULLETS,), dtype=jnp.float32),
            "vy": jnp.zeros((MAX_ENEMY_BULLETS,), dtype=jnp.float32),
            "active": jnp.full((MAX_ENEMY_BULLETS,), False, dtype=jnp.bool_)
        }
        self.enemy_fire_cooldown = 0
        self.enemy_fire_interval = 60
        self.enemy_bullet_speed = 5.0

        #Shooting cooldown    射击cd
        self.cooldown = 0
        self.cooldown_max = 5

        # 敌人状态
        self.enemies = {
            "x": jnp.full((MAX_ENEMIES,), -1.0, dtype=jnp.float32),
            "y": jnp.full((MAX_ENEMIES,), -1.0, dtype=jnp.float32),
            "w": jnp.full((MAX_ENEMIES,), 0.0, dtype=jnp.float32),
            "h": jnp.full((MAX_ENEMIES,), 0.0, dtype=jnp.float32),
            "vx": jnp.full((MAX_ENEMIES,), 0.0, dtype=jnp.float32),
            "active": jnp.full((MAX_ENEMIES,), False, dtype=jnp.bool_)
        }

        # Set window size     设置窗口大小
        self.screen_size = (WINDOW_WIDTH, WINDOW_HEIGHT)
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Gravitar")

    def draw_hud(self):
        # Background    背景
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, WINDOW_WIDTH, HUD_HEIGHT))

        # Score   分数
        font = pygame.font.SysFont("Arial", 14, bold=True)
        score_text = font.render(str(self.score), True, (255, 128, 255))
        text_rect = score_text.get_rect(center=(WINDOW_WIDTH //2, HUD_PADDING + 6))
        self.screen.blit(score_text, text_rect)

        # Health bar   血条
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
        return (18,)  # Supports actions numbered 0 to 17    支持编号0到17的动作

    def get_observation_space(self) -> Tuple[int]:
        """
        Returns the observation space of the environment.
        Returns: The observation space of the environment as a tuple.
        """
        # raise NotImplementedError("Abstract method")
        return (5,)

    def render(self, state: dict) -> Tuple[jnp.ndarray]:
        """
        Renders the environment state to a single image.
        Args:
            state: The environment state.

        Returns: A single image of the environment state.

        """

        self.screen.fill((0, 0, 0))
        if self.mode == "map":
            # Draw planets    画星球
            for planet in self.planets:
                color = planet.get("color", (0, 255, 255))
                pygame.draw.circle(self.screen, color, (int(planet["x"]), int(planet["y"])), int(planet["r"]))
        # Render triangular spaceship      渲染三角形飞船
        # Spaceship center position    飞船中心位置
        cx, cy = float(state["x"]), float(state["y"])
        angle = float(state["angle"])
        size = 10  # Size of the spaceship tip    飞船箭头大小
        # Three triangle vertices, angle indicates direction      三角形三个顶点位置，angle指向朝向
        tip = (cx + math.cos(angle) * size, cy + math.sin(angle) * size)
        left = (cx + math.cos(angle + 2.5) * size, cy + math.sin(angle + 2.5) * size)
        right = (cx + math.cos(angle - 2.5) * size, cy + math.sin(angle - 2.5) * size)

        # Clear background   清空背景
        self.draw_hud()

        # Draw spaceship 画飞船
        pygame.draw.polygon(self.screen, (0, 255, 0), [tip, left, right])

        # Draw enemy   画enemy
        if self.mode == "level":
            active_enemy_indices = jnp.where(self.enemies["active"])[0]
            for i_idx in range(len(active_enemy_indices)):
                i = int(active_enemy_indices[i_idx])
                ex = float(self.enemies["x"][i])
                ey = float(self.enemies["y"][i])
                ew = float(self.enemies["w"][i])
                eh = float(self.enemies["h"][i])

                # 假设敌人的 x,y 是中心点，所以需要调整为左上角坐标
                pygame.draw.rect(self.screen, (255, 0, 0), (int(ex - ew / 2), int(ey - eh / 2), int(ew), int(eh)))

        # Score  得分
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Render bullets   渲染子弹
        active_player_bullet_indices = jnp.where(self.bullets["active"])[0]

        for i_idx in range(len(active_player_bullet_indices)):
            i = int(active_player_bullet_indices[i_idx])  # 确保索引是 Python int
            bx, by = float(self.bullets["x"][i]), float(self.bullets["y"][i])
            pygame.draw.circle(self.screen, (255, 255, 0), (int(bx), int(by)), 2)

        # Render enemy bullets    渲染enemy子弹
        active_enemy_fb = jnp.where(self.enemy_bullets["active"])[0]
        for i_idx in range(len(active_enemy_fb)):
            i = int(active_enemy_fb[i_idx])
            bx, by = float(self.enemy_bullets["x"][i]), float(self.enemy_bullets["y"][i])
            pygame.draw.circle(self.screen, (255, 100, 0), (int(bx), int(by)), 3)

        # gameover screen
        if self.game_over:  # 使用 self.game_over 判断最终游戏结束
            font = pygame.font.SysFont(None, 72)
            text = font.render("game over", True, (255, 0, 0))
            text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT // 2))
            self.screen.blit(text, text_rect)


        pygame.display.flip()
        # pygame.time.delay(100) 这个是导致主界面卡的原因

        dummy_image = jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3))
        return (dummy_image,)

    def reset(self, key: jrandom.PRNGKey = None) -> Tuple[jnp.ndarray, dict]:
        self.lives = MAX_LIVES
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
        self.enemies = {
            "x": jnp.full((MAX_ENEMIES,), -1.0, dtype=jnp.float32),
            "y": jnp.full((MAX_ENEMIES,), -1.0, dtype=jnp.float32),
            "w": jnp.full((MAX_ENEMIES,), 0.0, dtype=jnp.float32),
            "h": jnp.full((MAX_ENEMIES,), 0.0, dtype=jnp.float32),
            "vx": jnp.full((MAX_ENEMIES,), 0.0, dtype=jnp.float32),
            "active": jnp.full((MAX_ENEMIES,), False, dtype=jnp.bool_)
        }
        self.planets = [
            {"x": 80, "y": 100, "r": 15, "level_id": 0, "color": (255, 100, 100)},
            {"x": 150, "y": 180, "r": 15, "level_id": 1, "color": (100, 255, 100)},
            {"x": 230, "y": 60, "r": 15, "level_id": 2, "color": (100, 100, 255)}
        ]

        state = {
            "x": jnp.array(WINDOW_WIDTH / 2, dtype=jnp.float32),
            "y": jnp.array((WINDOW_HEIGHT + HUD_HEIGHT) / 2, dtype=jnp.float32),
            "vx": jnp.array(0.0, dtype=jnp.float32),
            "vy": jnp.array(0.0, dtype=jnp.float32),
            "angle": jnp.array(-jnp.pi / 2, dtype=jnp.float32)
        }
        # 初始化 bullets
        self.bullets = {
            "x": jnp.full((MAX_PLAYER_BULLETS,), -1.0, dtype=jnp.float32),
            "y": jnp.full((MAX_PLAYER_BULLETS,), -1.0, dtype=jnp.float32),
            "vx": jnp.zeros((MAX_PLAYER_BULLETS,), dtype=jnp.float32),
            "vy": jnp.zeros((MAX_PLAYER_BULLETS,), dtype=jnp.float32),
            "active": jnp.full((MAX_PLAYER_BULLETS,), False, dtype=jnp.bool_)
        }

        self.enemy_bullets = {
            "x": jnp.full((MAX_ENEMY_BULLETS,), -1.0, dtype=jnp.float32),
            "y": jnp.full((MAX_ENEMY_BULLETS,), -1.0, dtype=jnp.float32),
            "vx": jnp.zeros((MAX_ENEMY_BULLETS,), dtype=jnp.float32),
            "vy": jnp.zeros((MAX_ENEMY_BULLETS,), dtype=jnp.float32),
            "active": jnp.full((MAX_ENEMY_BULLETS,), False, dtype=jnp.bool_)
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
        # Initialize spaceship state     初始化飞船状态
        state = {
            "x": jnp.array(WINDOW_WIDTH / 2, dtype=jnp.float32),
            "y": jnp.array((WINDOW_HEIGHT + HUD_HEIGHT) / 2, dtype=jnp.float32),
            "vx": jnp.array(0.0, dtype=jnp.float32),
            "vy": jnp.array(0.0, dtype=jnp.float32),
            "angle": jnp.array(-jnp.pi / 2, dtype=jnp.float32)
        }

        self.bullets = {
            "x": jnp.full((MAX_PLAYER_BULLETS,), -1.0, dtype=jnp.float32),
            "y": jnp.full((MAX_PLAYER_BULLETS,), -1.0, dtype=jnp.float32),
            "vx": jnp.zeros((MAX_PLAYER_BULLETS,), dtype=jnp.float32),
            "vy": jnp.zeros((MAX_PLAYER_BULLETS,), dtype=jnp.float32),
            "active": jnp.full((MAX_PLAYER_BULLETS,), False, dtype=jnp.bool_)
        }

        self.enemy_bullets = {
            "x": jnp.full((MAX_ENEMY_BULLETS,), -1.0, dtype=jnp.float32),
            "y": jnp.full((MAX_ENEMY_BULLETS,), -1.0, dtype=jnp.float32),
            "vx": jnp.zeros((MAX_ENEMY_BULLETS,), dtype=jnp.float32),
            "vy": jnp.zeros((MAX_ENEMY_BULLETS,), dtype=jnp.float32),
            "active": jnp.full((MAX_ENEMY_BULLETS,), False, dtype=jnp.bool_)
        }

        # Enemy parameters    enemy参数
        num_initial_enemies = min(MAX_ENEMIES, 5)  # 实际生成的敌人数量，不超过 MAX_ENEMIES
        enemy_w, enemy_h = 10, 10

        keys = jax.random.split(key, num_initial_enemies * 3)  # 为每个敌人生成足够的随机数

        # 重新初始化敌人数组，并只激活 num_initial_enemies 个
        new_enemies_x = jnp.full((MAX_ENEMIES,), -1.0, dtype=jnp.float32)
        new_enemies_y = jnp.full((MAX_ENEMIES,), -1.0, dtype=jnp.float32)
        new_enemies_w = jnp.full((MAX_ENEMIES,), float(enemy_w), dtype=jnp.float32)
        new_enemies_h = jnp.full((MAX_ENEMIES,), float(enemy_h), dtype=jnp.float32)
        new_enemies_vx = jnp.full((MAX_ENEMIES,), 0.0, dtype=jnp.float32)
        new_enemies_active = jnp.full((MAX_ENEMIES,), False, dtype=jnp.bool_)


        def _fill_enemies_body(i, current_enemies_array_parts):
            c_x, c_y, c_w, c_h, c_vx, c_active = current_enemies_array_parts

            # 1. x_rand and y_rand should be generated as integers first
            x_rand_int = jrandom.randint(keys[i * 3 + 0], (), 60, WINDOW_WIDTH - 80, dtype=jnp.int32)
            y_rand_int = jrandom.randint(keys[i * 3 + 1], (), 60, WINDOW_HEIGHT - 100, dtype=jnp.int32)

            # 2. Then convert them to float32 if your enemy position arrays are float32
            x_rand = x_rand_int.astype(jnp.float32)  # 或者 x_rand_int * 1.0
            y_rand = y_rand_int.astype(jnp.float32)  # 或者 y_rand_int * 1.0

            move_flag = jrandom.uniform(keys[i * 3 + 2]) < 0.5
            vx_val = jnp.where(jrandom.bernoulli(keys[i * 3 + 2]), 1.5, -1.5).astype(jnp.float32)
            final_vx = jnp.where(move_flag, vx_val, 0.0)

            # 更新当前索引 i 处的数组元素
            c_x = c_x.at[i].set(x_rand)
            c_y = c_y.at[i].set(y_rand)
            c_vx = c_vx.at[i].set(final_vx)
            c_active = c_active.at[i].set(True)  # 激活这个敌人

            return (c_x, c_y, c_w, c_h, c_vx, c_active)

        # 填充前 num_initial_enemies 个敌人
        final_enemies_x, final_enemies_y, final_enemies_w, final_enemies_h, final_enemies_vx, final_enemies_active = \
            jax.lax.fori_loop(0, num_initial_enemies, _fill_enemies_body,
                              (new_enemies_x, new_enemies_y, new_enemies_w, new_enemies_h, new_enemies_vx,
                               new_enemies_active))

        self.enemies = {
            "x": final_enemies_x,
            "y": final_enemies_y,
            "w": final_enemies_w,
            "h": final_enemies_h,
            "vx": final_enemies_vx,
            "active": final_enemies_active
        }

        # Output observation    输出observation
        obs = jnp.array([state["x"], state["y"], state["vx"], state["vy"], state["angle"]])

        return obs, state

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
        # raise NotImplementedError("Abstract method")
        # Select map    选地图
        if self.mode == "map":

            # Unpack state  解包状态
            obs, new_state, reward, done_map, info_map, reset_flag, level = step_core_map(
                state,
                action,
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                HUD_HEIGHT
            )

            x, y = float(new_state["x"]), float(new_state["y"])
            entered = False
            entered_level = -1
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
                info_map = {"entered_level": True}
            else:
                reward = -1.0
                done = False
                info_map = {}

            return obs, new_state, reward, done, info_map

        #In level mode    在level里
        obs, new_state, updated_player_bullets, updated_cooldown, \
            updated_enemies, updated_enemy_bullets, updated_enemy_fire_cooldown, \
            new_key, reward, done_level, info_level = step_core_level(
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
            self.enemy_fire_interval,
            MAX_PLAYER_BULLETS,
            MAX_ENEMY_BULLETS,
            MAX_ENEMIES
        )

        # 用 JIT 函数返回的新状态更新 self 的属性
        self.bullets = updated_player_bullets
        self.cooldown = updated_cooldown
        self.enemies = updated_enemies
        self.enemy_bullets = updated_enemy_bullets
        self.enemy_fire_cooldown = updated_enemy_fire_cooldown
        self.key = new_key

        # when the ship is dead?   死亡逻辑: 飞船被打或撞
        if done_level:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
                self.done = True  # 游戏结束
                return obs, new_state, reward, self.done, info_level
            else:
                self.key, subkey = jrandom.split(self.key)
                obs, new_state = self.reset_level(subkey, self.current_level)
                info_level["respawn"] = True
                self.done = False  # 单次死亡不算游戏结束
                return obs, new_state, reward, self.done, info_level

        # how to success?    通关逻辑: 敌人全灭
        if jnp.sum(self.enemies["active"]) == 0:  # 检查活跃敌人数量是否为0
            self.key, subkey = jrandom.split(self.key)
            obs, new_state = self.reset_map(subkey)
            reward = 10.0
            info_level["level_cleared"] = True
            self.done = False
            return obs, new_state, reward, self.done, info_level

        self.done = done_level
        return obs, new_state, reward, self.done, info_level


if __name__ == "__main__":
    env = JaxGravitar()

    # Initialize random seed    初始化随机种子
    key = jax.random.PRNGKey(0)

    # Get initial observation and state            得到初始观测和状态obs, state
    obs, state = env.reset(key)

    # loop 循环
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

        # Read keyboard        读键盘
        action = get_action_from_key()

        # Environment interaction              环境交互
        obs, state, reward, done, info = env.step(state, action)
        env.render(state)

        clock.tick(30)