import pygame as pg
from pygame.math import Vector2
from abc import ABC, abstractmethod
from typing import Type
from dataclasses import dataclass
import math
import sys


@dataclass
class CollisionData:
    """衝突判定時のデータ"""
    is_collided: bool
    collided_target: Type["Collider"] = None
    colilsion_point: Vector2 = None


class Collider(ABC):
    # タグごとにコライダーを管理
    _colliders: dict[str, list[Type["Collider"]]] = {}

    def _add_collider(self, tag: str):
        """自身をCollider.collidersに追加"""
        if not Collider._colliders.get(tag):
            Collider._colliders[tag] = []
        Collider._colliders[tag].append(self)

    @classmethod
    def _get_collider_group(cls, tag: str):
        """タグに属するコライダーを取得"""
        return cls._colliders.get(tag)

    def __init__(self, tag: str | tuple[str]) -> None:
        self.tag = tag
        if isinstance(tag, str):
            self._add_collider(tag)
        else:
            for t in tag:
                self._add_collider(t)

    @abstractmethod
    def update(self): ...

    @abstractmethod
    def detect_collision(self, targets_tag: list[str]) -> CollisionData:
        """衝突を検証する"""


class LineCollider(Collider):
    def __init__(self, start: Vector2, end: Vector2, tag: str | tuple[str]) -> None:
        super().__init__(tag)
        self.update(start, end)

    def update(self, start: Vector2, end: Vector2):
        self.start = start
        self.end = end

    def detect_collision(self, targets_tag: list[str]) -> CollisionData:
        return CollisionData(False)


class CircleCollider(Collider):
    def __init__(self, center: Vector2, radius: int | float, tag: str | tuple[str]) -> None:
        super().__init__(tag)
        self.update(center, radius)

    def update(self, center: Vector2, radius: int | float):
        self.center = center
        self.radius = radius

    def detect_collision(self, targets_tag: list[str]) -> CollisionData:
        return CollisionData(False)


class Player:
    def __init__(self, pos: Vector2):
        self.pos = pos
        self.radius = 5
        self.speed = 2
        self.direction = 0
        self.ray_controller = RayController(self.pos, self.direction)
        self.collider = CircleCollider(self.pos, self.radius, "player")

    def update(self):
        """毎フレームの処理"""
        self.__move()
        self.ray_controller.update(self.direction)

    def draw(self):
        """毎フレームの描画処理"""
        self.ray_controller.draw()
        # プレイヤー
        pg.draw.circle(screen, (255, 255, 0), self.pos, self.radius)

    def __move(self):
        """WASD移動"""
        vec = Vector2(0)
        pressed = pg.key.get_pressed()
        if pressed[pg.K_w]:
            vec.y -= 1
        if pressed[pg.K_a]:
            vec.x -= 1
        if pressed[pg.K_s]:
            vec.y += 1
        if pressed[pg.K_d]:
            vec.x += 1
        if pressed[pg.K_q]:
            self.direction -= self.speed
        if pressed[pg.K_e]:
            self.direction += self.speed
        if vec.length() > 0:
            self.pos += vec.normalize()*self.speed

        # 世界の外へいかないようにする処理
        if self.pos.x < 0:
            self.pos.x = 0
        if self.pos.y < 0:
            self.pos.y = 0
        if self.pos.x > MAP_SIZE[0]:
            self.pos.x = MAP_SIZE[0]
        if self.pos.y > MAP_SIZE[1]:
            self.pos.y = MAP_SIZE[1]


class Ray:
    def __init__(self, origin: Vector2, direction: int | float, length: int | float, targets_tag: tuple[str]):
        self.origin = origin
        self.direction = direction
        self.max_length = length
        self.length = length
        self.targets_tag = targets_tag
        self.collider = LineCollider(self.origin, self.get_end_pos(), "ray")

    def get_end_pos(self) -> Vector2:
        return Vector2(
            self.origin.x +
            math.cos(math.radians(self.direction))*self.length,
            self.origin.y +
            math.sin(math.radians(self.direction))*self.length
        )

    def update(self, direction: int | float, origin: Vector2 | None = None):
        self.direction = direction
        if origin:
            self.origin = origin


class RayController:
    def __init__(self, origin: Vector2, direction: int | float):
        self.origin = origin
        self.center_direction = direction
        self.angle_deg = 90
        self.number_of_rays = 100
        self.ray_step = self.angle_deg / (self.number_of_rays-1)
        self.ray_length = 200
        self.__set_direction(direction)

    def __set_direction(self, direction: int | float):
        self.center_direction = direction
        self.rays: list[Ray] = [
            Ray(self.origin,
                direction-self.angle_deg/2 + self.ray_step * i,
                self.ray_length, "wall")
            for i in range(self.number_of_rays)
        ]

    def update(self, direction: int | float):
        self.center_direction = direction
        for i, ray in enumerate(self.rays):
            ray.update(direction-self.angle_deg/2 + self.ray_step * i)

    def draw(self):
        for ray in self.rays:
            pg.draw.line(screen, (255, 255, 0), self.origin, ray.get_end_pos())
        self.__draw_direction()

    def __draw_direction(self):
        """向いている方向を表示"""
        pg.draw.line(screen, (255, 0, 0), self.origin, (
            self.origin.x +
            math.cos(math.radians(self.center_direction-self.angle_deg/2))*40,
            self.origin.y +
            math.sin(math.radians(self.center_direction-self.angle_deg/2))*40,
        ), 2)
        pg.draw.line(screen, (255, 0, 0), self.origin, (
            self.origin.x +
            math.cos(math.radians(self.center_direction+self.angle_deg/2))*40,
            self.origin.y +
            math.sin(math.radians(self.center_direction+self.angle_deg/2))*40,
        ), 2)
        pg.draw.line(screen, (255, 0, 0), self.origin, (
            self.origin.x + math.cos(math.radians(self.center_direction))*20,
            self.origin.y + math.sin(math.radians(self.center_direction))*20,
        ), 2)


class Wall:
    def __init__(self, start: Vector2, end: Vector2):
        self.collider = LineCollider()


pg.init()

# 定数宣言
MAP_SIZE = (600, 600)
SCREEN_SIZE = MAP_SIZE[0]*2, MAP_SIZE[1]
FPS = 60


# 変数宣言
screen = pg.display.set_mode(SCREEN_SIZE)
clock = pg.time.Clock()
player = Player(Vector2(MAP_SIZE[0]/2, MAP_SIZE[0]/2))


def mainloop():
    player.update()
    # 上視点
    screen.fill((100, 100, 100), pg.Rect(0, 0, *MAP_SIZE))
    pg.draw.line(screen, (255, 255, 255), (100, 100), (200, 500))
    pg.draw.line(screen, (255, 255, 255), (100, 100), (500, 200))
    pg.draw.line(screen, (255, 255, 255), (500, 200), (200, 500))
    player.draw()
    # 一人称視点
    screen.fill((200, 200, 200), pg.Rect(
        MAP_SIZE[0], 0, SCREEN_SIZE[0], SCREEN_SIZE[1]))


while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
    mainloop()
    pg.display.flip()
    clock.tick(FPS)
