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
    def _get_collider(cls,tag:str):
        return cls._colliders[tag]

    @abstractmethod
    def __init__(self) -> None: ...

    @abstractmethod
    def detect_collision(self) -> CollisionData: ...


class LineCollider(Collider):
    def __init__(self, start, end, tag: str) -> None:
        self.start = start
        self.end = end
        self.tag = tag
        self._add_collider(tag)

    def detect_collision(self, targets_tag: list[str]) -> CollisionData:
        return CollisionData()


class CircleCollider(Collider):
    def __init__(self) -> None:
        pass

    def detect_collision(self) -> CollisionData:
        pass


class Player:
    def __init__(self, pos: Vector2):
        self.pos = pos
        self.radius = 5
        self.speed = 2
        self.direction = 0
        self.ray_controller = RayController(self.pos, self.direction)
        self.collider = CircleCollider()

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
        self.collider = LineCollider()

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
                self.ray_length)
            for i in range(self.number_of_rays)
        ]

    def update(self, direction: int | float):
        self.center_direction = direction
        for i, ray in enumerate(self.rays):
            ray.update(direction-self.angle_deg/2 + self.ray_step * i)

    def draw(self):
        for ray in self.rays:
            pg.draw.line(screen, (255, 255, 0), self.origin, (
                self.origin.x +
                math.cos(math.radians(ray.direction))*ray.length,
                self.origin.y +
                math.sin(math.radians(ray.direction))*ray.length,
            ))
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
