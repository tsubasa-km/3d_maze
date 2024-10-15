import pygame as pg
from pygame.math import Vector2
from abc import ABC, abstractmethod
from typing import Type
from enum import Enum, auto
from dataclasses import dataclass, field
import math
import sys


class Tag(Enum):
    """当たり判定のタググループ"""
    PLAYER = auto()
    WALL = auto()
    RAY = auto()


class ColliderType(Enum):
    """当たり判定の形状"""
    CIRCLE = auto()
    LINE = auto()


@dataclass
class CollidedTarget:
    collider: Type["Collider"] = None
    point: Vector2 = None


@dataclass
class CollisionDTO:
    """衝突判定時のデータ"""
    is_collided: bool
    targets: list[CollidedTarget] = field(default_factory=list, init=False)

    def add_collided_target(self, collided_target: Type["Collider"], collided_point: Vector2):
        if self.is_collided:
            self.targets.append(CollidedTarget(
                collided_target, collided_point))

    def join(self, other: "CollisionDTO"):
        self.is_collided |= other.is_collided
        self.targets += other.targets


class Collider(ABC):
    # タグごとにコライダーを管理
    _colliders: dict[Tag, list[Type["Collider"]]] = {}

    @classmethod
    def _get_collider_group(cls, tags: list[Tag]) -> list[type["Collider"]]:
        """タグに属するコライダーを取得"""
        value = []
        for tag in tags:
            v = cls._colliders.get(tag)
            if v == None:
                continue
            value += v
        return value

    def __init__(self, tags: list[Tag], collider_type: ColliderType) -> None:
        self.tags = tags
        self.collider_type = collider_type
        self._add_collider(tags)

    def _add_collider(self, tags: list[Tag]):
        """自身をCollider.collidersに追加"""
        for tag in tags:
            if not Collider._colliders.get(tag):
                Collider._colliders[tag] = []
            Collider._colliders[tag].append(self)

    def _collision_detection_circles_and_line(self, circle: "CircleCollider", line: "LineCollider") -> CollisionDTO:
        """円と線分の衝突検証"""
        A = Vector2(*line.start)
        B = Vector2(*line.end)
        P = Vector2(circle.center)

        lineAB = B - A
        vecAP = P-A
        vecBP = P-B

        dotAX = lineAB.dot(vecAP) / lineAB.length()
        crossPX = lineAB.cross(vecAP) / lineAB.length()

        distance = abs(crossPX)
        if dotAX < 0:
            distance = vecAP.length()
        elif dotAX > lineAB.length():
            distance = vecBP.length()

        if distance < circle.radius:
            collided_target = circle if self is line else line
            collided_point = lineAB.normalize()*dotAX + A
            dto = CollisionDTO(True)
            dto.add_collided_target(collided_target, collided_point)
            return dto
        return CollisionDTO(False)

    def _collision_detection_line_and_line(self, Line1: "LineCollider", Line2: "LineCollider") -> CollisionDTO:
        """線分と線分の衝突検証"""
        A = Line1.start
        B = Line1.end
        C = Line2.start
        D = Line2.end

        deno = (B-A).cross(D-C)
        if deno == 0:
            return CollisionDTO(False)

        s = (C-A).cross(D-C) / deno
        t = (B-A).cross(A-C) / deno
        if s >= 0 and s <= 1 and t >= 0 and t <= 1:
            collided_target = Line1 if self is Line2 else Line2
            collided_point = A + (B-A)*s
            dto = CollisionDTO(True)
            dto.add_collided_target(collided_target, collided_point)
            return dto
        return CollisionDTO(False)

    def _collision_detection_circles_and_circle(self, circle1: "CircleCollider", circle2: "CircleCollider") -> CollisionDTO:
        """円と円の衝突検証"""
        C1, C2 = circle1.center, circle2.center
        r1, r2 = circle1.radius, circle2.radius
        vecC1C2 = C2 - C1
        if vecC1C2.length() < r1+r2:
            collided_target = circle1 if self is circle2 else circle2
            collided_point = vecC1C2.normalize()*vecC1C2.length()/2
            dto = CollisionDTO(True)
            dto.add_collided_target(collided_target, collided_point)
            return
        return CollisionDTO(False)

    @abstractmethod
    def update(self):
        """毎フレームの処理"""
        pass

    @abstractmethod
    def detect_collision(self, target_tag_list: list[Tag]) -> CollisionDTO:
        """衝突を検証する"""
        pass


class LineCollider(Collider):
    def __init__(self, start: Vector2, end: Vector2, tags: tuple[Tag]) -> None:
        super().__init__(tags, ColliderType.LINE)
        self.update(start, end)

    def update(self, start: Vector2, end: Vector2):
        self.start = start
        self.end = end

    def detect_collision(self, target_tag_list: list[Tag]) -> CollisionDTO:
        targets = self._get_collider_group(target_tag_list)
        value = CollisionDTO(False)
        for target in targets:
            v = CollisionDTO(False)
            match (target.collider_type):
                case ColliderType.CIRCLE:
                    v = self._collision_detection_circles_and_line(
                        target, self)
                case ColliderType.LINE:
                    v = self._collision_detection_line_and_line(self, target)
            value.join(v)
        return value


class CircleCollider(Collider):
    def __init__(self, center: Vector2, radius: int | float, tags: tuple[Tag]) -> None:
        super().__init__(tags, ColliderType.CIRCLE)
        self.update(center, radius)

    def update(self, center: Vector2, radius: int | float):
        self.center = center
        self.radius = radius

    def detect_collision(self, target_tag_list: list[Tag]) -> CollisionDTO:
        targets = self._get_collider_group(target_tag_list)
        value = CollisionDTO(False)
        for target in targets:
            v = CollisionDTO(False)
            match (target.collider_type):
                case ColliderType.CIRCLE:
                    v = self._collision_detection_circles_and_circle(
                        target, self)
                case ColliderType.LINE:
                    v = self._collision_detection_circles_and_line(
                        self, target)
            value.join(v)
        return value


class Obj(ABC):
    """画面に表示するオブジェクト"""
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self):
        """毎フレームの更新"""

    @abstractmethod
    def draw2d(self):
        """毎フレームの描画処理"""


class Player(Obj):
    def __init__(self, pos: Vector2):
        self.pos = pos
        self.radius = 5
        self.speed = 2
        self.direction = 0
        self.ray_controller = RayController(self.pos, self.direction)
        self.collider = CircleCollider(self.pos, self.radius, [Tag.PLAYER])

    def update(self):
        self.__move()

    def draw2d(self):
        self.ray_controller.draw2d()
        pg.draw.circle(screen, (255, 255, 0), self.pos, self.radius)

    def draw_fpv(self):
        """1人称視点の描画"""
        for i, ray in enumerate(self.ray_controller.rays):
            if ray.collision_dto.is_collided:
                raito = ray.length/ray.max_length
                height = 500/(10*raito)
                start = (MAP_SIZE[0]+i*(SCREEN_SIZE[0]-MAP_SIZE[0])/self.ray_controller.number_of_rays,
                         SCREEN_SIZE[1]/2-height/2)
                end = (start[0], SCREEN_SIZE[1]-start[1])
                color = (255*(1-raito),)*3

                pg.draw.line(screen, color, start, end, 2)

    def __move(self):
        """WASD移動"""
        force = Vector2(0)
        pressed = pg.key.get_pressed()
        if pressed[pg.K_w]:
            force.y -= 1
        if pressed[pg.K_a]:
            force.x -= 1
        if pressed[pg.K_s]:
            force.y += 1
        if pressed[pg.K_d]:
            force.x += 1
        if pressed[pg.K_q]:
            self.direction -= self.speed
        if pressed[pg.K_e]:
            self.direction += self.speed
        if force.length() > 0:
            print(self.direction)
            force.rotate_ip(self.direction)
            self.pos += force.normalize()*self.speed

        self.ray_controller.update(self.direction)
        self.collider.update(self.pos, self.radius)
        collision_dto = self.collider.detect_collision([Tag.WALL])
        if collision_dto.is_collided:
            for target in collision_dto.targets:
                x = target.point - self.pos
                self.pos += x.normalize()*(x.length() - self.radius)

        # 世界の外へいかないようにする処理
        if self.pos.x < 0:
            self.pos.x = 0
        if self.pos.y < 0:
            self.pos.y = 0
        if self.pos.x > MAP_SIZE[0]:
            self.pos.x = MAP_SIZE[0]
        if self.pos.y > MAP_SIZE[1]:
            self.pos.y = MAP_SIZE[1]


class Ray(Obj):
    def __init__(self, origin: Vector2, direction: int | float, length: int | float, targets_tag: tuple[str]):
        self.origin = origin
        self.direction = direction
        self.max_length = length
        self.length = length
        self.targets_tag = targets_tag
        self.collider = LineCollider(
            self.origin, self.get_end_pos(), [Tag.RAY])
        self.collision_dto: CollisionDTO = CollisionDTO(False)

    def get_end_pos(self) -> Vector2:
        return Vector2(
            self.origin.x +
            math.cos(math.radians(self.direction-90))*self.length,
            self.origin.y +
            math.sin(math.radians(self.direction-90))*self.length
        )

    def update(self, direction: int | float, origin: Vector2 | None = None):
        self.direction = direction
        self.length = self.max_length
        if origin:
            self.origin = origin
        self.collider.update(self.origin, self.get_end_pos())
        self.collision_dto = self.collider.detect_collision([Tag.WALL])
        for i, target in enumerate(self.collision_dto.targets):
            new_length = (target.point - self.origin).length()
            if self.length > new_length:
                self.length = new_length
            else:
                del self.collision_dto.targets[i]

    def draw2d(self):
        pg.draw.line(screen, (255, 255, 0), self.origin, self.get_end_pos())
        for target in self.collision_dto.targets:
            pg.draw.circle(screen, (255, 0, 0), target.point, 2)


class RayController(Obj):
    def __init__(self, origin: Vector2, direction: int | float):
        self.origin = origin
        self.center_direction = direction
        self.angle_deg = 60
        self.number_of_rays = 200
        self.ray_step = self.angle_deg / (self.number_of_rays-1)
        self.ray_length = 200
        self.__set_direction(direction)

    def __set_direction(self, direction: int | float):
        self.center_direction = direction
        self.rays: list[Ray] = [
            Ray(self.origin,
                direction-self.angle_deg/2 + self.ray_step * i,
                self.ray_length, [Tag.WALL])
            for i in range(self.number_of_rays)
        ]

    def update(self, direction: int | float):
        self.center_direction = direction
        for i, ray in enumerate(self.rays):
            ray.update(direction-self.angle_deg/2 + self.ray_step * i)

    def draw2d(self):
        for ray in self.rays:
            ray.draw2d()
        self.__draw_direction()

    def __draw_direction(self):
        """向いている方向を表示"""
        pg.draw.line(screen, (255, 0, 0), self.origin, (
            self.origin.x +
            math.cos(math.radians(self.center_direction-self.angle_deg/2-90))*40,
            self.origin.y +
            math.sin(math.radians(self.center_direction-self.angle_deg/2-90))*40,
        ), 2)
        pg.draw.line(screen, (255, 0, 0), self.origin, (
            self.origin.x +
            math.cos(math.radians(self.center_direction+self.angle_deg/2-90))*40,
            self.origin.y +
            math.sin(math.radians(self.center_direction+self.angle_deg/2-90))*40,
        ), 2)
        pg.draw.line(screen, (255, 0, 0), self.origin, (
            self.origin.x + math.cos(math.radians(self.center_direction-90))*20,
            self.origin.y + math.sin(math.radians(self.center_direction-90))*20,
        ), 2)


class Wall(Obj):
    def __init__(self, start: Vector2, end: Vector2):
        self.start = start
        self.end = end
        self.color = (255, 255, 255)
        self.width = 2
        self.collider = LineCollider(self.start, self.end, [Tag.WALL])

    def update(self):
        pass

    def draw2d(self):
        pg.draw.line(screen, self.color, self.start, self.end, self.width)


pg.init()

# 定数宣言
MAP_SIZE = (600, 600)
SCREEN_SIZE = MAP_SIZE[0]*2, MAP_SIZE[1]
FPS = 60


# 変数宣言
screen = pg.display.set_mode(SCREEN_SIZE)
clock = pg.time.Clock()
player = Player(Vector2(MAP_SIZE[0]/2, MAP_SIZE[0]/2))

# 四角
walls = [
    Wall(Vector2(100, 100), Vector2(100, 500)),
    Wall(Vector2(100, 500), Vector2(500, 500)),
    Wall(Vector2(500, 500), Vector2(500, 100)),
    Wall(Vector2(500, 100), Vector2(100, 100)),
]

# 三角
# walls = [
#     Wall(Vector2(100, 100), Vector2(200, 500)),
#     Wall(Vector2(100, 100), Vector2(500, 200)),
#     Wall(Vector2(500, 200), Vector2(200, 500)),
#     Wall(Vector2(550, 250), Vector2(250, 550)),
# ]


def mainloop():
    # 上視点
    screen.fill((100, 100, 100), pg.Rect(0, 0, *MAP_SIZE))
    for wall in walls:
        wall.update()
        wall.draw2d()
    player.update()
    player.draw2d()
    # 一人称視点
    screen.fill((0, 0, 0), pg.Rect(
        MAP_SIZE[0], 0, SCREEN_SIZE[0], SCREEN_SIZE[1]))
    player.draw_fpv()


while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
    mainloop()
    pg.display.flip()
    clock.tick(FPS)
