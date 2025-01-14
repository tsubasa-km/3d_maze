import pygame as pg
from pygame.math import Vector2
from abc import ABC, abstractmethod
from typing import Type, Literal, overload
from enum import Enum, auto
from dataclasses import dataclass, field
import math
import sys
from random import randint, choice, shuffle
from copy import deepcopy
from pprint import pprint


class Config:
    FPS = 60
    MAP_SIZE = (700,)*2
    MAZE_SIZE = (20,)*2

    class Player:
        SPEED = 2
        ROTATE_SPEED = 10
        RADIUS = 5
        COLOR = (255, 255, 0)

    class Ray:
        IS_VISIBLE = True
        SHOW_DIRECTION_ONLY = True
        COLOR = (255, 255, 0)
        LENGTH = 100
        NUMBER = 100
        ANGLE = 60
        HIT_COLOR = (255, 0, 0)

    class Wall:
        COLOR = (255, 255, 255)
        IS_SOLID = False
        HEIGHT = 1  # ~4


class Tag(Enum):
    """当たり判定のタググループ"""
    PLAYER = auto()
    WALL = auto()
    RAY = auto()
    GOAL = auto()


class ColliderType(Enum):
    """当たり判定の形状"""
    CIRCLE = auto()
    LINE = auto()
    BOX = auto()


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
    parent_obj: "Obj"

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

    @abstractmethod
    def update(self):
        """毎フレームの処理"""
        pass

    @abstractmethod
    def detect_collision(self, target_tag_list: list[Tag]) -> CollisionDTO:
        """衝突を検証する"""
        pass

    def __init__(self, parent_obj: "Obj", tags: list[Tag], collider_type: ColliderType, is_temporary=False) -> None:
        self.parent_obj = parent_obj
        self.tags = tags
        self.collider_type = collider_type
        if not is_temporary:
            self._add_collider(tags)

    def __del__(self):
        for i, c in enumerate(self._colliders):
            if c is self:
                self._colliders[i]
                break

    def _add_collider(self, tags: list[Tag]):
        """自身をCollider.collidersに追加"""
        for tag in tags:
            if not Collider._colliders.get(tag):
                Collider._colliders[tag] = []
            Collider._colliders[tag].append(self)

    def get_aabb(self) -> "BoxCollider":
        """オブジェクトの外接AABBをBoxCollider形式で返す"""
        pass

    def _aabb_collision(self, other: Type["Collider"]):
        return self._collision_detection_box_and_box(
            self.get_aabb(), other.get_aabb()).is_collided

    def _collision_detection_circle_and_line(self, circle: "CircleCollider", line: "LineCollider") -> CollisionDTO:
        """円と線分の衝突検証"""
        A = Vector2(*line.start)
        B = Vector2(*line.end)
        P = circle.center

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

    def _collision_detection_circle_and_circle(self, circle1: "CircleCollider", circle2: "CircleCollider") -> CollisionDTO:
        """円と円の衝突検証"""
        C1, C2 = circle1.center, circle2.center
        r1, r2 = circle1.radius, circle2.radius
        vecC1C2 = C2 - C1
        if vecC1C2.length() < r1+r2:
            collided_target = circle1 if self is circle2 else circle2
            collided_point = vecC1C2.normalize()*vecC1C2.length()/2
            dto = CollisionDTO(True)
            dto.add_collided_target(collided_target, collided_point)
            return dto
        return CollisionDTO(False)

    def _collision_detection_circle_and_box(self, circle: "CircleCollider", box: "BoxCollider"):
        def distance_squared(point1, point2):
            return (point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2
        collided_point = None
        center_in_box_x = box.pos.x - \
            circle.radius <= circle.center.x <= box.pos.x + box.width + circle.radius
        center_in_box_y = box.pos.y - \
            circle.radius <= circle.center.y <= box.pos.y + box.height + circle.radius
        if not (center_in_box_x and center_in_box_y):
            return CollisionDTO(False)
        corners = [
            Vector2(box.pos.x, box.pos.y),
            Vector2(box.pos.x + box.width, box.pos.y),
            Vector2(box.pos.x + box.width, box.pos.y + box.height),
            Vector2(box.pos.x, box.pos.y + box.height)
        ]
        for corner in corners:
            if distance_squared(corner, circle.center) < circle.radius ** 2:
                collided_point = corner
                dto = CollisionDTO(True)
                dto.add_collided_target(circle, collided_point)
                return dto
        if box.pos.x <= circle.center.x <= box.pos.x + box.width:
            if abs(circle.center.y - box.pos.y) < circle.radius:
                collided_point = Vector2(circle.center.x, box.pos.y)
            elif abs(circle.center.y - (box.pos.y + box.height)) < circle.radius:
                collided_point = Vector2(
                    circle.center.x, box.pos.y + box.height)
        elif box.pos.y <= circle.center.y <= box.pos.y + box.height:
            if abs(circle.center.x - box.pos.x) < circle.radius:
                collided_point = Vector2(box.pos.x, circle.center.y)
            elif abs(circle.center.x - (box.pos.x + box.width)) < circle.radius:
                collided_point = Vector2(
                    box.pos.x + box.width, circle.center.y)
        if collided_point is not None:
            dto = CollisionDTO(True)
            dto.add_collided_target(circle, collided_point)
            return dto

        return CollisionDTO(False)

    def _collision_detection_line_and_box(self, line: "LineCollider", box: "BoxCollider"):
        box_lines = [
            (Vector2(box.pos.x, box.pos.y), Vector2(
                box.pos.x, box.pos.y + box.height)),  # 左辺
            (Vector2(box.pos.x + box.width, box.pos.y),
             Vector2(box.pos.x + box.width, box.pos.y + box.height)),  # 右辺
            (Vector2(box.pos.x, box.pos.y), Vector2(
                box.pos.x + box.width, box.pos.y)),  # 上辺
            (Vector2(box.pos.x, box.pos.y + box.height),
             Vector2(box.pos.x + box.width, box.pos.y + box.height))  # 下辺
        ]

        box_line_colliders = [
            LineCollider(box.parent_obj, *line, box.tags, is_temporary=True) for line in box_lines
        ]
        value = CollisionDTO(False)
        for side_line in box_line_colliders:
            if self is line:
                dto = line._collision_detection_line_and_line(side_line, line)
            else:
                dto = side_line._collision_detection_line_and_line(
                    side_line, line)
            value.join(dto)
        return value

    def _collision_detection_box_and_box(self, box1: "BoxCollider", box2: "BoxCollider"):
        if (box1.pos.x + box1.width <= box2.pos.x or box1.pos.x >= box2.pos.x + box2.width or
                box1.pos.y + box1.height <= box2.pos.y or box1.pos.y >= box2.pos.y + box2.height):
            return CollisionDTO(False)
        collided_point = Vector2(
            (max(box1.pos.x, box2.pos.x) + min(box1.pos.x +
             box1.width, box2.pos.x + box2.width)) / 2,
            (max(box1.pos.y, box2.pos.y) + min(box1.pos.y +
             box1.height, box2.pos.y + box2.height)) / 2
        )
        dto = CollisionDTO(True)
        dto.add_collided_target(box2, collided_point)
        return dto


class LineCollider(Collider):
    def __init__(self, parent_obj: "Obj", start: Vector2, end: Vector2, tags: tuple[Tag], *args, **kwargs) -> None:
        super().__init__(parent_obj, tags, ColliderType.LINE, *args, **kwargs)
        self.update(start, end)

    def update(self, start: Vector2, end: Vector2):
        self.start = start
        self.end = end

    def get_aabb(self) -> "BoxCollider":
        """線分の外接矩形をBoxColliderで取得"""
        left = min(self.start.x, self.end.x)
        top = min(self.start.y, self.end.y)
        width = abs(self.end.x - self.start.x)
        height = abs(self.end.y - self.start.y)
        return BoxCollider(self.parent_obj, Vector2(left, top), width, height, self.tags, is_temporary=True)

    def detect_collision(self, target_tag_list: list[Tag]) -> CollisionDTO:
        targets = self._get_collider_group(target_tag_list)
        value = CollisionDTO(False)
        for target in filter(lambda t: self._aabb_collision(t), targets):
            v = CollisionDTO(False)
            match (target.collider_type):
                case ColliderType.CIRCLE:
                    v = self._collision_detection_circle_and_line(
                        target, self)
                case ColliderType.LINE:
                    v = self._collision_detection_line_and_line(self, target)
                case ColliderType.BOX:
                    v = self._collision_detection_line_and_box(self, target)
            value.join(v)
        return value


class CircleCollider(Collider):
    def __init__(self, parent_obj: "Obj", center: Vector2, radius: int | float, tags: tuple[Tag], *args, **kwargs) -> None:
        super().__init__(parent_obj, tags, ColliderType.CIRCLE, *args, **kwargs)
        self.update(center, radius)

    def update(self, center: Vector2, radius: int | float):
        self.center = center
        self.radius = radius

    def get_aabb(self) -> "BoxCollider":
        """円の外接矩形をBoxColliderで取得"""
        left = self.center.x - self.radius
        top = self.center.y - self.radius
        width = self.radius * 2
        height = self.radius * 2
        return BoxCollider(self.parent_obj, Vector2(left, top), width, height, self.tags, is_temporary=True)

    def detect_collision(self, target_tag_list: list[Tag]) -> CollisionDTO:
        targets = self._get_collider_group(target_tag_list)
        value = CollisionDTO(False)
        for target in filter(lambda t: self._aabb_collision(t), targets):
            v = CollisionDTO(False)
            match (target.collider_type):
                case ColliderType.CIRCLE:
                    v = self._collision_detection_circle_and_circle(
                        target, self)
                case ColliderType.LINE:
                    v = self._collision_detection_circle_and_line(
                        self, target)
                case ColliderType.BOX:
                    v = self._collision_detection_circle_and_box(self, target)
            value.join(v)
        return value


class BoxCollider(Collider):
    def __init__(self, parent_obj: "Obj", pos: Vector2, width: int | float, height: int | float, tags: tuple[Tag], *args, **kwargs) -> None:
        super().__init__(parent_obj, tags, ColliderType.BOX, *args, **kwargs)
        self.update(pos, width, height)

    def update(self, pos: Vector2, width: int | float, height: int | float):
        self.pos = pos
        self.width = width
        self.height = height

    def get_aabb(self) -> "BoxCollider":
        return self

    def detect_collision(self, target_tag_list: list[Tag]) -> CollisionDTO:
        targets = self._get_collider_group(target_tag_list)
        value = CollisionDTO(False)
        for target in filter(lambda t: self._aabb_collision(t), targets):
            v = CollisionDTO(False)
            match (target.collider_type):
                case ColliderType.CIRCLE:
                    v = self._collision_detection_circle_and_box(
                        target, self)
                case ColliderType.LINE:
                    v = self._collision_detection_line_and_box(
                        target, self)
                case ColliderType.BOX:
                    v = self._collision_detection_box_and_box(
                        self, target)
            value.join(v)
        return value


class Obj(ABC):
    """空間に配置するオブジェクト"""
    @abstractmethod
    def __init__(self, color: tuple[int, int, int]):
        self.color = color

    @abstractmethod
    def update(self):
        """毎フレームの更新"""

    @abstractmethod
    def draw2d(self):
        """毎フレームの描画処理"""


class Player(Obj):
    def __init__(self, pos: Vector2):
        self.pos = pos
        self.prev_pos = pos.copy()
        self.radius = 5
        self.speed = 2
        self.direction = 0
        self.ray_controller = RayController(self.pos, self.direction)
        self.collider = CircleCollider(
            self, self.pos, self.radius, [Tag.PLAYER])
        super().__init__(Config.Player.COLOR)

    def add_force(self, force):
        self.prev_pos = self.pos.copy()
        self.pos += force

    def back_to(self, to: Vector2):
        """1フレーム前の位置を変更しないで移動する。"""
        self.pos += to - self.pos

    def update(self):
        self.__move()
        self.__rotate()
        self.__detect_collision()

    def draw2d(self):
        if Config.Ray.IS_VISIBLE:
            self.ray_controller.draw2d()
        pg.draw.circle(screen, self.color, self.pos, self.radius)

    def draw_fpv(self):
        """1人称視点の描画"""
        for i, ray in enumerate(self.ray_controller.rays):
            if ray.collision_dto.is_collided:
                raito = ray.length/ray.max_length
                if raito != 0:
                    height = min(
                        SCREEN_SIZE[1], SCREEN_SIZE[1]*ray.collision_dto.targets[0].collider.parent_obj.height/(10*raito))
                else:
                    height = SCREEN_SIZE[1]
                start = (MAP_SIZE[0]+i*(SCREEN_SIZE[0]-MAP_SIZE[0])/self.ray_controller.number_of_rays,
                         SCREEN_SIZE[1]/2-height/2)
                end = (start[0], SCREEN_SIZE[1]-start[1])
                color = (255*(1-raito),)*3

                pg.draw.line(screen, color, start, end, 2)

    def __rotate(self):
        mouse_origin = Vector2(MAP_SIZE[0]*1.5, SCREEN_SIZE[1]/2)
        mouse_pos = Vector2(*pg.mouse.get_pos())
        d = mouse_pos.x-mouse_origin.x
        self.direction += d/Config.Player.ROTATE_SPEED
        pg.mouse.set_pos(tuple(mouse_origin))

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
        if force.length() > 0:
            force.rotate_ip(self.direction)
            self.add_force(force.normalize()*self.speed)

    def __detect_collision(self):
        self.collider.update(self.pos, self.radius)
        # 壁をすり抜けたとき
        movement_vec = Ray(self.prev_pos, Vector2(0).angle_to(
            self.pos-self.prev_pos)+90, (self.pos-self.prev_pos).length(), [Tag.WALL]).detect_collision()
        dto = movement_vec.collision_dto
        if dto.is_collided:
            for target in dto.targets:
                self.back_to(target.point - (self.pos -
                             self.prev_pos).normalize()*self.radius)

        # 壁に当たった時
        collision_dto = self.collider.detect_collision([Tag.WALL])
        if collision_dto.is_collided:
            for target in collision_dto.targets:
                x = target.point - self.pos
                if x.length() == 0:
                    x += self.prev_pos.normalize()*0.01
                self.add_force(x.normalize()*(x.length() - self.radius))
        self.ray_controller.update(self.direction)

        # 世界の外へいかないようにする処理
        if self.pos.x < 0:
            self.pos.x = 0
        if self.pos.y < 0:
            self.pos.y = 0
        if self.pos.x > MAP_SIZE[0]:
            self.pos.x = MAP_SIZE[0]
        if self.pos.y > MAP_SIZE[1]:
            self.pos.y = MAP_SIZE[1]

        # ゴールに到達した時
        if self.collider.detect_collision([Tag.GOAL]).is_collided:
            self.__goal()

    def __goal(self):
        text = pg.font.Font(None, 100).render("GOAL", True, (255, 0, 0))
        screen.blit(text, (SCREEN_SIZE[0]//2 - text.get_width() //
                    2, SCREEN_SIZE[1]//2 - text.get_height()//2))


class Ray(Obj):
    def __init__(self, origin: Vector2, direction: int | float, length: int | float, targets_tags: tuple[str]):
        self.origin = origin
        self.direction = direction
        self.max_length = length
        self.length = length
        self.targets_tags = targets_tags
        self.collider = LineCollider(
            self, self.origin, self.get_end_pos(), [Tag.RAY])
        self.collision_dto: CollisionDTO = CollisionDTO(False)
        super().__init__(Config.Ray.COLOR)

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
        self.detect_collision()

    def detect_collision(self):
        """衝突判定"""
        self.collider.update(self.origin, self.get_end_pos())
        self.collision_dto = self.collider.detect_collision(self.targets_tags)
        for target in self.collision_dto.targets:
            new_length = (target.point - self.origin).length()
            if self.length >= new_length:  # より距離の近いターゲット
                self.length = new_length
        new_targets = []
        for target in self.collision_dto.targets:
            if self.length >= (target.point - self.origin).length():
                new_targets.append(target)
        self.collision_dto.targets = new_targets
        return self

    def draw2d(self):
        pg.draw.line(screen, self.color, self.origin, self.get_end_pos())
        for target in self.collision_dto.targets:
            pg.draw.circle(screen, Config.Ray.HIT_COLOR, target.point, 2)


class RayController:
    def __init__(self, origin: Vector2, direction: int | float):
        self.origin = origin
        self.center_direction = direction
        self.angle_deg = Config.Ray.ANGLE
        self.number_of_rays = Config.Ray.NUMBER
        self.ray_step = self.angle_deg / (self.number_of_rays-1)
        self.ray_length = Config.Ray.LENGTH
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
        if Config.Ray.SHOW_DIRECTION_ONLY:
            self.__draw_direction()
        else:
            for ray in self.rays:
                ray.draw2d()

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
            self.origin.x +
            math.cos(math.radians(self.center_direction-90))*20,
            self.origin.y +
            math.sin(math.radians(self.center_direction-90))*20,
        ), 2)


class Wall(Obj):
    @overload
    def __init__(self, pos: Vector2, width: float, height: float):
        """Create a wall with a box shape"""
    @overload
    def __init__(self, start: Vector2, end: Vector2):
        """Create a wall with a line shape"""

    def __init__(self, *args):
        self.wall_type: Literal["line", "box"] = "line" if len(
            args) == 2 else "box"
        self.height = Config.Wall.HEIGHT
        if self.wall_type == "line":
            self.start = args[0]
            self.end = args[1]
            self.w = 2
            self.collider = LineCollider(
                self, self.start, self.end, [Tag.WALL])
        else:
            self.pos = args[0]
            self.w = args[1]
            self.h = args[2]
            self.collider = BoxCollider(
                self, self.pos, self.w, self.h, [Tag.WALL])
        super().__init__(Config.Wall.COLOR)

    def update(self):
        pass

    def draw2d(self):
        match self.wall_type:
            case "line":
                pg.draw.line(screen, self.color, self.start,
                             self.end, self.w)
            case "box":
                pg.draw.rect(screen, self.color, [
                             self.pos.x, self.pos.y, self.w, self.h],
                             width=1 if Config.Wall.IS_SOLID else 0)


class Map:
    def __init__(self, matrix: list[list[Literal["#", " ", "S", "G"]]]) -> None:
        block_size = (MAP_SIZE[0]//len(matrix[0]), MAP_SIZE[1]//len(matrix))
        self.walls: list[Wall] = []
        self.start_pos = Vector2(MAP_SIZE[0]//2, MAP_SIZE[1]//2)

        def create_joined_wall(row, idx, y):
            if idx == 0 or row[idx-1] != "#":
                n = 1
                while len(row) > idx+n and row[idx+n] == "#":
                    n += 1
                self.walls.append(Wall(Vector2(block_size[0]*idx, block_size[1]*y),
                                       block_size[0]*n, block_size[1]))

        for _y, row in enumerate(matrix):
            for _x, b in enumerate(row):
                x = block_size[0] * _x
                y = block_size[1] * _y
                match b:
                    case "S":
                        self.start_pos = Vector2(
                            x+block_size[0]//2, y+block_size[1]//2)
                    case "G":
                        CircleCollider(self, Vector2(x+block_size[0]//2, y+block_size[1]//2),
                                       5, [Tag.GOAL])
                    case "#":
                        create_joined_wall(row, _x, _y)

    @classmethod
    def create_maze(cls, width: int, height: int):
        sys.setrecursionlimit(10**6)
        matrix = [["#"]*(width+1) for _ in range(height+1)]

        matrix[1][1] = " "  # 初期地点
        dx = [(1, 2), (-1, -2), (0, 0), (0, 0)]  # x軸のベクトル
        dy = [(0, 0), (0, 0), (1, 2), (-1, -2)]  # y軸のベクトル

        def make_maze(ny, nx):
            array = list(range(4))
            shuffle(array)  # ランダムに行く方向を決める
            for i in array:
                if ny+dy[i][1] < 1 or ny+dy[i][1] >= height:  # 周りの壁を越えていたらスルー
                    continue
                if nx+dx[i][1] < 1 or nx+dx[i][1] >= width:  # 周りの壁を越えていたらスルー
                    continue
                if matrix[ny+dy[i][1]][nx+dx[i][1]] == " ":  # 2つ先のマスがすでに開いていたらスルー
                    continue
                for j in range(2):  # 通路を掘る
                    matrix[ny+dy[i][j]][nx+dx[i][j]] = " "
                make_maze(ny+dy[i][1], nx+dx[i][1])  # 掘った先のところに移動
        make_maze(1, 1)
        matrix[1][1] = "S"
        matrix[-2][-2] = "G"
        return cls(matrix)

    def draw(self):
        for wall in self.walls:
            wall.draw2d()


pg.init()
pg.mouse.set_visible(False)

# 定数宣言
MAP_SIZE = Config.MAP_SIZE
SCREEN_SIZE = MAP_SIZE[0]*2, MAP_SIZE[1]
FPS = Config.FPS


# 変数宣言
screen = pg.display.set_mode(SCREEN_SIZE)
clock = pg.time.Clock()


map_2d = Map.create_maze(*Config.MAZE_SIZE)
player = Player(map_2d.start_pos.copy())


def mainloop():
    # 上視点
    screen.fill((100, 100, 100), pg.Rect(0, 0, *MAP_SIZE))
    map_2d.draw()
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
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                pg.quit()
                sys.exit()
    mainloop()
    pg.display.flip()
    clock.tick(FPS)
