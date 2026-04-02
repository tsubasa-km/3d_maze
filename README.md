# 3D Maze (Windows only)

このセットアップは **Windows + WSL2 + Docker Desktop + WSLg** のみを対象にしています。

## 必要環境
- Windows 11
- WSL2
- Docker Desktop（WSL2 backend 有効）
- WSLg（GUI表示）
- `uv`

## uv で実行（WSL側）
```bash
uv sync
uv run python 3d_maze.py
```

## Docker で実行（WSL側）
```bash
docker build -t 3d-maze .
docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e PULSE_SERVER=$PULSE_SERVER \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /mnt/wslg:/mnt/wslg \
  3d-maze
```

## Docker Compose（ホットリロード）
`3d_maze.py` を保存すると自動再起動します。

```bash
docker compose up --build
```

## メモ
- 本READMEは Windows 以外（macOS / Linux）はサポート対象外です。
