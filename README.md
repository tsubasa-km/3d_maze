# 3D Maze

pygame で動くシンプルな 3D 風迷路ゲームです。

## クイックスタート（uv）
```bash
uv sync
uv run python 3d_maze.py
```

## クイックスタート（Docker）
```bash
docker build -t 3d-maze .
docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  3d-maze
```

## Docker Compose（ホットリロード）
`3d_maze.py` を保存すると、コンテナ内プロセスが自動で再起動します。

```bash
docker compose up --build
```

- Linux/X11 前提です。
- 初回のみ必要に応じてホスト側で `xhost +local:` を実行してください。

## Windows / WSL / macOS の差異（pygame + Docker）
- `uv` + `pygame` 自体はクロスプラットフォームで、PyPI には Windows / macOS / manylinux 向け wheel が配布されています。まずは `uv` 実行が最も安定です。
- Windows の Docker Desktop は WSL 2 バックエンド利用が前提（設定名: **Use WSL 2 based engine**）。
- WSLg は Linux GUI アプリ（X11 / Wayland）を Windows デスクトップへ統合表示できます。
- macOS は Docker Desktop が Apple Silicon / Intel 両対応ですが、Apple Silicon では Rosetta 2 が一部ツール向けに推奨されます。
- Docker Desktop のコンテナは Windows / macOS では Linux VM 経由で動くため、Linux ホストのような「そのまま X11 ソケット共有」とは挙動差が出ます（GUI 転送設定が追加で必要になりやすい）。

## 参考情報
- Docker Desktop (Windows, WSL2 backend): https://docs.docker.com/desktop/features/wsl/
- WSLg (Linux GUI apps on Windows): https://learn.microsoft.com/windows/wsl/tutorials/gui-apps
- Docker Desktop on Mac (Apple Silicon / Rosetta note): https://docs.docker.com/desktop/setup/install/mac-install/
- Docker Desktop networking (Linux host と Mac/Windows の実行形態の差): https://docs.docker.com/desktop/features/networking/
