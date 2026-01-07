# test-spec: ToDo アプリ仕様書

## 概要
小さな ToDo アプリの仕様書（サンプル）。個人用のタスク作成・編集・完了・削除、一覧取得をサポートします。学習目的での仕様ドキュメントです。

## 目的
- ユーザーがタスクを作成、状態変更（未完了→完了）、編集、削除できること。
- シンプルな REST API を通じて CRUD 操作が可能であること。

## ユーザー/ペルソナ
- 一般ユーザー: 日々のタスクを管理する個人。

## ユーザーストーリーと受け入れ基準

1) タスクを作成できる
- ストーリー: 「ユーザーとして、新しいタスクを作成したい。タイトルと任意で説明を入れられる」
- 受け入れ基準:
  - タイトル必須、説明は任意。
  - 作成に成功すると HTTP 201 を返す。
  - レスポンスに作成されたタスクの ID、タイトル、説明、作成日時、完了フラグ（false）を含む。

2) タスク一覧を取得できる
- ストーリー: 「ユーザーとして、自分のタスク一覧を取得したい」
- 受け入れ基準:
  - GET /api/todos でタスク配列を返す。
  - デフォルトは作成日時の降順。

3) タスクを編集できる
- ストーリー: 「ユーザーとして、既存のタスクのタイトル/説明を更新したい」
- 受け入れ基準:
  - 更新成功で HTTP 200、更新後のタスクを返す。

4) タスクを完了/未完了にできる
- ストーリー: 「ユーザーとして、タスクを完了にできる」
- 受け入れ基準:
  - 完了状態変更で HTTP 200。

5) タスクを削除できる
- ストーリー: 「ユーザーとして、不要なタスクを削除したい」
- 受け入れ基準:
  - 削除成功で HTTP 204 を返す。

## API エンドポイント（仕様）

- GET /api/todos
  - 説明: タスク一覧取得
  - レスポンス: 200, [{id, title, description, created_at, completed}]

- POST /api/todos
  - 説明: タスク作成
  - リクエスト: { title: string, description?: string }
  - レスポンス: 201, {id, title, description, created_at, completed}

- GET /api/todos/:id
  - 説明: 単一タスク取得
  - レスポンス: 200 or 404

- PUT /api/todos/:id
  - 説明: タスク更新（タイトル/説明/完了フラグ）
  - リクエスト: { title?: string, description?: string, completed?: boolean }
  - レスポンス: 200, 更新後タスク

- DELETE /api/todos/:id
  - 説明: タスク削除
  - レスポンス: 204

## データモデル

- Todo
  - id: integer (auto increment)
  - title: string (not null)
  - description: text (nullable)
  - created_at: datetime
  - completed: boolean (default false)

## 非機能要件
- 小規模のため認証は省略（練習目的）。
- 単体テストを用意すること。

## 成功基準
- 上記 API を通じて CRUD が動作すること。
- 単体テストが通ること。
