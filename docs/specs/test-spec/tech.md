# 技術仕様・注意点 (tech.md)

## 技術スタック（例）
- Python + Flask / FastAPI（API 実装）
- SQLite（開発用シンプル DB）
- pytest（テスト）

## DB スキーマ（例）

CREATE TABLE todos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  description TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  completed BOOLEAN DEFAULT 0
);

## API 契約の注意
- ステータスコードの取り扱いは spec.md の定義に従う。
- PUT は全更新ではなく、部分更新を受け取り可能にする（受け取ったフィールドのみ更新）。

## テスト
- ユニットテスト: 各エンドポイントのハッピーパス + 代表的なエラーケース (404, 400)

## 開発フロー（簡易）
1. `tasks.md` のタスクを 1 件取り、`[-]` として状態を更新する
2. 既存実装（リポジトリ）を検索して再利用できるか確認する
3. 実装 → ログ（log-implementation） → tasks.md を `[x]` にする
