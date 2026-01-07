# test-spec — ToDo アプリ（サンプル）

このフォルダは MCP spec-workflow の練習用に作成された、ToDo アプリの仕様一式です。

目的:
- 小さな ToDo アプリ機能の仕様（requirements/design/tasks/tech）を示し、spec-workflow の手順を試すための自己完結ファイル群を用意する。

構成:
- `spec.md` — 仕様の全体（ユーザーストーリー、受け入れ基準、API、データモデル）
- `product.md` — 製品的背景・目的
- `tech.md` — 技術的検討事項、DBスキーマ、API契約
- `tasks.md` — 実装タスク (MCP ワークフローで編集する想定)

削除方法 (後でまとめて削除する場合):
1. リポジトリルートから次を実行してください:

```bash
# 作業コピーの変更をコミット済みであることを確認してから削除
rm -rf docs/specs/test-spec
```

2. Git 管理下で完全に消す場合:

```bash
git rm -r docs/specs/test-spec
git commit -m "remove test-spec - temporary MCP spec"
git push
```

---

このフォルダはいつでも削除可能です。練習のあとに消したければ README の手順を使ってください。
