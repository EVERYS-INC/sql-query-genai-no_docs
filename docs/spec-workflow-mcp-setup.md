# Spec-Workflow-MCP セットアップガイド

GitHub Copilot Agent モードで Spec-Workflow-MCP を使用し、ダッシュボードから仕様書の承認を行うためのセットアップ手順です。

## 概要

Spec-Workflow-MCP は、構造化されたスペック駆動開発のための Model Context Protocol (MCP) サーバーです。

**主な機能:**
- 構造化ワークフロー（要件 → 設計 → タスク）
- リアルタイム Web ダッシュボード
- 仕様書の承認・リビジョン管理
- 日本語を含む11言語対応

**参考:** https://github.com/Pimzino/spec-workflow-mcp

---

## 前提条件

- **VSCode** がインストールされていること
- **Node.js / npm** がインストールされていること
- **GitHub Copilot** が有効になっていること

---

## セットアップ手順

### Step 1: VSCode で Agent モードを有効化

1. VSCode の設定を開く（`Cmd + ,`）
2. 検索バーに `chat.agent.enabled` と入力
3. **Agent Mode を有効化**（チェックを入れる）

### Step 2: MCP サーバーの設定

本プロジェクトでは `.vscode/mcp.json` に設定済みです。

```json
{
  "servers": {
    "github": { "type": "http", "url": "https://api.githubcopilot.com/mcp/" },
    "spec-workflow": {
      "command": "npx",
      "args": ["-y", "@pimzino/spec-workflow-mcp@latest", "/Users/kahara33/repos/sql-query-genai-no_docs"]
    }
  }
}
```

### Step 3: MCP サーバーの起動

以下のいずれかの方法でサーバーを起動します:

**方法A: mcp.json から起動**
1. `.vscode/mcp.json` を開く
2. ファイル上部に表示される **「Start」ボタン** をクリック

**方法B: コマンドパレットから起動**
1. `Cmd + Shift + P` でコマンドパレットを開く
2. `MCP: List Servers` を実行
3. `spec-workflow` を選択して起動

### Step 4: ダッシュボードの起動

ターミナルで以下を実行:

```bash
npx -y @pimzino/spec-workflow-mcp@latest --dashboard --port 5010
```

ブラウザで **http://localhost:5010** にアクセスすると、ダッシュボードが表示されます。

### Step 5: GitHub Copilot Agent モードで使用

1. Copilot Chat を開く（タイトルバーのアイコンをクリック）
2. チャットボックスのポップアップメニューから **「Agent」** を選択
3. 左上の **ツールアイコン** をクリックして、利用可能な MCP ツールを確認
4. `spec-workflow` のツールが表示されていることを確認

---

## 使用方法

### ワークフロー

Spec-Workflow-MCP は以下の順序で仕様を作成します:

1. **要件定義 (Requirements)** - 機能要件を定義
2. **設計 (Design)** - 技術設計を作成
3. **タスク (Tasks)** - 実装タスクに分割

### ダッシュボードでの承認

1. ダッシュボード（http://localhost:5000）を開く
2. 作成された仕様書を確認
3. 承認またはリビジョンを依頼

### Agent モードでのプロンプト例

```
@spec-workflow 新しい機能の要件を作成してください
```

```
@spec-workflow 現在の仕様の状態を確認してください
```

---

## オプション: VSCode 拡張機能

VSCode Marketplace から **「Spec Workflow MCP」** 拡張機能をインストールすると、サイドバーから直接操作できます。

1. VSCode の拡張機能パネルを開く
2. `Spec Workflow MCP` を検索
3. インストール

---

## トラブルシューティング

### MCP サーバーが起動しない

- Node.js がインストールされているか確認
- `npx -v` でバージョンを確認

### ダッシュボードにアクセスできない

- ポート 5000 が使用されていないか確認
- ファイアウォール設定を確認

### ツールが表示されない

- MCP サーバーが起動しているか確認
- VSCode を再起動してみる
- `MCP: List Servers` でサーバーの状態を確認

---

## 参考リンク

- [Spec Workflow MCP - GitHub](https://github.com/Pimzino/spec-workflow-mcp)
- [Use MCP servers in VS Code](https://code.visualstudio.com/docs/copilot/chat/mcp-servers)
- [GitHub Docs - Extending Copilot Chat with MCP](https://docs.github.com/copilot/customizing-copilot/using-model-context-protocol/extending-copilot-chat-with-mcp)
