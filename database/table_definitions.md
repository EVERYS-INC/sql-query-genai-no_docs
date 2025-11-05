# 製造業工場データベース テーブル定義書

## データベース概要
製造業の工場における生産管理、機器管理、品質管理のためのデータベース。
過去3ヶ月間の生産実績、機器稼働状況、品質検査結果等を記録。

## テーブル一覧

### 1. production_lines（生産ライン）
生産ラインの基本情報を管理

| カラム名 | データ型 | 説明 | 制約 |
|---------|---------|------|------|
| line_id | INTEGER | ラインID | 主キー |
| line_name | TEXT | ライン名（例：ライン1） | NOT NULL |
| location | TEXT | 設置場所（例：A棟） | NOT NULL |
| capacity_per_hour | INTEGER | 時間当たり生産能力 | NOT NULL |
| created_at | TIMESTAMP | 作成日時 | DEFAULT CURRENT_TIMESTAMP |

### 2. products（製品）
製造する製品の情報を管理

| カラム名 | データ型 | 説明 | 制約 |
|---------|---------|------|------|
| product_id | INTEGER | 製品ID | 主キー |
| product_code | TEXT | 製品コード（例：PRD-001） | UNIQUE NOT NULL |
| product_name | TEXT | 製品名（例：製品A（標準型）） | NOT NULL |
| category | TEXT | カテゴリ（電子部品/機械部品/特殊部品） | NOT NULL |
| standard_time_minutes | REAL | 標準作業時間（分） | NOT NULL |
| target_quality_rate | REAL | 目標品質率（0.92～0.99） | NOT NULL |

### 3. machines（機器）
工場内の機器・設備情報を管理

| カラム名 | データ型 | 説明 | 制約 |
|---------|---------|------|------|
| machine_id | INTEGER | 機器ID | 主キー |
| machine_code | TEXT | 機器コード（例：MCH-001） | UNIQUE NOT NULL |
| machine_name | TEXT | 機器名（例：プレス機_1-1） | NOT NULL |
| machine_type | TEXT | 機器タイプ（プレス機/成形機/組立機/検査装置/包装機） | NOT NULL |
| line_id | INTEGER | 所属ラインID | 外部キー → production_lines |
| installation_date | DATE | 設置日 | NOT NULL |
| rated_capacity | INTEGER | 定格能力 | NOT NULL |

### 4. operators（作業員）
作業員の情報を管理

| カラム名 | データ型 | 説明 | 制約 |
|---------|---------|------|------|
| operator_id | INTEGER | 作業員ID | 主キー |
| employee_code | TEXT | 社員コード（例：EMP-0001） | UNIQUE NOT NULL |
| name | TEXT | 氏名 | NOT NULL |
| skill_level | TEXT | スキルレベル（初級/中級/上級/エキスパート） | NOT NULL |
| department | TEXT | 所属部署（製造1課/製造2課/製造3課/品質管理課/保全課） | NOT NULL |
| hire_date | DATE | 入社日 | NOT NULL |

### 5. shifts（シフト）
勤務シフトの定義

| カラム名 | データ型 | 説明 | 制約 |
|---------|---------|------|------|
| shift_id | INTEGER | シフトID | 主キー |
| shift_name | TEXT | シフト名（早番/遅番/夜勤） | NOT NULL |
| start_time | TIME | 開始時刻 | NOT NULL |
| end_time | TIME | 終了時刻 | NOT NULL |

### 6. production_records（生産実績）
日々の生産実績データ

| カラム名 | データ型 | 説明 | 制約 |
|---------|---------|------|------|
| record_id | INTEGER | レコードID | 主キー |
| production_date | DATE | 生産日 | NOT NULL |
| shift_id | INTEGER | シフトID | 外部キー → shifts |
| line_id | INTEGER | ラインID | 外部キー → production_lines |
| product_id | INTEGER | 製品ID | 外部キー → products |
| planned_quantity | INTEGER | 計画数量 | NOT NULL |
| actual_quantity | INTEGER | 実績数量 | NOT NULL |
| good_quantity | INTEGER | 良品数 | NOT NULL |
| defect_quantity | INTEGER | 不良品数 | NOT NULL |
| production_time_minutes | REAL | 生産時間（分） | NOT NULL |
| operator_id | INTEGER | 作業員ID | 外部キー → operators |

### 7. machine_operations（機器稼働履歴）
機器の稼働状況を記録

| カラム名 | データ型 | 説明 | 制約 |
|---------|---------|------|------|
| operation_id | INTEGER | 稼働ID | 主キー |
| machine_id | INTEGER | 機器ID | 外部キー → machines |
| operation_date | DATE | 稼働日 | NOT NULL |
| shift_id | INTEGER | シフトID | 外部キー → shifts |
| start_time | TIMESTAMP | 開始時刻 | NOT NULL |
| end_time | TIMESTAMP | 終了時刻 | NOT NULL |
| status | TEXT | 稼働状態 | NOT NULL |
| running_minutes | REAL | 稼働時間（分） | NOT NULL |
| stop_minutes | REAL | 停止時間（分） | NOT NULL |
| idle_minutes | REAL | アイドル時間（分） | NOT NULL |
| produced_quantity | INTEGER | 生産数 | NOT NULL |

### 8. quality_inspections（品質検査）
品質検査の結果を記録

| カラム名 | データ型 | 説明 | 制約 |
|---------|---------|------|------|
| inspection_id | INTEGER | 検査ID | 主キー |
| inspection_date | DATE | 検査日 | NOT NULL |
| product_id | INTEGER | 製品ID | 外部キー → products |
| line_id | INTEGER | ラインID | 外部キー → production_lines |
| batch_number | TEXT | バッチ番号（例：BATCH-2024-01-01-1234） | NOT NULL |
| sample_size | INTEGER | サンプルサイズ | NOT NULL |
| passed_count | INTEGER | 合格数 | NOT NULL |
| failed_count | INTEGER | 不合格数 | NOT NULL |
| defect_type | TEXT | 不良タイプ（寸法不良/外観不良/機能不良/仕様外） | NULL可 |
| inspector_id | INTEGER | 検査員ID | 外部キー → operators |

### 9. maintenance_records（メンテナンス記録）
機器のメンテナンス履歴

| カラム名 | データ型 | 説明 | 制約 |
|---------|---------|------|------|
| maintenance_id | INTEGER | メンテナンスID | 主キー |
| machine_id | INTEGER | 機器ID | 外部キー → machines |
| maintenance_date | DATE | メンテナンス日 | NOT NULL |
| maintenance_type | TEXT | メンテナンスタイプ（定期点検/部品交換/清掃/調整/修理） | NOT NULL |
| description | TEXT | 作業内容詳細 | NOT NULL |
| duration_hours | REAL | 作業時間（時間） | NOT NULL |
| cost | REAL | コスト（円） | NOT NULL |
| technician_id | INTEGER | 技術者ID | 外部キー → operators |
| next_maintenance_date | DATE | 次回メンテナンス予定日 | NULL可 |

### 10. downtime_reasons（停止理由）
機器停止の理由マスタ

| カラム名 | データ型 | 説明 | 制約 |
|---------|---------|------|------|
| reason_id | INTEGER | 理由ID | 主キー |
| reason_code | TEXT | 理由コード（例：DT-001） | UNIQUE NOT NULL |
| reason_description | TEXT | 理由説明（計画停止/段取り替え等） | NOT NULL |
| category | TEXT | カテゴリ（計画/故障/外部要因/品質/その他） | NOT NULL |

### 11. machine_downtimes（機器停止履歴）
機器の停止履歴を詳細に記録

| カラム名 | データ型 | 説明 | 制約 |
|---------|---------|------|------|
| downtime_id | INTEGER | 停止ID | 主キー |
| machine_id | INTEGER | 機器ID | 外部キー → machines |
| start_time | TIMESTAMP | 停止開始時刻 | NOT NULL |
| end_time | TIMESTAMP | 停止終了時刻 | NOT NULL |
| reason_id | INTEGER | 停止理由ID | 外部キー → downtime_reasons |
| duration_minutes | REAL | 停止時間（分） | NOT NULL |
| notes | TEXT | 備考 | NULL可 |

## データの特徴

### データ期間
- 過去90日間（3ヶ月）のデータを保持

### データ規模
- 生産ライン: 5ライン
- 製品: 8種類
- 機器: 20台（各ライン4台）
- 作業員: 50名
- シフト: 3交代制（早番・遅番・夜勤）

### 重要な指標
1. **稼働率**: running_minutes / (running_minutes + stop_minutes + idle_minutes)
2. **品質率**: good_quantity / actual_quantity
3. **生産効率**: actual_quantity / planned_quantity
4. **設備総合効率(OEE)**: 稼働率 × 性能率 × 品質率