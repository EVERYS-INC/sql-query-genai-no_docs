# Azure OpenAI用 サンプルデータ説明書

## データベース概要
製造業の工場データベースで、生産管理、機器管理、品質管理のデータを含みます。

## 主要テーブルのサンプルデータ

### production_lines（生産ライン）サンプル
```
line_id | line_name | location | capacity_per_hour
1       | ライン1   | A棟      | 100
2       | ライン2   | A棟      | 120
3       | ライン3   | B棟      | 80
```

### products（製品）サンプル
```
product_id | product_code | product_name        | category  | standard_time_minutes
1          | PRD-001      | 製品A（標準型）     | 電子部品  | 15.5
2          | PRD-002      | 製品A（高性能型）   | 電子部品  | 18.0
3          | PRD-003      | 製品B（標準型）     | 機械部品  | 22.0
```

### production_records（生産実績）サンプル
```
production_date | line_id | product_id | planned_quantity | actual_quantity | good_quantity | defect_quantity
2024-01-15     | 1       | 1          | 150             | 145            | 142          | 3
2024-01-15     | 2       | 3          | 100             | 105            | 101          | 4
```

### machine_operations（機器稼働）サンプル
```
machine_id | operation_date | running_minutes | stop_minutes | idle_minutes | produced_quantity
1          | 2024-01-15    | 420.5          | 35.5        | 24.0        | 280
2          | 2024-01-15    | 455.0          | 15.0        | 10.0        | 320
```

## 主要な集計指標の計算方法

### 1. 稼働率（Availability）
```sql
running_minutes / (running_minutes + stop_minutes + idle_minutes) * 100 as availability_rate
```

### 2. 品質率（Quality Rate）
```sql
good_quantity / actual_quantity * 100 as quality_rate
```

### 3. 生産効率（Performance Rate）
```sql
actual_quantity / planned_quantity * 100 as performance_rate
```

### 4. OEE（設備総合効率）
```sql
(running_minutes / 480) * (actual_quantity / planned_quantity) * (good_quantity / actual_quantity) * 100 as oee
```

## データの関連性

1. **生産ライン → 機器**: 1つのラインに複数の機器が所属
2. **生産実績 → 製品/ライン/シフト/作業員**: 生産実績は複数のマスタと関連
3. **機器 → 稼働履歴/停止履歴/メンテナンス**: 機器ごとに詳細な履歴を管理
4. **品質検査 → 製品/ライン**: 製品とラインごとの品質状況を記録

## クエリ作成時の注意点

1. **日付範囲**: データは過去90日間のみ存在
2. **シフト**: 3交代制（早番6-14時、遅番14-22時、夜勤22-6時）
3. **停止理由**: 計画停止と計画外停止を区別して集計可能
4. **品質不良**: defect_typeで不良の種類を分析可能
5. **作業員スキル**: skill_levelで作業員の熟練度を考慮した分析が可能

## よく使用される分析パターン

1. **日別/月別の生産推移**
2. **ライン別/製品別の稼働率・品質率**
3. **機器別の停止時間と理由分析**
4. **シフト別の生産性比較**
5. **不良率の推移と要因分析**
6. **メンテナンスコストと稼働率の相関**
7. **作業員スキルと品質の関係**