import sqlite3
import random
from datetime import datetime, timedelta
import os

def create_database():
    db_path = os.path.join(os.path.dirname(__file__), 'factory_data.db')
    
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 生産ライン
    cursor.execute('''
    CREATE TABLE production_lines (
        line_id INTEGER PRIMARY KEY,
        line_name TEXT NOT NULL,
        location TEXT NOT NULL,
        capacity_per_hour INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 製品マスタ
    cursor.execute('''
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        product_code TEXT UNIQUE NOT NULL,
        product_name TEXT NOT NULL,
        category TEXT NOT NULL,
        standard_time_minutes REAL NOT NULL,
        target_quality_rate REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 機器マスタ
    cursor.execute('''
    CREATE TABLE machines (
        machine_id INTEGER PRIMARY KEY,
        machine_code TEXT UNIQUE NOT NULL,
        machine_name TEXT NOT NULL,
        machine_type TEXT NOT NULL,
        line_id INTEGER NOT NULL,
        installation_date DATE NOT NULL,
        rated_capacity INTEGER NOT NULL,
        FOREIGN KEY (line_id) REFERENCES production_lines(line_id)
    )
    ''')
    
    # 作業員マスタ
    cursor.execute('''
    CREATE TABLE operators (
        operator_id INTEGER PRIMARY KEY,
        employee_code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        skill_level TEXT NOT NULL,
        department TEXT NOT NULL,
        hire_date DATE NOT NULL
    )
    ''')
    
    # シフトマスタ
    cursor.execute('''
    CREATE TABLE shifts (
        shift_id INTEGER PRIMARY KEY,
        shift_name TEXT NOT NULL,
        start_time TIME NOT NULL,
        end_time TIME NOT NULL
    )
    ''')
    
    # 生産実績
    cursor.execute('''
    CREATE TABLE production_records (
        record_id INTEGER PRIMARY KEY,
        production_date DATE NOT NULL,
        shift_id INTEGER NOT NULL,
        line_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        planned_quantity INTEGER NOT NULL,
        actual_quantity INTEGER NOT NULL,
        good_quantity INTEGER NOT NULL,
        defect_quantity INTEGER NOT NULL,
        production_time_minutes REAL NOT NULL,
        operator_id INTEGER NOT NULL,
        FOREIGN KEY (shift_id) REFERENCES shifts(shift_id),
        FOREIGN KEY (line_id) REFERENCES production_lines(line_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id),
        FOREIGN KEY (operator_id) REFERENCES operators(operator_id)
    )
    ''')
    
    # 機器稼働履歴
    cursor.execute('''
    CREATE TABLE machine_operations (
        operation_id INTEGER PRIMARY KEY,
        machine_id INTEGER NOT NULL,
        operation_date DATE NOT NULL,
        shift_id INTEGER NOT NULL,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP NOT NULL,
        status TEXT NOT NULL,
        running_minutes REAL NOT NULL,
        stop_minutes REAL NOT NULL,
        idle_minutes REAL NOT NULL,
        produced_quantity INTEGER NOT NULL,
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id),
        FOREIGN KEY (shift_id) REFERENCES shifts(shift_id)
    )
    ''')
    
    # 品質検査記録
    cursor.execute('''
    CREATE TABLE quality_inspections (
        inspection_id INTEGER PRIMARY KEY,
        inspection_date DATE NOT NULL,
        product_id INTEGER NOT NULL,
        line_id INTEGER NOT NULL,
        batch_number TEXT NOT NULL,
        sample_size INTEGER NOT NULL,
        passed_count INTEGER NOT NULL,
        failed_count INTEGER NOT NULL,
        defect_type TEXT,
        inspector_id INTEGER NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products(product_id),
        FOREIGN KEY (line_id) REFERENCES production_lines(line_id),
        FOREIGN KEY (inspector_id) REFERENCES operators(operator_id)
    )
    ''')
    
    # メンテナンス記録
    cursor.execute('''
    CREATE TABLE maintenance_records (
        maintenance_id INTEGER PRIMARY KEY,
        machine_id INTEGER NOT NULL,
        maintenance_date DATE NOT NULL,
        maintenance_type TEXT NOT NULL,
        description TEXT NOT NULL,
        duration_hours REAL NOT NULL,
        cost REAL NOT NULL,
        technician_id INTEGER NOT NULL,
        next_maintenance_date DATE,
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id),
        FOREIGN KEY (technician_id) REFERENCES operators(operator_id)
    )
    ''')
    
    # 停止理由マスタ
    cursor.execute('''
    CREATE TABLE downtime_reasons (
        reason_id INTEGER PRIMARY KEY,
        reason_code TEXT UNIQUE NOT NULL,
        reason_description TEXT NOT NULL,
        category TEXT NOT NULL
    )
    ''')
    
    # 機器停止履歴
    cursor.execute('''
    CREATE TABLE machine_downtimes (
        downtime_id INTEGER PRIMARY KEY,
        machine_id INTEGER NOT NULL,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP NOT NULL,
        reason_id INTEGER NOT NULL,
        duration_minutes REAL NOT NULL,
        notes TEXT,
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id),
        FOREIGN KEY (reason_id) REFERENCES downtime_reasons(reason_id)
    )
    ''')
    
    conn.commit()
    return conn

def insert_master_data(conn):
    cursor = conn.cursor()
    
    # 生産ライン
    production_lines = [
        (1, 'ライン1', 'A棟', 100),
        (2, 'ライン2', 'A棟', 120),
        (3, 'ライン3', 'B棟', 80),
        (4, 'ライン4', 'B棟', 150),
        (5, 'ライン5', 'C棟', 200)
    ]
    cursor.executemany('INSERT INTO production_lines (line_id, line_name, location, capacity_per_hour) VALUES (?, ?, ?, ?)', production_lines)
    
    # 製品マスタ
    products = [
        (1, 'PRD-001', '製品A（標準型）', '電子部品', 15.5, 0.98),
        (2, 'PRD-002', '製品A（高性能型）', '電子部品', 18.0, 0.97),
        (3, 'PRD-003', '製品B（標準型）', '機械部品', 22.0, 0.96),
        (4, 'PRD-004', '製品B（特殊型）', '機械部品', 25.5, 0.95),
        (5, 'PRD-005', '製品C（小型）', '電子部品', 12.0, 0.99),
        (6, 'PRD-006', '製品C（大型）', '電子部品', 20.0, 0.97),
        (7, 'PRD-007', '製品D', '機械部品', 30.0, 0.94),
        (8, 'PRD-008', '製品E', '特殊部品', 45.0, 0.92)
    ]
    cursor.executemany('INSERT INTO products (product_id, product_code, product_name, category, standard_time_minutes, target_quality_rate) VALUES (?, ?, ?, ?, ?, ?)', products)
    
    # シフトマスタ
    shifts = [
        (1, '早番', '06:00', '14:00'),
        (2, '遅番', '14:00', '22:00'),
        (3, '夜勤', '22:00', '06:00')
    ]
    cursor.executemany('INSERT INTO shifts (shift_id, shift_name, start_time, end_time) VALUES (?, ?, ?, ?)', shifts)
    
    # 機器マスタ
    machines = []
    machine_types = ['プレス機', '成形機', '組立機', '検査装置', '包装機']
    for line_id in range(1, 6):
        for i in range(4):
            machine_id = (line_id - 1) * 4 + i + 1
            machine_code = f'MCH-{machine_id:03d}'
            machine_name = f'{machine_types[i]}_{line_id}-{i+1}'
            machine_type = machine_types[i]
            installation_date = (datetime.now() - timedelta(days=random.randint(365, 1825))).strftime('%Y-%m-%d')
            rated_capacity = random.randint(80, 150)
            machines.append((machine_id, machine_code, machine_name, machine_type, line_id, installation_date, rated_capacity))
    
    cursor.executemany('INSERT INTO machines (machine_id, machine_code, machine_name, machine_type, line_id, installation_date, rated_capacity) VALUES (?, ?, ?, ?, ?, ?, ?)', machines)
    
    # 作業員マスタ
    operators = []
    skill_levels = ['初級', '中級', '上級', 'エキスパート']
    departments = ['製造1課', '製造2課', '製造3課', '品質管理課', '保全課']
    
    for i in range(1, 51):
        employee_code = f'EMP-{i:04d}'
        name = f'作業員{i}'
        skill_level = random.choice(skill_levels)
        department = random.choice(departments)
        hire_date = (datetime.now() - timedelta(days=random.randint(180, 3650))).strftime('%Y-%m-%d')
        operators.append((i, employee_code, name, skill_level, department, hire_date))
    
    cursor.executemany('INSERT INTO operators (operator_id, employee_code, name, skill_level, department, hire_date) VALUES (?, ?, ?, ?, ?, ?)', operators)
    
    # 停止理由マスタ
    downtime_reasons = [
        (1, 'DT-001', '計画停止', '計画'),
        (2, 'DT-002', '段取り替え', '計画'),
        (3, 'DT-003', '定期メンテナンス', '計画'),
        (4, 'DT-004', '機械故障', '故障'),
        (5, 'DT-005', '電気系統故障', '故障'),
        (6, 'DT-006', '材料待ち', '外部要因'),
        (7, 'DT-007', '品質不良対応', '品質'),
        (8, 'DT-008', '作業員不在', '外部要因'),
        (9, 'DT-009', 'その他', 'その他')
    ]
    cursor.executemany('INSERT INTO downtime_reasons (reason_id, reason_code, reason_description, category) VALUES (?, ?, ?, ?)', downtime_reasons)
    
    conn.commit()

def insert_transaction_data(conn):
    cursor = conn.cursor()
    
    # 過去3ヶ月分のデータを生成
    start_date = datetime.now() - timedelta(days=90)
    
    for day_offset in range(90):
        current_date = start_date + timedelta(days=day_offset)
        date_str = current_date.strftime('%Y-%m-%d')
        
        # 各シフトのデータを生成
        for shift_id in range(1, 4):
            # 各ラインの生産実績
            for line_id in range(1, 6):
                # 1シフトで2-4製品を生産
                num_products = random.randint(2, 4)
                selected_products = random.sample(range(1, 9), num_products)
                
                for product_id in selected_products:
                    planned_quantity = random.randint(80, 200)
                    efficiency = random.uniform(0.85, 1.05)
                    actual_quantity = int(planned_quantity * efficiency)
                    quality_rate = random.uniform(0.92, 0.99)
                    good_quantity = int(actual_quantity * quality_rate)
                    defect_quantity = actual_quantity - good_quantity
                    production_time = random.uniform(420, 480)
                    operator_id = random.randint(1, 50)
                    
                    cursor.execute('''
                    INSERT INTO production_records (production_date, shift_id, line_id, product_id, 
                                                   planned_quantity, actual_quantity, good_quantity, 
                                                   defect_quantity, production_time_minutes, operator_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (date_str, shift_id, line_id, product_id, planned_quantity, actual_quantity,
                          good_quantity, defect_quantity, production_time, operator_id))
            
            # 機器稼働履歴
            for machine_id in range(1, 21):
                shift_start = datetime.strptime(f"{date_str} {['06:00', '14:00', '22:00'][shift_id-1]}", '%Y-%m-%d %H:%M')
                shift_end = shift_start + timedelta(hours=8)
                
                # 稼働率を計算
                availability = random.uniform(0.75, 0.95)
                running_minutes = 480 * availability
                stop_minutes = 480 * (1 - availability) * random.uniform(0.3, 0.7)
                idle_minutes = 480 - running_minutes - stop_minutes
                produced_quantity = int(running_minutes * random.uniform(3, 8))
                
                cursor.execute('''
                INSERT INTO machine_operations (machine_id, operation_date, shift_id, start_time, 
                                               end_time, status, running_minutes, stop_minutes, 
                                               idle_minutes, produced_quantity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (machine_id, date_str, shift_id, shift_start, shift_end, '正常',
                      running_minutes, stop_minutes, idle_minutes, produced_quantity))
                
                # 機器停止履歴（停止時間がある場合）
                if stop_minutes > 30:
                    num_stops = random.randint(1, 3)
                    for _ in range(num_stops):
                        stop_start = shift_start + timedelta(minutes=random.randint(0, 400))
                        duration = random.randint(10, int(stop_minutes/num_stops))
                        stop_end = stop_start + timedelta(minutes=duration)
                        reason_id = random.randint(1, 9)
                        
                        cursor.execute('''
                        INSERT INTO machine_downtimes (machine_id, start_time, end_time, reason_id, 
                                                      duration_minutes, notes)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ''', (machine_id, stop_start, stop_end, reason_id, duration, None))
        
        # 品質検査記録（1日5-10件）
        num_inspections = random.randint(5, 10)
        for _ in range(num_inspections):
            product_id = random.randint(1, 8)
            line_id = random.randint(1, 5)
            batch_number = f'BATCH-{date_str}-{random.randint(1000, 9999)}'
            sample_size = random.randint(50, 200)
            quality_rate = random.uniform(0.90, 0.99)
            passed_count = int(sample_size * quality_rate)
            failed_count = sample_size - passed_count
            defect_types = ['寸法不良', '外観不良', '機能不良', '仕様外', None]
            defect_type = random.choice(defect_types) if failed_count > 0 else None
            inspector_id = random.randint(1, 50)
            
            cursor.execute('''
            INSERT INTO quality_inspections (inspection_date, product_id, line_id, batch_number, 
                                           sample_size, passed_count, failed_count, defect_type, 
                                           inspector_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date_str, product_id, line_id, batch_number, sample_size, passed_count,
                  failed_count, defect_type, inspector_id))
    
    # メンテナンス記録
    for machine_id in range(1, 21):
        # 各機器に2-5件のメンテナンス記録
        num_maintenance = random.randint(2, 5)
        for i in range(num_maintenance):
            maintenance_date = (start_date + timedelta(days=random.randint(0, 89))).strftime('%Y-%m-%d')
            maintenance_types = ['定期点検', '部品交換', '清掃', '調整', '修理']
            maintenance_type = random.choice(maintenance_types)
            descriptions = {
                '定期点検': '月次定期点検実施',
                '部品交換': 'ベアリング交換',
                '清掃': 'フィルター清掃および内部清掃',
                '調整': '位置調整および校正',
                '修理': '異常音対応のため部品交換'
            }
            description = descriptions[maintenance_type]
            duration_hours = random.uniform(0.5, 4.0)
            cost = random.uniform(5000, 50000)
            technician_id = random.randint(40, 50)
            next_maintenance_date = (datetime.strptime(maintenance_date, '%Y-%m-%d') + 
                                    timedelta(days=random.randint(30, 90))).strftime('%Y-%m-%d')
            
            cursor.execute('''
            INSERT INTO maintenance_records (machine_id, maintenance_date, maintenance_type, 
                                           description, duration_hours, cost, technician_id, 
                                           next_maintenance_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (machine_id, maintenance_date, maintenance_type, description, duration_hours,
                  cost, technician_id, next_maintenance_date))
    
    conn.commit()

if __name__ == '__main__':
    # データベースディレクトリの作成
    os.makedirs('database', exist_ok=True)
    
    print("製造業向けサンプルデータベースを作成中...")
    
    # データベース作成
    conn = create_database()
    print("テーブル作成完了")
    
    # マスタデータ投入
    insert_master_data(conn)
    print("マスタデータ投入完了")
    
    # トランザクションデータ投入
    insert_transaction_data(conn)
    print("トランザクションデータ投入完了")
    
    conn.close()
    print("\nデータベース作成が完了しました: database/factory_data.db")