import pandas as pd
import json
import os
import glob

input_path = './data/*.csv'
output_file = 'splatoon3.inter'
all_data = []

for file in glob.glob(input_path):
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        match_winner = row['win']
        for team_prefix in ['A', 'B']:
            current_team_name = 'alpha' if team_prefix == 'A' else 'bravo'
            label = 1.0 if match_winner == current_team_name else 0.0
            for i in range(1, 5):
                p = f"{team_prefix}{i}-"
                weapon_col, ability_col = f"{p}weapon", f"{p}abilities"
                if pd.isna(row[weapon_col]) or pd.isna(row[ability_col]):
                    continue
                try:
                    abilities = json.loads(row[ability_col])
                    for name, value in abilities.items():
                        # Trueなら1.0、数値(1.9等)ならそのまま
                        weight = 1.0 if isinstance(value, bool) and value else float(value)
                        if weight > 0:
                            all_data.append({
                                'weapon_id:token': row[weapon_col],
                                'ability_id:token': name, # 単数形に戻す
                                'mode:token': row['mode'],
                                'stage:token': row['stage'],
                                'weight:float': weight,   # 重みをカラムとして持たせる
                                'label:float': label
                            })
                except:
                    continue

inter_df = pd.DataFrame(all_data)
# タブ区切りで保存
inter_df.to_csv(output_file, sep='\t', index=False)
print(f"変換完了！ {len(inter_df)} 行。重み（weight）を反映しました。")
