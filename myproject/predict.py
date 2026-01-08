import torch
import numpy as np
from recbole.utils import get_model
from recbole.data import create_dataset
from recbole.data.interaction import Interaction

# NumPy互換性エラー対策
np.long = np.int64

def recommend_comprehensive(weapon_name, mode_name, stage_name):
    # 1. モデルファイルの指定
    model_file = 'saved/FM-Jan-08-2026_03-48-50.pth' 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # 2. データセット構築
    dataset = create_dataset(config)
    model = get_model(config['model'])(config, dataset).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # ID変換
    try:
        w_id = dataset.token2id('weapon_id', weapon_name)
        m_id = dataset.token2id('mode', mode_name)
        s_id = dataset.token2id('stage', stage_name)
    except KeyError as e:
        print(f"エラー: 指定された名前 {e} が見つかりません。")
        return

    ability_tokens = dataset.field2id_token['ability_id'][1:]
    num_items = len(ability_tokens)

    with torch.no_grad():
        # --- ターゲットブキのスコア計算 ---
        input_dict = {
            'weapon_id': torch.full((num_items,), w_id, dtype=torch.int64).to(device),
            'ability_id': torch.arange(1, num_items + 1, dtype=torch.int64).to(device),
            'mode': torch.full((num_items,), m_id, dtype=torch.int64).to(device),
            'stage': torch.full((num_items,), s_id, dtype=torch.int64).to(device),
        }
        val = torch.full((num_items, 1), 1.0, dtype=torch.float).to(device)
        idx = torch.zeros((num_items, 1), dtype=torch.float).to(device)
        input_dict['weight'] = torch.cat([val, idx], dim=-1)
        
        target_scores = model.predict(Interaction(input_dict))

        # --- 特化度(偏差)のための平均スコア計算 ---
        avg_scores = torch.zeros(num_items).to(device)
        all_weapon_ids = torch.arange(1, dataset.num('weapon_id'))
        sample_size = min(30, len(all_weapon_ids))
        indices = torch.randperm(len(all_weapon_ids))[:sample_size]
        
        for idx_w in indices:
            input_dict['weapon_id'] = torch.full((num_items,), all_weapon_ids[idx_w], dtype=torch.int64).to(device)
            avg_scores += model.predict(Interaction(input_dict))
        avg_scores /= sample_size

        lift_scores = target_scores - avg_scores

    # 3. データの整理と順位付け
    results = []
    for i, token in enumerate(ability_tokens):
        results.append({
            'name': token,
            'score': target_scores[i].item(),
            'lift': lift_scores[i].item()
        })
    
    # 総合順位（スコアのみの順位）を先に計算
    results.sort(key=lambda x: x['score'], reverse=True)
    for rank, item in enumerate(results):
        item['raw_rank'] = rank + 1

    # 特化度（偏差）順に最終ソート
    results.sort(key=lambda x: x['lift'], reverse=True)

    # 4. 表示（紙面節約・統合テーブル）
    print(f"\n===== 🦑 【{weapon_name}】特化度解析 (ルール:{mode_name} / ステージ:{stage_name}) =====")
    print("-" * 88)
    # ヘッダー：特化度順位、名前、特化度、予測スコア、(総合順位)
    print(f"{'順位':<4} | {'ギアパワー名':<25} | {'特化度(偏差)':<12} | {'予測スコア':<10} | {'(総合順位)'}")
    print("-" * 88)
    for i, res in enumerate(results[:15]): # 上位15件を表示
        print(f"{i+1:>4} | {res['name']:<25} | {res['lift']:+11.4f} | {res['score']:<10.4f} | {res['raw_rank']:>2}位")
    print("-" * 88)
    print("※ 順位：特化度(偏差)が高い順")
    print("※ (総合順位)：特化度を考慮しない、純粋な予測スコアのみの順位")

if __name__ == '__main__':
    # 試したいブキ・ルール・ステージを指定
    recommend_comprehensive('52gal', 'area', 'yunohana')
