import torch
import numpy as np
from recbole.utils import get_model
from recbole.data import create_dataset
from recbole.data.interaction import Interaction

# NumPy互換性エラー対策
np.long = np.int64

def recommend_with_lift(weapon_name):
    # 1. 最新のモデルファイル名に更新
    model_file = 'saved/FM-Jan-08-2026_03-48-50.pth' 
    
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # 2. データセット名が 'splatoon3_xmatch' になっていることを確認
    dataset = create_dataset(config)
    model = get_model(config['model'])(config, dataset).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    try:
        weapon_id = dataset.token2id('weapon_id', weapon_name)
    except KeyError:
        print(f"エラー: ブキ '{weapon_name}' がデータセットに見つかりません。")
        return

    ability_tokens = dataset.field2id_token['ability_id'][1:]
    num_items = len(ability_tokens)

    with torch.no_grad():
        # ターゲットブキのスコアを計算 (重み1.0固定)
        input_dict = {
            'weapon_id': torch.full((num_items,), weapon_id, dtype=torch.int64).to(device),
            'ability_id': torch.arange(1, num_items + 1, dtype=torch.int64).to(device),
            'mode': torch.zeros(num_items, dtype=torch.int64).to(device),
            'stage': torch.zeros(num_items, dtype=torch.int64).to(device),
        }
        # 数値特徴量(weight)の設定
        val = torch.full((num_items, 1), 1.0, dtype=torch.float).to(device)
        idx = torch.zeros((num_items, 1), dtype=torch.float).to(device)
        input_dict['weight'] = torch.cat([val, idx], dim=-1)
        
        target_scores = model.predict(Interaction(input_dict))

        # 3. 特化度（偏差）を出すための「全ブキ平均スコア」の計算
        # dataset.num('weapon_id') を使用
        all_weapon_ids = torch.arange(1, dataset.num('weapon_id'))
        sample_size = min(30, len(all_weapon_ids))
        avg_scores = torch.zeros(num_items).to(device)
        
        # ランダムにサンプルブキを抽出して平均を取る
        indices = torch.randperm(len(all_weapon_ids))[:sample_size]
        for idx_w in indices:
            w_id = all_weapon_ids[idx_w]
            input_dict['weapon_id'] = torch.full((num_items,), w_id, dtype=torch.int64).to(device)
            avg_scores += model.predict(Interaction(input_dict))
        avg_scores /= sample_size

        # 特化度 = ターゲットスコア - 全ブキ平均
        lift_scores = target_scores - avg_scores

    # ・・・（前半の計算部分はそのまま）・・・

    # 結果をリストに格納
    results = []
    for i, token in enumerate(ability_tokens):
        results.append({
            'name': token,
            'score': target_scores[i].item(),
            'lift': lift_scores[i].item()
        })
    
    # 1. 予測スコア順（純粋に採用率・評価が高い順）
    results_raw = sorted(results, key=lambda x: x['score'], reverse=True)

    # 2. 特化度順（他のブキとの差が大きい順）
    results_lift = sorted(results, key=lambda x: x['lift'], reverse=True)

    print(f"\n===== 🦑 【{weapon_name}】 推論結果レポート (Xマッチ限定モデル) =====")

    # --- ランキング1: 純粋な予測スコア順 ---
    print(f"\n📊 [1. 総合おすすめ順] (汎用的に評価が高いもの)")
    print("-" * 65)
    print(f"{'順位':<4} | {'ギアパワー名':<25} | {'予測スコア':<10}")
    print("-" * 65)
    for i, res in enumerate(results_raw[:10]):
        print(f"{i+1:>4} | {res['name']:<25} | {res['score']:.4f}")

    # --- ランキング2: 特化度（リフト値）順 ---
    print(f"\n✨ [2. 特化度順] (このブキならではのギア)")
    print("-" * 65)
    print(f"{'順位':<4} | {'ギアパワー名':<25} | {'特化度(偏差)':<10}")
    print("-" * 65)
    for i, res in enumerate(results_lift[:10]):
        print(f"{i+1:>4} | {res['name']:<25} | {res['lift']:+ .4f}")

    print("\n※「総合おすすめ」には、どのブキでも強い汎用ギア（ステジャン等）が出やすくなります。")
    print("※「特化度」には、そのブキの弱点を補うものや長所を伸ばすものが選ばれます。")

# ・・・（末尾の実行部分はそのまま）・・・

if __name__ == '__main__':
    # .52ガロンで検証
    recommend_with_lift('52gal')
