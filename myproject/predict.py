import numpy as np
# NumPyの互換性修正
np.long = np.int64 

import torch
from recbole.utils import get_model
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

# 1. モデルの読み込み
# ※saved/の中にある最新の.pthファイル名に書き換えてください
model_file = 'saved/FM-Jan-05-2026_00-14-34.pth' 
checkpoint = torch.load(model_file, weights_only=False)
config = checkpoint['config']
init_seed(config['seed'], config['reproducibility'])

# 2. データセットの準備
dataset = create_dataset(config)

# 3. モデルの構築と重みのロード
model = get_model(config['model'])(config, dataset).to(config['device'])
model.load_state_dict(checkpoint['state_dict'])
model.eval()

def recommend_for_weapon(weapon_name, topk=10):
    # field2id_token を使用するように修正
    if weapon_name not in dataset.field2id_token['weapon_id']:
        print(f"\nブキ '{weapon_name}' はデータセットに存在しません。")
        # ヒントとして登録されている名前をいくつか出す
        example_names = list(dataset.field2id_token['weapon_id'][1:6])
        print(f"データにあるブキ名の例: {example_names}")
        return
    
    weapon_id = dataset.token2id('weapon_id', weapon_name)
    
    # ギアパワー（Item）のリストを取得
    ability_tokens = dataset.field2id_token['ability_id'][1:] 
    ability_ids = torch.arange(1, len(ability_tokens) + 1).to(config['device'])
    
    # 推論用データの作成
    input_data = {
        'weapon_id': torch.full_like(ability_ids, weapon_id),
        'ability_id': ability_ids,
        'mode': torch.zeros_like(ability_ids), 
        'stage': torch.zeros_like(ability_ids)
    }
    
    from recbole.data.interaction import Interaction
    interaction = Interaction(input_data)
    
    # 予測実行（各ギアの勝率への貢献度をスコア化）
    scores = model.predict(interaction)
    topk_scores, topk_indices = torch.topk(scores, topk)
    
    print(f"\n--- {weapon_name} のおすすめギアパワー TOP{topk} ---")
    for i in range(topk):
        ability_name = ability_tokens[topk_indices[i]]
        print(f"{i+1}: {ability_name: <20} (Score: {topk_scores[i]:.4f})")

# 実行テスト
recommend_for_weapon('sshooter') # スプラシューター
recommend_for_weapon('liter4k')  # リッター4K
