これで事前準備が終わりました。finetuned_bert_japaneseから取得したボキャブラリ数、埋め込みベクトルの次元数、train, val, testに対する教師モデルの出力を用いてモデル蒸留を行うコードを書いてください。
## 生徒モデル
- M1チップのmacOS上で実行するものとします。cudaを指定しないでください。
- deviceを 'cpu' と指定してください。
- レイヤー構成を教師モデルと同様にするため、以下のように設定してください。ただしエンコーダは1層とします。

```
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(32768, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
```

- encoderの総数はパラメータで指定できるようにしてください。
- encoderを1層で用意してください。
- 学習率, warmup_steps, weight_decayをモデル蒸留で利用される代表的な値で設定してください。
- モデルのパラメータ数を出力してください。
- 教師モデルの出力（確率分布）を学習するために、KLダイバージェンスとクロスエントロピー誤差の加重和を損失関数として設定してください。
  - 教師モデルの出力を soft_labels とし、元の教師ラベルを hard_labels としてください。
  - ハイパーパラメータalphaでトレードオフを調整するものとし、デフォルトでalpha=0.5としてください。
  - 損失を求める関数 compute_lossは、Transformers.Trainerを継承したサブクラスCustomTrainerを作成し、その中で実装してください。
  - CustomTrainerクラスでget_train_dataloader関数をオーバーライドし、train_datasetを参照できるようにしてください。これはsoft_labelsという特殊なキーを含んでいるための措置です。具体的には次の実装を加えてください。
    def get_train_dataloader(self):
  	    return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size)
    - get_eval_datalodar, get_test_dataloder についても同様にオーバーライドして対応してください。

## モデル蒸留による生徒モデルの学習
- trainer.train() は使用しないでください。
- 教師モデルは model.save_pretrained("./finetuned_bert_japanese") として保存されているものを読み込んでください。
- 学習データを教師モデルに入力した際の出力を生徒モデルの教師データとしてください。
- 学習率などはモデル蒸留で利用される代表的なパラメータを設定してください。
- 最大エポック数は20とします。
- EarlyStoppingによりvalの損失をモニタリングしつつ、3回改善されなかった場合には学習を停止してください。このために ``from transformers import EarlyStoppingCallback`` を利用してください。
- train, valに対するエポック毎の損失推移を描画してください。
- 最終モデルを distilled_model という名前で保存し、train, val, testそれぞれに対する分類精度を出力してください。またそれぞれの推論に要した実行時間を出力してください。
- trainer.train() は使用しないでください。