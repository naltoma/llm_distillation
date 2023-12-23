# LLMモデルを軽量化してみる（確率分布を教師データとして学ばせる）

＜目次＞
- <a href="#summary">サマリ</a>
- <a href="#init">はじめに</a>
- <a href="#distillation">モデル蒸留</a>
- <a href="#step1">step 1: 教師モデルの構築（LLMで文書分類タスクをファインチューニングで学習）</a>
- <a href="#step2">step 2: 教師モデルの出力（確率分布）を新たな教師データとして取得</a>
- <a href="#step3">step 3: 生徒モデルの構築（新たな教師データを元に、小さな生徒モデルで学習することで軽量なモデルを構築）</a>
- <a href="#note">Tipsというか注意点</a>

---
## <a name="summary">サマリ</a>
- 教師モデルの構築（LLMで文書分類タスクをファインチューニングで学習）。
- 教師モデルの出力（確率分布）を新たな教師データとして取得。
- 生徒モデルの構築（新たな教師データを元に、小さな生徒モデルで学習することで軽量なモデルを構築）。
- 裏テーマ：これらのコードをできるだけ ChatGPT 4 に任せる。

軽量化結果は以下の通り。なお実行環境は M1 MacBook Air, 16GB です。

| 項目 | 教師モデル | 生徒モデル | 削減度合い |
| --- | --- | --- | --- |
| パラメータ数 | 111,208,706 | 33,242,114 | 30%弱に削減 |
| 学習データ: 分類精度(実行時間sec) | 1.00 (290s) | 0.97 (25s) | 精度ほぼ同等 (約8%に削減) |
| テストデータ: 分類精度(実行時間sec) | 0.98 (97s) | 0.96 (8s) | 同上 |

- ChatGPT4への入力とコード一覧
    - Step 1: [ [prompt-step1.md](./prompt-step1.md) | [bert_finetune.ipynb](./bert_finetune.ipynb) ]
    - Step 2: [ [prompt-step2.md](./prompt-step2.md) | [make_dataset.ipynb](./make_dataset.ipynb) ]
    - Step 3: [ [prompt-step3.md](./prompt-step3.md) | [distillation.ipynb](./distillation.ipynb) ]

---
## <a name="init">はじめに</a>
深層学習はパラメータ数が大きく、学習でも推論でもそれなりに計算資源が要求されます。それを如何に軽量化するかという研究がいろいろやられているのですが、そのあたりの概要は[ディープラーニングを軽量化する「モデル圧縮」３手法](https://laboro.ai/activity/column/engineer/ディープラーニングを軽量化するモデル圧縮/)あたりを読んでみてください。

---
## <a name="distillation">モデル蒸留</a>
今回の主題はモデル蒸留と呼ばれる方法を実際にやってみたという記事です。モデル蒸留の説明もは上記リンク先の解説が分かりやすいのでがっつり省略しますが、基本的には (1) 一度大きなモデルでちゃんと学習する、(2)そのモデルの出力（確率分布）を教師データとして用意する、(3)小さなモデルで学習する、という流れで軽量なモデル構築を目指すアプローチです。

(1)が必要な時点で大変じゃんという話はあるのだけど、そういう大変な処理とは別に「推論をスマホで動くレベルにしたい」とかあるわけで、軽量なモデルがあると嬉しいわけですね。

モデル蒸留ではなく、「最初から小さなモデルで学習をすると良いのでは？」というのは半分正解半分誤りです。半分正解と答えた理由は、小さなモデルで学習する（質の高いモデルを構築する）ためには[特徴量設計](https://zero2one.jp/ai-word/numerization-of-feature-values/)が必要です。逆に言えばそれができるのであれば深層学習に頼る必要はありません。丁寧に特徴量設計すると軽量なモデルで精度の高いモデルを構築できます。これに対しその特徴量設計まで任せられるようになってきたのが深層学習でとても素晴らしいのですが、深層学習を機能させるためには「大規模なモデル＆大量のデータ」が要求されます。しくしく。

一方で、「(2)そのモデルの出力（確率分布）を教師データとして用意する」という部分は別の観点からも面白い部分があります。それは通常の学習が「アノテーションされた教師データとの完全一致を目指す」のに対し、モデル蒸留では「教師モデルの出力の模倣を目指す」という部分です。例として文書分類（政治記事 vs スポーツ記事）を考えてみると次のようになります。
- 通常の学習：文書1は「政治記事」である。文書2は「スポーツ記事である」。というように明確にカテゴリが付与されている状態を教師データとする。
- モデル蒸留における学習：文書1は「政治記事である確率が0.9、スポーツ記事である確率が0.1」である。文書2は「政治記事である確率が0.3、スポーツ記事である確率が0.7」である。というように、学習済みモデルによる出力（確率分布）そのものを教師データとする。

このように通常の学習ではアノテーションされた教師データとの完全一致を目指すのに対し、モデル蒸留では教師モデルの出力そのものの模倣を目指しています。こうすることで

教師モデルの出力は「尤もらしい学習結果」になっているであろうことを期待しているわけですが、通常の学習では0か1かという明確に区分されたカテゴリとして教師データを与えてしまっているのですがここで次のような文書が与えられた場合にはどちらのカテゴリを付与するべきでしょうか。

- 例文：[大谷選手がＭＶＰを獲得したこと等についての会見](https://www.kantei.go.jp/jp/101_kishida/actions/202111/19bura.html)より。
  > 令和３年１１月１９日、岸田総理は、総理大臣官邸で大谷翔平選手がＭＶＰを獲得したこと等について会見を行いました。

政治的な側面とスポーツ的な側面の双方が入っています。これはどちらかのカテゴリ一つに属するというよりは両方に含めるほうがより妥当かもしれません。それならそれで[マルチラベル問題](https://ibisforest.org/index.php?マルチラベル)として扱えば良いという話もあるのですが、ここで言いたいことは「学習モデルの質が十分高いならば、その出力自体がより適切なら教師データになっているのではないか」という点です。このことを利用しているのがモデル蒸留の考え方です。こうすることで「元のラベルよりも適切な出力分布を教師データ」として学習することができるようになります。

なお、元のラベルと学習モデルの出力の両方を加味して学習することもできます。今回はそうしています。

---
## <a name="step1">step 1: 教師モデルの構築（LLMで文書分類タスクをファインチューニングで学習）</a>
モデルは[cl-tohoku/bert-base-japanese-v3](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)、データは[livedoor ニュースコーパス](https://www.rondhuit.com/download.html#news%20corpus)から「独女通信」と「ITライフハック」だけを利用しました。

裏テーマである「できるだけChatGPT4にコードを書かせる」ため、[step1-prompt.md](./step1-prompt.md)でコードを作成。こちらは類似コードが大量にあったお陰でかなりスムーズに最終コード生成まで持っていけました。

ChatGPT4での想定外だったのは以下の点です。
- モデルやデータを保存する場所を ``~/`` として相対パスで指定しましたが、そのままではそう解釈してくれずそのまま利用するコードを書いてきた。（ので、絶対パスの変換するように指示を追加）
- [transformers.EarlyStoppingCallback](https://huggingface.co/docs/transformers/main_classes/callback#transformers.EarlyStoppingCallback)を使ってくれない。（ので、直接モジュール読み込みを指定）

2番目については、BERTが出始めの頃にはまだモジュール実装されてなく「BERT + EarlyStopping」をライブラリ利用してるケースがそもそも見当たらなかったから、なのかな？

- Step 1: [ [prompt-step1.md](./prompt-step1.md) | [bert_finetune.ipynb](./bert_finetune.ipynb) ]

---
## <a name="step2">step 2: 教師モデルの出力（確率分布）を新たな教師データとして取得</a>
単に学習済みモデルを読み込み、その出力を取得してデータセットとして保存するだけ。ここも特に問題なくChatGPT4でスムーズに実行できるコードを取得。

なお、ここでは「文章に対する出力（確率分布）」を求めてファイル保存しています。こうしておくと大きいモデルはメモリに読み込まない状態でこの後の学習を実行できるし、推論時間も大幅削減できます。

- Step 2: [ [prompt-step2.md](./prompt-step2.md) | [make_dataset.ipynb](./make_dataset.ipynb) ]

---
## <a name="step3">step 3: 生徒モデルの構築（新たな教師データを元に、小さな生徒モデルで学習することで軽量なモデルを構築）</a>
事前学習済みモデルとは異なり、ここではモデルそのものをどう用意するかから考える必要があります。今回は単にモデルが小さくなれば良いやということで[教師モデル](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)のエンコーダを1層にする（元は12層）ことにしました。が、これをプロンプトで用意するのが難しく、なかなか出力されるコードが安定しませんでした。そうなった主な要因は複雑なレイヤー構造を「文章」で指示するのが難しかったことと、蒸留学習する部分がどう指示しても中途半端に不整合なコード（実行できないコード）になってしまったことでした。

1つ目のレイヤー構造に関しては、試行錯誤した結果「教師モデルのレイヤー構造をそのまま設計図として示した上で【エンコーダの総数をパラメータで指定できるようにしろ】」ぐらいの指示を与えることで安定した結果に。コードもとてもシンプルなものになっています。BertConfig使って数行書くだけ。ただし embeddings についてはゼロから学習するのは非効率であったため、教師モデルの埋め込みをそのままコピーして利用しています。

2つ目の蒸留学習については試行錯誤する範囲ではどうしても実行できるコードになりませんでした。通常と異なる部分は以下のとおりです。

- (a) hard_label (元のラベル), soft_label (教師モデルの出力分布) を教師データとして設定すること。
- (b) (a) に基づいたカスタム損失関数を利用すること。
- (c) foward時に hard_label, soft_label を除外して処理すること。

このうち(a), (b)については問題なく組み込んだコードを書いてくれます。
しかし、(c) については[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)を使おうとすると、(a)で拡張した辞書データのままでは動かなくて、get_train_dataloaderやら様々な部分で「forward処理時にこのキーを除外して」という処理をあちこちで override していく必要があります。これが一箇所だけの修正ではなくあらゆる部分で関わってくるのですが、全体として整合の取れるコードになりませんでした。手動修正も試みましたが抽象化されてる部分の解読がうまくいかず、見送ることになりました。

そこで Trainer を使わず、直接ループ文を自前で書く形で書くよう指示してみました。が、これでもなかなかうまくいかない。蒸留学習させるコード例自体が少ないのかわからないのですが、「一部は指示通りに書くが、全体としては整合取れないコード」を出力し続けます。ただしこちらだと Trainer.train() を使わないのでデバッグしやすいコードを生成してくれました。ということで最終的にはこの step 3 については「大筋は実行できるがバグのあるコード」までを生成してもらい、その後は手動で書き直しました。

- Step 3: [ [prompt-step3.md](./prompt-step3.md) | [distillation.ipynb](./distillation.ipynb) ]

---
## <a name="note">Tipsというか注意点</a>
Note: コードとは独立した部分でもかなり時間を食ってしまいました。ハイパーパラメータの一つ、学習率です。ここ最近は「事前学習済みモデルをファインチューニングする」事が多かったのですが、この場合には破壊的学習（破壊的忘却）をしてしまわないように学習率を低く抑える（例えば ``1e-5`` 〜 ``1e-6`` あたりを使う）ことが多いのですが、このモデル蒸留では embeddings 以外は初期状態からの学習です。このため学習率が小さすぎると「そもそもなかなか学習が更新されないので損失も殆ど減らない＝EarlyStoppingで即停止する」状況になりがちです。このことにコードは正しそうだけどうまく学習してくれない状態が続いてしまいました。実際には初期学習率を高めに設定するだけで良かったのに（遠い目）。
