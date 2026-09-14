import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

def preprocess_image(image_path):
    """
    画像を読み込み、VGG16用に前処理する。

    Args:
        image_path (str): 画像ファイルのパス

    Returns:
        np.ndarray: 前処理済みの画像データ
    """

    # 画像を読み込む
    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    # NumPy配列に変換
    x = img_to_array(image)

    # バッチ次元を追加
    x = np.expand_dims(x, axis=0)

    # VGG16用に前処理
    x = tf.keras.applications.vgg16.preprocess_input(x)

    return x


# モデルのダウンロード
model = tf.keras.applications.VGG16(
    include_top=True,# 全結合層を含める
    weights='imagenet', # ImageNetでの事前学習済みモデル
    input_tensor=None, # デフォルトの入力サイズ(224, 224, 3)
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax' # クラス分類にSoftmax関数を使用
)


# 画像の前処理
x = preprocess_image('画像名.jpg')

# 予測
predictions = model.predict(x)

print(predictions)

# 上位3件の予測結果を取得
results = tf.keras.applications.vgg16.decode_predictions(
    predictions,
    top=3
)[0]


# 結果表示
print("デコード結果")
print("ーーーーーーーーーーーーー")
print(results)
print("ーーーーーーーーーーーーー")

for (_, label, score) in results:
    print(f"{label}: {score}")
