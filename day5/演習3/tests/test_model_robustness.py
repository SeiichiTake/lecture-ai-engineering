import os
import pytest
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline

# テスト用データとモデルのパス
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/titanic_model.pkl")


@pytest.fixture
def model():
    """モデルを読み込む"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def sample_data():
    """テストデータを読み込む"""
    if not os.path.exists(DATA_PATH):
        pytest.skip("データファイルが存在しないためスキップします")
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def get_required_features(model):
    """モデルが必要とする特徴量を取得する"""
    # モデルで使われている特徴量名を確認
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "steps") and hasattr(model.steps[0][1], "feature_names_in_"):
        return list(model.steps[0][1].feature_names_in_)

    # デフォルトの特徴量リストを返す
    return ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]


def test_model_handles_missing_values(model, sample_data, get_required_features):
    """
    モデルが欠損値を適切に処理できるかテスト

    理由: 実運用環境では欠損値を含むデータが入力される可能性があり、
    モデルがエラーなく処理できることを確認する必要がある
    """
    # テスト用データを準備
    X = sample_data.copy()
    if "Survived" in X.columns:
        X = X.drop("Survived", axis=1)

    # 必要な特徴量のみを選択
    required_features = get_required_features

    # 必要な特徴量を確保（データに含まれていない場合はスキップ）
    missing_cols = [col for col in required_features if col not in X.columns]
    if missing_cols:
        pytest.skip(f"必要な特徴量がデータに存在しません: {missing_cols}")

    X = X[required_features]

    # ランダムに欠損値を挿入
    for col in X.columns:
        mask = np.random.choice([True, False], size=X.shape[0], p=[0.1, 0.9])
        X.loc[mask, col] = np.nan

    try:
        # モデルが例外を発生させずに予測できるかテスト
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray), (
            "予測結果は numpy 配列である必要があります"
        )
        assert len(predictions) == len(X), (
            "予測結果の長さはデータと一致する必要があります"
        )
    except Exception as e:
        pytest.fail(f"欠損値のあるデータに対して予測中にエラーが発生: {str(e)}")


def test_model_handles_outliers(model, sample_data, get_required_features):
    """
    モデルが異常値を適切に処理できるかテスト

    理由: 実運用環境では異常値を含むデータが入力される可能性があり、
    モデルがそのような極端な値に対してもロバストであるべき
    """
    # テスト用データを準備
    X = sample_data.copy()
    if "Survived" in X.columns:
        X = X.drop("Survived", axis=1)

    # 必要な特徴量のみを選択
    required_features = get_required_features

    # 必要な特徴量を確保
    missing_cols = [col for col in required_features if col not in X.columns]
    if missing_cols:
        pytest.skip(f"必要な特徴量がデータに存在しません: {missing_cols}")

    X = X[required_features]

    # Age に極端な値を挿入
    if "Age" in X.columns:
        X.loc[0, "Age"] = 120  # 非常に高齢

    # Fare に極端な値を挿入
    if "Fare" in X.columns:
        X.loc[1, "Fare"] = 10000  # 非常に高額な運賃

    try:
        # モデルが例外を発生させずに予測できるかテスト
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray), (
            "予測結果は numpy 配列である必要があります"
        )
    except Exception as e:
        pytest.fail(f"異常値のあるデータに対して予測中にエラーが発生: {str(e)}")


def test_model_handles_categorical_anomalies(model, sample_data, get_required_features):
    """
    モデルがカテゴリカル特徴量の異常値を適切に処理できるかテスト

    理由: カテゴリカル特徴量に未知のカテゴリが含まれる場合の挙動を検証することで、
    モデルのロバスト性を確保する
    """
    # テスト用データを準備
    X = sample_data.copy()
    if "Survived" in X.columns:
        X = X.drop("Survived", axis=1)

    # 必要な特徴量のみを選択
    required_features = get_required_features

    # 必要な特徴量を確保
    missing_cols = [col for col in required_features if col not in X.columns]
    if missing_cols:
        pytest.skip(f"必要な特徴量がデータに存在しません: {missing_cols}")

    X = X[required_features]

    # 性別に未知のカテゴリを挿入
    if "Sex" in X.columns:
        X.loc[0, "Sex"] = "unknown"

    # Pclassに範囲外の値を挿入
    if "Pclass" in X.columns:
        X.loc[1, "Pclass"] = 10

    try:
        # モデルが例外を発生させずに予測できるかテスト
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray), (
            "予測結果は numpy 配列である必要があります"
        )
    except Exception as e:
        pytest.fail(f"カテゴリ異常のあるデータに対して予測中にエラーが発生: {str(e)}")


def test_model_feature_importance_stability(model, sample_data):
    """
    モデルの特徴量重要度が安定しているかをテスト

    理由: 特徴量重要度が極端に偏っていたり、ランダムに見える場合、
    モデルが有意な学習をしていない可能性がある
    """
    # モデルがRandomForestClassifierであるか確認
    feature_importances = None

    if hasattr(model, "steps"):
        # パイプラインの場合
        classifier = None
        for name, step in model.steps:
            if name == "classifier" or hasattr(step, "feature_importances_"):
                classifier = step
                break

        if classifier is not None and hasattr(classifier, "feature_importances_"):
            feature_importances = classifier.feature_importances_

    elif hasattr(model, "feature_importances_"):
        # 単体のモデルの場合
        feature_importances = model.feature_importances_

    if feature_importances is None:
        pytest.skip("このモデルには特徴量重要度が実装されていません")

    # すべての特徴量重要度がゼロではないことを確認
    assert np.sum(feature_importances) > 0, "特徴量重要度がすべてゼロです"

    # 特徴量重要度が極端に偏っていないか確認（単一特徴に95%以上依存していないか）
    assert np.max(feature_importances) < 0.95, "モデルが単一特徴に過度に依存しています"


def test_model_prediction_distribution(model, sample_data, get_required_features):
    """
    モデルの予測分布が極端に偏っていないかテスト

    理由: 予測が一つのクラスに極端に偏っている場合、
    モデルが十分に識別できていない可能性がある
    """
    # テストデータを準備
    X = sample_data.copy()
    if "Survived" in X.columns:
        X = X.drop("Survived", axis=1)

    # 必要な特徴量のみを選択
    required_features = get_required_features

    # 必要な特徴量を確保
    missing_cols = [col for col in required_features if col not in X.columns]
    if missing_cols:
        pytest.skip(f"必要な特徴量がデータに存在しません: {missing_cols}")

    X = X[required_features]

    # 予測
    predictions = model.predict(X)

    # クラスの分布をカウント
    unique_classes, counts = np.unique(predictions, return_counts=True)

    # 少なくとも2つのクラスがあることを確認
    assert len(unique_classes) >= 2, "モデルは単一クラスのみを予測しています"

    # 予測が極端に偏っていないことを確認（99%以上が同じクラス予測でないこと）
    max_class_ratio = np.max(counts) / len(predictions)
    assert max_class_ratio < 0.99, (
        f"予測が単一クラスに極端に偏っています ({max_class_ratio:.2%})"
    )


def test_model_batch_consistency(model, sample_data, get_required_features):
    """
    異なるバッチサイズでの予測が一貫しているかテスト

    理由: 実運用では様々なバッチサイズで予測を行うため、
    バッチサイズに依存せず一貫した結果を返すことが重要
    """
    # テストデータを準備
    X = sample_data.copy()
    if "Survived" in X.columns:
        X = X.drop("Survived", axis=1)

    # 必要な特徴量のみを選択
    required_features = get_required_features

    # 必要な特徴量を確保
    missing_cols = [col for col in required_features if col not in X.columns]
    if missing_cols:
        pytest.skip(f"必要な特徴量がデータに存在しません: {missing_cols}")

    X = X[required_features]

    # 全データでの予測
    full_predictions = model.predict(X)

    # 小さなバッチに分けて予測
    batch_size = 10
    batched_predictions = []

    for i in range(0, len(X), batch_size):
        batch = X.iloc[i : i + batch_size]
        batch_pred = model.predict(batch)
        batched_predictions.extend(batch_pred)

    # 全データでの予測と、バッチ処理した予測が一致することを確認
    assert np.array_equal(full_predictions, batched_predictions), (
        "バッチサイズによって予測結果が変わっています"
    )


def test_model_fairness_across_groups(model, sample_data, get_required_features):
    """
    モデルが特定のグループ（例：性別）に対してバイアスがないかテスト

    理由: 公平性はAIモデルにおいて重要な倫理的課題であり、
    特定の属性に対して極端な予測傾向がないことを確認する
    """
    # データを準備
    data = sample_data.copy()

    # Survivedがない場合はスキップ
    if "Survived" not in data.columns:
        pytest.skip("目的変数(Survived)がデータに存在しません")

    # 性別ごとの実際の生存率を計算
    sex_survival = data.groupby("Sex")["Survived"].mean()

    # テストデータを準備（予測用）
    X = data.drop("Survived", axis=1)
    required_features = get_required_features

    # 必要な特徴量を確保
    missing_cols = [col for col in required_features if col not in X.columns]
    if missing_cols:
        pytest.skip(f"必要な特徴量がデータに存在しません: {missing_cols}")

    X = X[required_features]

    # 予測
    predictions = model.predict(X)

    # 予測結果をデータフレームに追加
    data["Predicted"] = predictions

    # 性別ごとの予測生存率を計算
    sex_predicted_survival = data.groupby("Sex")["Predicted"].mean()

    # 各グループでの実際の生存率と予測生存率の差の絶対値
    differences = abs(sex_survival - sex_predicted_survival)

    # 差が0.2（20%ポイント）を超えていないことを確認
    for sex, diff in differences.items():
        assert diff < 0.2, (
            f"{sex}グループでの予測と実際の生存率の差が大きすぎます: {diff:.2f}"
        )

    # 両グループ間での予測率の差が実際の差と大きく乖離していないことを確認
    actual_disparity = abs(sex_survival.max() - sex_survival.min())
    predicted_disparity = abs(
        sex_predicted_survival.max() - sex_predicted_survival.min()
    )

    disparity_diff = abs(actual_disparity - predicted_disparity)
    assert disparity_diff < 0.15, (
        f"グループ間の予測格差が実際と大きく異なります: {disparity_diff:.2f}"
    )


def test_model_performance_under_stress(model, sample_data, get_required_features):
    """
    モデルが大量のデータに対して効率的に動作するかテスト

    理由: 実運用環境では大量のデータを処理する可能性があり、
    計算効率とスケーラビリティが重要
    """
    import time

    # テストデータを準備
    X = sample_data.copy()
    if "Survived" in X.columns:
        X = X.drop("Survived", axis=1)

    # 必要な特徴量のみを選択
    required_features = get_required_features

    # 必要な特徴量を確保
    missing_cols = [col for col in required_features if col not in X.columns]
    if missing_cols:
        pytest.skip(f"必要な特徴量がデータに存在しません: {missing_cols}")

    X = X[required_features]

    # データを複製して大きなデータセットを作成（元のデータを10倍に）
    large_X = pd.concat([X] * 10, ignore_index=True)

    # 予測時間を計測
    start_time = time.time()
    model.predict(large_X)
    end_time = time.time()

    prediction_time = end_time - start_time

    # 1万件あたり5秒以内に処理できることを確認
    time_per_10k = prediction_time / len(large_X) * 10000
    assert time_per_10k < 5.0, (
        f"予測速度が遅すぎます: 10,000件あたり{time_per_10k:.2f}秒"
    )
