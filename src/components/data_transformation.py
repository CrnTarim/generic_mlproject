import sys
from dataclasses import dataclass
import os
import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from src.components.data_ingestion import DataIngestion


# Konfigürasyon: Preprocessor dosyasının kaydedileceği yol
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Veriyi dönüştürecek preprocessor objesini oluştur
    def get_data_transformer_object(self):
        try:
            # Sayısal ve kategorik sütunların manuel listesi
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Sayısal veriler için pipeline: eksik değer -> scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Kategorik veriler için pipeline: eksik değer -> OneHot -> scale (sparse yapıda mean alınmaz)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))  # OneHot sonrası sparse yapı için
                ]
            )

            # Her iki pipeline’ı birleştir
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # Eğitim ve test verisi üzerinde preprocessing uygula
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Verileri oku
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data read successfully")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Girdi (X) ve hedef (y) ayrımı
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on training and testing data")

            # Preprocessing uygula (fit + transform)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Girdi ile hedefi tekrar birleştir (model eğitimi için)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")

            # Preprocessor objesini dosyaya kaydet (ileride tekrar kullanılmak üzere)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Eğitim ve test verisi + kaydedilen objenin yolu return edilir
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # 1. Veri al
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2. Dönüştür
    transformer = DataTransformation()
    train_arr, test_arr, obj_path = transformer.initiate_data_transformation(train_path, test_path)

    print("✅ Train verisi şekli:", train_arr.shape)
    print("✅ Test verisi şekli:", test_arr.shape)
    print("✅ Preprocessor kaydedildi:", obj_path)







#manuel inputu değiştir
#dı ile yapmaya calıs


# 1️⃣ Preprocessor dosyasının yolunu tanımlar (artifacts/preprocessor.pkl)
# 2️⃣ Sayısal ve kategorik sütunları belirler
# 3️⃣ Her sütun tipi için ayrı Pipeline oluşturur
# 4️⃣ Sayısal verileri: impute (median) → scale
# 5️⃣ Kategorik verileri: impute (most_frequent) → one-hot → scale (with_mean=False)
# 6️⃣ Tüm pipeline’ları ColumnTransformer ile birleştirir
# 7️⃣ Train ve test CSV’lerini okur
# 8️⃣ math_score hedef sütununu ayırır
# 9️⃣ Preprocessor objesini fit_transform ve transform ile uygular
# 🔟 İşlenmiş X ile y’yi birleştirir (np.c_[])
# 1️⃣1️⃣ Preprocessor objesini .pkl dosyası olarak kaydeder
# 1️⃣2️⃣ Train, test ve obje yolunu return eder
