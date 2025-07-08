import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv('notebook/data/stud.csv')  # Load dataset
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)# dosya yolunu kontrol et
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)  # Save raw data
            
            logging.info("Train-test split completed successfully")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)  # Save train data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)  # Save test data

            logging.info("Ingestion of data is cpmleted")
            
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path # return paths

        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    
    #python -m src.components.data_ingestion



# 1️⃣ Çıktı dosya yollarını tanımlar (artifacts klasörü içinde)
# 2️⃣ Veriyi okuma ve train/test ayırma işlemlerini yapan sınıf
# 3️⃣ Veriyi CSV'den oku
# 4️⃣ artifacts klasörü yoksa oluştur
# 5️⃣ Ham veriyi kaydet
# 6️⃣ Veriyi %80 train, %20 test olarak ayır
# 7️⃣ Train ve test verilerini kaydet
# 8️⃣ Loglama
# 9️⃣ Dosya yollarını geri döndür
# 🔚 Dosya doğrudan çalıştırıldığında işlemi başlat