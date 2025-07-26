from sklearn.model_selection import train_test_split
from config.settings import TEST_SIZE

def preprocessing(X, Y, stratify=False) :

  x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, stratify=Y if stratify else None
    )

  print(f"Train shape : {x_train.shape}")
  print(f"Test shape : {x_test.shape}")

  return x_train, x_test, y_train, y_test