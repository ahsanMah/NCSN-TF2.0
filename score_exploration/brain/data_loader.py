
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def load(onehot=True, ood=True):

    fname="brain_train_data.p"
    with open(fname, mode='rb') as f:
        data = pickle.load(f)


    TRAIN_TEST_RATIO=0.9
    SEED = 42

    X = data["images"]
    y = data["labels"].reshape(-1,1)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                       train_size=TRAIN_TEST_RATIO,
                                                       stratify=y,
                                                       random_state=SEED
                                                       )
    if ood:
        y_train[y_train > 0] = 1
        y_test[y_test > 0] = 1
    
    
    class_weight = compute_class_weight("balanced", 
                                    classes=np.unique(np.flatten(y_train)),
                                    y=np.flatten(y_train))
    class_weight = {i:w for i,w in enumerate(class_weight)}
    print(class_weight)
    
    if onehot:
        enc = OneHotEncoder(sparse=False).fit(y_train)
        y_train = enc.transform(y_train)
        y_test = enc.transform(y_test)
    
    
    
    return X_train, X_test, y_train, y_test, class_weight
